from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any, AsyncGenerator, Awaitable, Callable, List, Optional, Sequence, TypeVar, cast
from sqlalchemy import select
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from code_ai.utils.model import Base, FunboostConsumeResult

funboost_module = import_module("funboost")
FunctionResultStatus = cast(Any, getattr(funboost_module, "FunctionResultStatus"))
funboost_config_deafult = cast(Any, getattr(funboost_module, "funboost_config_deafult"))
Serialization = cast(
    Any,
    getattr(import_module("funboost.core.serialization"), "Serialization"),
)

logger = logging.getLogger(__name__)

ASYNC_DRIVER_MAP = {
    "postgresql": "asyncpg",
    "mysql": "aiomysql",
    "sqlite": "aiosqlite",
}

T = TypeVar("T")


def _build_async_database_url() -> str:
    raw_url = getattr(funboost_config_deafult.BrokerConnConfig, "SQLACHEMY_ENGINE_URL", "")
    if not raw_url:
        raw_url = "sqlite:///./funboost.db"

    parsed_url = make_url(raw_url)
    drivername = parsed_url.drivername
    if "+" in drivername:
        dialect, driver = drivername.split("+", 1)
    else:
        dialect, driver = drivername, ""

    if driver in {"asyncpg", "aiomysql", "aiosqlite"}:
        return str(parsed_url)

    async_driver = ASYNC_DRIVER_MAP.get(dialect)
    if not async_driver:
        raise ValueError(f"Unsupported database dialect: {dialect}")

    return str(parsed_url.set(drivername=f"{dialect}+{async_driver}"))


DATABASE_URL = _build_async_database_url()

engine_kwargs: dict[str, Any] = {
    "echo": False,
    "future": True,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
}

if DATABASE_URL.startswith("sqlite+aiosqlite"):
    engine_kwargs["connect_args"] = {"timeout": 30}
else:
    engine_kwargs["pool_size"] = 20
    engine_kwargs["max_overflow"] = 0

engine = create_async_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


def _run_sync(coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Execute an async callable even when the current thread already owns an event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    result: Optional[T] = None
    error: Optional[BaseException] = None

    def runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro_factory())
        except BaseException as exc:  # pragma: no cover - propagate after join
            error = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error
    return cast(T, result)


async def _initialize_schema() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


_run_sync(_initialize_schema)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an AsyncSession with proper transaction handling."""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def _serialize_status(function_result_status: FunctionResultStatus) -> dict[str, Any]:
    status_dict = function_result_status.get_status_dict()
    payload: dict[str, Any] = {}
    for key, value in status_dict.items():
        if isinstance(value, (dict, list)):
            payload[key] = Serialization.to_json_str(value)
            continue
        payload[key] = value
    return payload


async def save_result_status(function_result_status: FunctionResultStatus) -> None:
    """Persist a single funboost result using an asynchronous session."""
    payload = _serialize_status(function_result_status)
    model = FunboostConsumeResult(**payload)
    async with get_session() as session:
        session.add(model)
        try:
            await session.commit()
        except SQLAlchemyError as exc:
            logger.exception("Failed to persist funboost result: %s", exc)
            await session.rollback()
            raise


async def save_result_status_batch(status_list: Sequence[FunctionResultStatus]) -> None:
    """Persist multiple funboost results in a single transaction."""
    if not status_list:
        return
    models = [FunboostConsumeResult(**_serialize_status(item)) for item in status_list]
    async with get_session() as session:
        session.add_all(models)
        try:
            await session.commit()
        except SQLAlchemyError as exc:
            logger.exception("Failed to persist funboost batch: %s", exc)
            await session.rollback()
            raise


def save_result_status_to_sqlalchemy(function_result_status: FunctionResultStatus) -> None:
    """Sync wrapper required by funboost hooks."""
    _run_sync(lambda: save_result_status(function_result_status))


def save_result_status_to_sqlalchemy_by_batch(
    function_result_status_list: Sequence[FunctionResultStatus],
) -> None:
    _run_sync(lambda: save_result_status_batch(list(function_result_status_list)))


async def query_result_status(queue_name: str, limit: int = 100) -> List[FunboostConsumeResult]:
    """Return the latest consume results for the specified queue."""
    async with get_session() as session:
        stmt = (
            select(FunboostConsumeResult)
            .where(FunboostConsumeResult.queue_name == queue_name)
            .order_by(FunboostConsumeResult.insert_time.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(result.scalars())


def query_result_status_to_sqlalchemy(queue_name: str, limit: int = 100) -> List[FunboostConsumeResult]:
    return _run_sync(lambda: query_result_status(queue_name, limit))
