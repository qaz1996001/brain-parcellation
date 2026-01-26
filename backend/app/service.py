"""
基礎服務層 - 統一的資料庫會話管理和服務基類。

此模組提供兩個核心組件：
1. SessionManager: 資料庫會話管理器，支援嵌套事務和自動清理
2. BaseRepositoryService: 增強的服務基類，提供統一的會話管理

核心功能
--------
- 非同步會話管理: 自動建立、使用和清理資料庫會話
- 事務支援: 支援常規和嵌套事務（savepoints）
- 重試機制: 自動重試暫時性資料庫錯誤
- 批次操作: 在單一事務中執行多個操作
- 資源清理: 應用程式關閉時自動清理所有會話

設計原則
--------
- 資源安全: 確保會話正確關閉，避免連接洩漏
- 事務一致性: 支援原子性操作和自動回滾
- 錯誤恢復: 自動重試暫時性錯誤
- 靈活性: 支援使用現有會話或建立新會話

Classes
-------
SessionManager
    資料庫會話管理器，提供會話建立、事務管理和資源清理。
    
BaseRepositoryService
    增強的服務基類，擴展 SQLAlchemyAsyncRepositoryService，
    提供統一的會話管理能力。

Notes
-----
此模組建立在 Advanced Alchemy 之上，提供額外的會話管理功能。
所有服務類應繼承 BaseRepositoryService 以獲得統一的會話管理。

Examples
--------
使用 SessionManager：

>>> from backend.app.service import SessionManager
>>> from sqlalchemy.ext.asyncio import async_sessionmaker
>>> 
>>> session_factory = async_sessionmaker(...)
>>> manager = SessionManager(session_factory)
>>> 
>>> async with manager.get_session() as session:
>>>     result = await session.execute(query)

繼承 BaseRepositoryService：

>>> from backend.app.service import BaseRepositoryService
>>> from backend.app.model import MyModel
>>> 
>>> class MyService(BaseRepositoryService[MyModel]):
>>>     async def my_method(self):
>>>         async with self.session_manager.get_session() as session:
>>>             # 使用會話進行操作
>>>             pass

See Also
--------
backend.app.database : 資料庫配置
advanced_alchemy.extensions.fastapi.service : Advanced Alchemy 服務基類
sqlalchemy.ext.asyncio : SQLAlchemy 非同步擴展
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, TypeVar, Generic, AsyncGenerator, Callable, Any
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncSessionTransaction,
    async_sessionmaker,
)
from sqlalchemy.exc import SQLAlchemyError
from advanced_alchemy.extensions.fastapi import service
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SessionManager:
    """
    資料庫會話管理器 - 提供統一的會話生命週期管理。
    
    此類提供集中化的 SQLAlchemy 非同步會話管理，確保：
    - 正確的事務處理
    - 錯誤恢復機制
    - 資源自動清理
    - 嵌套事務支援
    
    Attributes
    ----------
    _session_factory : async_sessionmaker
        會話工廠，用於建立新的資料庫會話。
    _active_sessions : dict[int, AsyncSession]
        當前活躍的會話字典，鍵為會話 ID，值為會話實例。
    _lock : asyncio.Lock
        非同步鎖，用於保護會話字典的並發訪問。
    
    Methods
    -------
    get_session()
        建立並管理新的資料庫會話，自動清理。
    use_session(session)
        使用現有會話或建立新會話。
    transaction(session, nested)
        在會話中建立事務（支援嵌套）。
    execute_with_retry(func, *args, **kwargs)
        執行函數並在暫時性錯誤時自動重試。
    close_all_sessions()
        關閉所有活躍的會話。
    get_active_session_count()
        獲取當前活躍會話數量。
    
    Notes
    -----
    會話追蹤：
        所有建立的會話都會被追蹤，確保在應用程式關閉時
        正確清理，避免連接洩漏。
    
    線程安全：
        使用 asyncio.Lock 保護會話字典的並發訪問，
        確保多個協程同時建立會話時的安全性。
    
    Examples
    --------
    基本使用：
    
    >>> from sqlalchemy.ext.asyncio import async_sessionmaker
    >>> manager = SessionManager(async_sessionmaker(...))
    >>> 
    >>> async with manager.get_session() as session:
    >>>     result = await session.execute(query)
    >>>     await session.commit()
    
    使用現有會話：

    >>> async with manager.use_session(existing_session) as session:
    >>>     # 如果提供了會話，使用它；否則建立新的
    >>>     await session.execute(query)
    
    嵌套事務：
    
    >>> async with manager.get_session() as session:
    >>>     async with manager.transaction(session, nested=True) as trans:
    >>>         # 嵌套事務（savepoint）
    >>>         await session.execute(query)
    >>>         # 如果出錯，只回滾此嵌套事務
    """

    def __init__(self, session_factory: async_sessionmaker):
        """
        Initialize the SessionManager with a session factory.

        Args:
            session_factory: An async session factory for creating new sessions
        """
        self._session_factory = session_factory
        self._active_sessions: dict[int, AsyncSession] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Create and manage a new database session with automatic cleanup.

        This method creates a new session and ensures it's properly closed
        after use, even if an error occurs.

        Yields:
            AsyncSession: A new database session

        Example:
            async with session_manager.get_session() as session:
                # Use session for database operations
                result = await session.execute(query)
        """
        session = self._session_factory()
        session_id = id(session)

        async with self._lock:
            self._active_sessions[session_id] = session

        try:
            yield session
        finally:
            async with self._lock:
                self._active_sessions.pop(session_id, None)

            if session.is_active:
                await session.close()

    @asynccontextmanager
    async def use_session(
        self, session: Optional[AsyncSession] = None
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Use an existing session or create a new one if none provided.

        This method allows for flexible session management where you can
        either provide an existing session or let the manager create one.

        Args:
            session: Optional existing session to use

        Yields:
            AsyncSession: Either the provided session or a new one

        Example:
            async with session_manager.use_session(existing_session) as session:
                # If existing_session is provided, it will be used
                # Otherwise, a new session is created
                await session.execute(query)
        """
        if session is not None:
            yield session
        else:
            async with self.get_session() as new_session:
                yield new_session

    @asynccontextmanager
    async def transaction(
        self, session: AsyncSession, nested: bool = True
    ) -> AsyncGenerator[AsyncSessionTransaction, None]:
        """
        Create a transaction within the given session.

        Supports both regular and nested transactions (savepoints).

        Args:
            session: The session to create the transaction in
            nested: Whether to create a nested transaction (savepoint)

        Yields:
            AsyncSessionTransaction: The transaction object

        Example:
            async with session_manager.transaction(session) as trans:
                # Perform operations within transaction
                await session.execute(insert_query)
                # Transaction automatically commits if no exception
        """
        if nested and session.in_transaction():
            # Create a savepoint for nested transaction
            async with session.begin_nested() as transaction:
                try:
                    yield transaction
                except Exception:
                    await transaction.rollback()
                    raise
        else:
            # Create a regular transaction
            async with session.begin() as transaction:
                try:
                    yield transaction
                except Exception:
                    await transaction.rollback()
                    raise

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        session: Optional[AsyncSession] = None,
        **kwargs,
    ) -> Any:
        """
        Execute a function with automatic retry on transient failures.

        This method is useful for handling temporary database issues like
        connection problems or deadlocks.

        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            session: Optional session to use
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function execution

        Raises:
            The last exception if all retries fail
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                async with self.use_session(session) as db_session:
                    return await func(*args, session=db_session, **kwargs)

            except SQLAlchemyError as e:
                last_exception = e

                if attempt < max_retries - 1:
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(
                        f"Database operation failed after {max_retries} attempts: {e}"
                    )

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Database operation failed but no exception was captured")

    async def close_all_sessions(self) -> None:
        """
        Close all active sessions managed by this SessionManager.

        This method should be called during application shutdown to ensure
        all database connections are properly closed.
        """
        async with self._lock:
            sessions = list(self._active_sessions.values())
            self._active_sessions.clear()

        for session in sessions:
            try:
                if session.is_active:
                    await session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")

    def get_active_session_count(self) -> int:
        """
        Get the number of currently active sessions.

        Returns:
            The count of active sessions
        """
        return len(self._active_sessions)


ModelT = TypeVar("ModelT")


class BaseRepositoryService(
    service.SQLAlchemyAsyncRepositoryService[ModelT], Generic[ModelT]  # type: ignore[type-arg]
):
    """
    增強的服務基類 - 提供統一的會話管理能力。
    
    此基類擴展 SQLAlchemyAsyncRepositoryService，添加額外的會話管理功能，
    使跨多個儲存庫調用的複雜事務操作更容易處理。
    
    Parameters
    ----------
    **kwargs
        傳遞給父服務類的參數。
    
    Attributes
    ----------
    session_manager : SessionManager
        會話管理器實例，提供會話建立和事務管理。
    _session_factory : async_sessionmaker
        會話工廠，從儲存庫中提取或建立。
    
    Methods
    -------
    execute_in_transaction(func, *args, **kwargs)
        在事務中執行函數，確保原子性。
    execute_batch_operations(operations, session, stop_on_error)
        在單一事務中執行多個操作。
    with_new_session(func, *args, **kwargs)
        使用全新的會話執行函數。
    cleanup()
        清理服務使用的資源。
    get_session_stats()
        獲取會話使用統計。
    
    Notes
    -----
    會話工廠提取：
        此類會自動從 Advanced Alchemy 儲存庫中提取會話工廠。
        支援多種提取策略，確保兼容性。
    
    事務管理：
        所有資料庫操作都應通過此類的方法進行，以確保
        正確的事務處理和資源清理。
    
    Examples
    --------
    基本繼承：
    
    >>> from backend.app.service import BaseRepositoryService
    >>> from backend.app.model import UserModel
    >>> 
    >>> class UserService(BaseRepositoryService[UserModel]):
    >>>     async def create_user_with_profile(self, name, email):
    >>>         async def _create(session):
    >>>             user = await self.create({"name": name}, session=session)
    >>>             profile = await profile_repo.create(
    >>>                 {"user_id": user.id, "email": email},
    >>>                 session=session
    >>>             )
    >>>             return user, profile
    >>>         
    >>>         return await self.execute_in_transaction(_create)
    
    批次操作：
    
    >>> operations = [
    >>>     (repo.create, (data1,), {}),
    >>>     (repo.update, (id1, data2), {}),
    >>>     (repo.delete, (id2,), {})
    >>> ]
    >>> results = await service.execute_batch_operations(operations)
    
    See Also
    --------
    SessionManager : 會話管理器
    advanced_alchemy.extensions.fastapi.service.SQLAlchemyAsyncRepositoryService : 父類
    """

    def __init__(self, **kwargs):
        """
        Initialize the service with enhanced session management.

        Args:
            **kwargs: Arguments passed to the parent service class
        """
        super().__init__(**kwargs)

        # The session in repository context is already an async session that can be used
        # We need to get the session factory that creates these sessions
        # In advanced_alchemy, the repository has a session attribute which is the current session
        # We need to find the factory that creates such sessions

        # Try to get the session factory from various possible locations
        session_factory = None

        # First, check if repository has a direct session_factory attribute
        if hasattr(self.repository, "_sessionmaker"):
            session_factory = self.repository._sessionmaker
        elif hasattr(self.repository, "session_factory"):
            session_factory = self.repository.session_factory
        else:
            # If not found directly, we need to create one from the engine
            # The repository.session is an AsyncSession instance
            # We can get the bind (engine) from it and create a sessionmaker
            if hasattr(self.repository, "session") and hasattr(
                self.repository.session, "bind"
            ):
                from sqlalchemy.ext.asyncio import async_sessionmaker

                engine = self.repository.session.bind

                # Create a new session factory with the same engine
                session_factory = async_sessionmaker(
                    bind=engine, class_=AsyncSession, expire_on_commit=False
                )
            else:
                # Last resort: try to get from the class attribute if available
                if hasattr(self.repository.__class__, "session_factory"):
                    session_factory = self.repository.__class__.session_factory

        if session_factory is None:
            raise AttributeError(
                "Unable to find or create session factory. "
                "Please ensure your repository is properly configured with advanced_alchemy."
            )

        self._session_factory = session_factory
        self._session_manager = SessionManager(self._session_factory)

    @property
    def session_manager(self) -> SessionManager:
        """
        Get the session manager instance.

        Returns:
            SessionManager: The session manager for this service
        """
        return self._session_manager

    async def execute_in_transaction(
        self, func: Callable, *args, session: Optional[AsyncSession] = None, **kwargs
    ) -> Any:
        """
        Execute a function within a database transaction.

        If a session is provided, it will be used. Otherwise, a new session
        and transaction will be created. This ensures all database operations
        within the function are atomic.

        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            session: Optional existing session to use
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function execution

        Example:
            async def create_user_with_profile(name, email, session):
                user = await user_repo.create({"name": name}, session=session)
                profile = await profile_repo.create({"user_id": user.id, "email": email}, session=session)
                return user, profile

            user, profile = await service.execute_in_transaction(
                create_user_with_profile,
                "John Doe",
                "john@example.com"
            )
        """
        async with self.session_manager.use_session(session) as db_session:
            async with self.session_manager.transaction(db_session):
                return await func(*args, session=db_session, **kwargs)

    async def execute_batch_operations(
        self,
        operations: list[tuple[Callable, tuple, dict]],
        session: Optional[AsyncSession] = None,
        stop_on_error: bool = True,
    ) -> list[tuple[bool, Any]]:
        """
        Execute multiple operations within a single transaction.

        This method is useful for performing multiple related database
        operations that should all succeed or all fail together.

        Args:
            operations: List of (function, args, kwargs) tuples to execute
            session: Optional existing session to use
            stop_on_error: Whether to stop and rollback on first error

        Returns:
            List of (success, result) tuples for each operation

        Example:
            operations = [
                (repo.create, (data1,), {}),
                (repo.update, (id1, data2), {}),
                (repo.delete, (id2,), {})
            ]
            results = await service.execute_batch_operations(operations)
        """
        results = []

        async with self.session_manager.use_session(session) as db_session:
            async with self.session_manager.transaction(db_session):
                for func, args, kwargs in operations:
                    try:
                        # Ensure session is passed to the function
                        if "session" not in kwargs:
                            kwargs["session"] = db_session

                        result = await func(*args, **kwargs)
                        results.append((True, result))

                    except Exception as e:
                        results.append((False, e))

                        if stop_on_error:
                            logger.error(f"Batch operation failed: {e}")
                            raise

        return results

    async def with_new_session(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with a completely new session.

        This is useful when you need to ensure a function runs in isolation
        from any existing session context.

        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function execution
        """
        async with self.session_manager.get_session() as session:
            return await func(*args, session=session, **kwargs)

    async def cleanup(self) -> None:
        """
        Cleanup resources used by this service.

        This method should be called when the service is being shut down
        to ensure all database connections are properly closed.
        """
        await self.session_manager.close_all_sessions()

    def get_session_stats(self) -> dict[str, Any]:
        """
        Get statistics about session usage.

        Returns:
            Dictionary containing session statistics
        """
        return {
            "active_sessions": self.session_manager.get_active_session_count(),
            "session_factory": str(self._session_factory),
        }
