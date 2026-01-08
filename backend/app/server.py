# app/server.py
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
from starlette.middleware.cors import CORSMiddleware

from code_ai.pipeline.upload.schema import InferenceCompleteRequest
from .database import alchemy
from .routers import router

logger = logging.getLogger(__name__)

AI_APP_TITLE = os.getenv("AI_APP_TITLE", "SHH AI API")
AI_APP_DESCRIPTION = os.getenv("AI_APP_DESCRIPTION", "API FOR SHH AI")
AI_APP_VERSION = os.getenv("AI_APP_VERSION", "1.0.0")


async def init_cache():
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_USERNAME = os.getenv("REDIS_USERNAME")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_PORT = os.getenv("REDIS_PORT")
    REDIS_DB_FASTAPI_CACHE = os.getenv("REDIS_DB_FASTAPI_CACHE", 6)

    REDIS_URL = f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_FASTAPI_CACHE}"
    redis = aioredis.from_url(REDIS_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events for the application.

    Args:
        app: FastAPI application instance
    """

    # Startup event
    logger.info("Starting application ")
    await init_cache()
    # asyncio.create_task(task_scheduler.start())
    yield
    # Shutdown event
    logger.info("Shutting down application ")
    # await task_scheduler.stop()


app = FastAPI(
    title=AI_APP_TITLE,
    description=AI_APP_DESCRIPTION,
    version=AI_APP_VERSION,
    lifespan=lifespan,
    # root_path="/api/v1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")
alchemy.init_app(app)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """
    Health check endpoint with environment information.

    Requirement: 環境資訊 API 暴露 - Scenario: Health Check 包含環境

    Returns:
        Dictionary containing status and environment information
    """
    return {"status": "healthy"}


@app.post("/upload_json")
async def upload_json(inference_complete_request: InferenceCompleteRequest,
                      ) -> dict[str, str]:
    """

    """
    # json_dict = await request.json()
    logger.info(f'upload_json {inference_complete_request}')
    return {"status": "healthy"}


@app.post("/ai-inference/inference-complete")
async def inference_complete(inference_complete_request: InferenceCompleteRequest,
                             ) -> dict[str, str]:
    """

    """
    # json_dict = await request.json()
    logger.info(f'inference_complete_request {inference_complete_request}')
    return {"status": "healthy"}
