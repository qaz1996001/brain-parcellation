# app/server.py
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

# # Environment configuration support
# from backend.app.config import get_environment, get_config
#
# # Load environment configuration at module level (immutable after startup)
# ENVIRONMENT = get_environment()
# CONFIG = get_config()
#
# # Configure logging based on environment
# logging.basicConfig(
#     level=getattr(logging, CONFIG["log_level"]),
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
logger = logging.getLogger(__name__)
#
# # Log environment information for audit trail
# logger.info(f"Environment: {ENVIRONMENT}")
# logger.info(f"Log level: {CONFIG['log_level']}")
# logger.info(f"Data root: {CONFIG['data_root']}")


AI_APP_TITLE = os.getenv("AI_APP_TITLE",'SHH AI API')
AI_APP_DESCRIPTION = os.getenv("AI_APP_DESCRIPTION",'API FOR SHH AI')
AI_APP_VERSION = os.getenv("AI_APP_VERSION",'1.0.0')


async def init_cache():
    REDIS_HOST     = os.getenv("REDIS_HOST")
    REDIS_USERNAME = os.getenv("REDIS_USERNAME")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_PORT     = os.getenv("REDIS_PORT")
    REDIS_DB       = os.getenv("REDIS_DB_FASTAPI_CACHE",6)

    REDIS_URL = f'redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    redis = aioredis.from_url(REDIS_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

from .routers import router
from .database import alchemy


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events for the application.

    Args:
        app: FastAPI application instance
    """

    # Startup event
    logger.info(f"Starting application in {ENVIRONMENT} environment")
    await init_cache()
    # asyncio.create_task(task_scheduler.start())
    yield
    # Shutdown event
    logger.info(f"Shutting down application in {ENVIRONMENT} environment")
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
    CORSMiddleware,
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
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "log_level": CONFIG["log_level"]
    }