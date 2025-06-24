# app/server.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis



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
    await init_cache()
    # asyncio.create_task(task_scheduler.start())
    yield
    # Shutdown event
    # await task_scheduler.stop()


app = FastAPI(
    title="SHH AI API",
    description="API for SHH AI",
    version="1.0.0",
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