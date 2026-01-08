# app/inference/deps.py
"""Dependency injection for inference module.

Provides Redis client and cache utilities for series-level inference.

Linus: "Keep it simple" - Redis client as singleton dependency.
"""

import os
import logging
from typing import Optional
from functools import lru_cache

from redis import asyncio as aioredis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@lru_cache
def get_redis_url() -> str:
    """Get Redis URL from environment.

    Returns:
        str: Redis URL for inference cache
    """
    host = os.getenv("REDIS_HOST", "localhost")
    username = os.getenv("REDIS_USERNAME", "")
    password = os.getenv("REDIS_PASSWORD", "")
    port = os.getenv("REDIS_PORT", "6379")
    # Use dedicated DB for inference cache (different from FastAPI cache)
    db = os.getenv("REDIS_DB_INFERENCE_CACHE", "7")

    if username and password:
        return f"redis://{username}:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


# Global Redis client (singleton)
_redis_client: Optional[Redis] = None


async def get_redis_client() -> Redis:
    """Get async Redis client for inference cache.

    Returns:
        Redis: Async Redis client

    Raises:
        ConnectionError: If Redis connection fails
    """
    global _redis_client

    if _redis_client is None:
        redis_url = get_redis_url()
        try:
            _redis_client = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await _redis_client.ping()
            logger.info("Connected to Redis for inference cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}") from e

    return _redis_client


async def close_redis_client() -> None:
    """Close Redis client connection.

    Call during application shutdown.
    """
    global _redis_client

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis client closed")


# Cache key prefix for inference cache
CACHE_KEY_PREFIX = "inference_cache:"

# Default TTL (24 hours in seconds)
DEFAULT_CACHE_TTL = 86400
