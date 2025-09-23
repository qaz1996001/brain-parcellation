"""
Cache management module using Redis.

Provides async cache operations with proper connection management
and health checking.
"""
import pickle
from typing import Any, Optional

import redis.asyncio as redis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

from .config import get_settings

# Global Redis client
_redis_client: Optional[redis.Redis] = None

settings = get_settings()


async def init_cache(redis_url: Optional[str] = None) -> None:
    """
    Initialize Redis cache connection.
    
    Args:
        redis_url: Redis connection URL (uses settings if not provided)
    """
    global _redis_client
    
    url = redis_url or settings.get_redis_url(settings.redis_cache_db)
    _redis_client = redis.from_url(url, encoding="utf-8", decode_responses=True)
    
    # Initialize FastAPI cache
    FastAPICache.init(RedisBackend(_redis_client), prefix="fastapi-cache")


async def close_cache() -> None:
    """Close Redis cache connection."""
    global _redis_client
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


async def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    if not _redis_client:
        raise RuntimeError("Cache not initialized. Call init_cache() first.")
    return _redis_client


async def check_cache_health() -> bool:
    """Check if Redis cache is healthy."""
    try:
        if not _redis_client:
            return False
        
        # Ping Redis
        await _redis_client.ping()
        return True
    except Exception:
        return False


# Cache operations
async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache."""
    try:
        client = await get_redis()
        value = await client.get(key)
        
        if value:
            return pickle.loads(value.encode('latin-1'))
        return None
    except Exception as e:
        # Log error but don't raise - cache should be non-critical
        return None


async def cache_set(
    key: str,
    value: Any,
    expire: Optional[int] = None
) -> bool:
    """
    Set value in cache.
    
    Args:
        key: Cache key
        value: Value to cache
        expire: Expiration time in seconds
    
    Returns:
        True if successful, False otherwise
    """
    try:
        client = await get_redis()
        serialized = pickle.dumps(value).decode('latin-1')
        
        if expire:
            await client.setex(key, expire, serialized)
        else:
            await client.set(key, serialized)
        
        return True
    except Exception as e:
        # Log error but don't raise
        return False


async def cache_delete(key: str) -> bool:
    """Delete value from cache."""
    try:
        client = await get_redis()
        await client.delete(key)
        return True
    except Exception:
        return False


async def cache_clear_pattern(pattern: str) -> int:
    """
    Clear all keys matching pattern.
    
    Args:
        pattern: Redis key pattern (e.g., "user:*")
    
    Returns:
        Number of keys deleted
    """
    try:
        client = await get_redis()
        keys = await client.keys(pattern)
        
        if keys:
            return await client.delete(*keys)
        return 0
    except Exception:
        return 0
