"""
Database configuration and management using SQLAlchemy async.

Follows best practices for async database operations with proper
connection pooling and session management.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from .config import get_settings

# Global engine and session maker
_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[async_sessionmaker] = None

# Base for models
Base = declarative_base()

settings = get_settings()


async def init_database(database_url: Optional[str] = None) -> None:
    """
    Initialize database engine and session maker.
    
    Args:
        database_url: Database connection URL (uses settings if not provided)
    """
    global _engine, _async_session_maker
    
    url = database_url or settings.database_url
    
    # Create async engine with optimized settings
    _engine = create_async_engine(
        url,
        echo=settings.debug,  # SQL logging in debug mode
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=settings.database_pool_recycle,
        future=True,
    )
    
    # Create session maker
    _async_session_maker = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Create tables if needed (development only)
    if settings.debug:
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


async def close_database() -> None:
    """Close database connections."""
    global _engine
    
    if _engine:
        await _engine.dispose()
        _engine = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        Database session with automatic cleanup
    """
    if not _async_session_maker:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database session.
    
    Use this for non-FastAPI contexts where dependency injection isn't available.
    """
    if not _async_session_maker:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_health() -> bool:
    """Check if database is healthy."""
    try:
        if not _engine:
            return False
        
        # Execute simple query
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        
        return True
    except Exception:
        return False


# Transaction helpers
@asynccontextmanager
async def transaction(session: AsyncSession):
    """
    Explicit transaction context manager.
    
    Usage:
        async with transaction(session):
            # Your transactional code here
            pass
    """
    async with session.begin():
        yield session


# Bulk operations helpers
async def bulk_insert(
    session: AsyncSession,
    model_class,
    records: list,
    batch_size: int = 1000
) -> None:
    """
    Bulk insert records with batching.
    
    Args:
        session: Database session
        model_class: SQLAlchemy model class
        records: List of dictionaries with record data
        batch_size: Number of records per batch
    """
    # Process in batches
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        
        # Create model instances
        instances = [model_class(**record) for record in batch]
        
        # Add to session and flush
        session.add_all(instances)
        await session.flush()


async def bulk_update(
    session: AsyncSession,
    model_class,
    updates: list,
    batch_size: int = 1000
) -> None:
    """
    Bulk update records with batching.
    
    Args:
        session: Database session
        model_class: SQLAlchemy model class
        updates: List of tuples (id, update_dict)
        batch_size: Number of records per batch
    """
    # Process in batches
    for i in range(0, len(updates), batch_size):
        batch = updates[i:i + batch_size]
        
        for record_id, update_data in batch:
            await session.execute(
                model_class.__table__.update()
                .where(model_class.id == record_id)
                .values(**update_data)
            )
        
        await session.flush()
