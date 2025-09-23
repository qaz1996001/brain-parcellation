"""
Unit tests for database utilities.

Tests async database operations and connection pooling.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from code_ai.utils.database import (
    init_database,
    init_redis,
    get_db,
    get_redis,
    bulk_create_records,
    bulk_update_records,
    paginated_query,
    cached,
    transaction_context,
)


class TestDatabaseInitialization:
    """Test database initialization functions."""
    
    @pytest.mark.asyncio
    async def test_init_database(self):
        """Test database initialization."""
        with patch("code_ai.utils.database.create_async_engine") as mock_engine:
            mock_engine.return_value = MagicMock()
            
            await init_database("postgresql+asyncpg://user:pass@localhost/db")
            
            mock_engine.assert_called_once_with(
                "postgresql+asyncpg://user:pass@localhost/db",
                echo=False,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
                future=True,
            )
    
    @pytest.mark.asyncio
    async def test_init_redis(self):
        """Test Redis initialization."""
        with patch("code_ai.utils.database.redis.from_url") as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            await init_redis("redis://localhost:6379")
            
            mock_redis.assert_called_once_with(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True,
            )


class TestDatabaseSession:
    """Test database session management."""
    
    @pytest.mark.asyncio
    async def test_get_db_session(self):
        """Test get_db dependency injection."""
        mock_session = AsyncMock()
        
        with patch("code_ai.utils.database._async_session_maker") as mock_maker:
            mock_maker.return_value.__aenter__.return_value = mock_session
            mock_maker.return_value.__aexit__.return_value = None
            
            async for session in get_db():
                assert session == mock_session
                # Verify session methods are available
                assert hasattr(session, "commit")
                assert hasattr(session, "rollback")
    
    @pytest.mark.asyncio
    async def test_get_db_exception_handling(self):
        """Test get_db exception handling."""
        mock_session = AsyncMock()
        mock_session.commit.side_effect = Exception("Database error")
        
        with patch("code_ai.utils.database._async_session_maker") as mock_maker:
            mock_maker.return_value.__aenter__.return_value = mock_session
            mock_maker.return_value.__aexit__.return_value = None
            
            with pytest.raises(Exception, match="Database error"):
                async for session in get_db():
                    await session.commit()  # This will raise
            
            # Verify rollback was called
            mock_session.rollback.assert_called_once()


class TestBulkOperations:
    """Test bulk database operations."""
    
    @pytest.mark.asyncio
    async def test_bulk_create_records(self, async_db_session):
        """Test bulk create records."""
        # Mock records
        records = [
            {"id": 1, "name": "Record 1"},
            {"id": 2, "name": "Record 2"},
            {"id": 3, "name": "Record 3"},
        ]
        
        # Mock the model class
        class MockModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        with patch.object(async_db_session, "add_all") as mock_add_all:
            with patch.object(async_db_session, "commit") as mock_commit:
                result = await bulk_create_records(
                    async_db_session,
                    MockModel,
                    records
                )
                
                assert len(result) == 3
                mock_add_all.assert_called_once()
                mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bulk_create_records_batch_size(self, async_db_session):
        """Test bulk create with batch size."""
        # Create 25 records to test batching
        records = [{"id": i, "name": f"Record {i}"} for i in range(25)]
        
        class MockModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        add_all_calls = []
        
        async def mock_add_all(items):
            add_all_calls.append(len(items))
        
        with patch.object(async_db_session, "add_all", side_effect=mock_add_all):
            with patch.object(async_db_session, "commit"):
                await bulk_create_records(
                    async_db_session,
                    MockModel,
                    records,
                    batch_size=10
                )
                
                # Should be called 3 times: 10, 10, 5
                assert len(add_all_calls) == 3
                assert add_all_calls == [10, 10, 5]


class TestPaginatedQuery:
    """Test paginated query functionality."""
    
    @pytest.mark.asyncio
    async def test_paginated_query_basic(self, async_db_session):
        """Test basic paginated query."""
        # Mock query result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = ["item1", "item2"]
        
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 42
        
        with patch.object(async_db_session, "execute", side_effect=[mock_result, mock_count_result]):
            items, total = await paginated_query(
                async_db_session,
                "SELECT * FROM users",
                page=1,
                size=20
            )
            
            assert items == ["item1", "item2"]
            assert total == 42
    
    @pytest.mark.asyncio
    async def test_paginated_query_validation(self, async_db_session):
        """Test paginated query parameter validation."""
        # Test invalid page
        with pytest.raises(ValueError, match="Page must be >= 1"):
            await paginated_query(async_db_session, "SELECT 1", page=0)
        
        # Test invalid size
        with pytest.raises(ValueError, match="Size must be between 1 and 1000"):
            await paginated_query(async_db_session, "SELECT 1", page=1, size=0)
        
        with pytest.raises(ValueError, match="Size must be between 1 and 1000"):
            await paginated_query(async_db_session, "SELECT 1", page=1, size=1001)


class TestCacheDecorator:
    """Test cache decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_cached_decorator_basic(self, mock_redis):
        """Test basic cache decorator functionality."""
        call_count = 0
        
        @cached(expire=300)
        async def test_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call - should execute function
        result1 = await test_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call - should use cache
        result2 = await test_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
        # Different argument - should execute function
        result3 = await test_function(10)
        assert result3 == 20
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cached_decorator_key_prefix(self, mock_redis):
        """Test cache decorator with key prefix."""
        @cached(expire=300, key_prefix="user")
        async def get_user(user_id: int) -> dict:
            return {"id": user_id, "name": f"User {user_id}"}
        
        await get_user(123)
        
        # Check that key includes prefix
        keys = list(mock_redis.data.keys())
        assert len(keys) == 1
        assert keys[0].startswith("user:")
    
    @pytest.mark.asyncio
    async def test_cached_decorator_error_handling(self, mock_redis):
        """Test cache decorator error handling."""
        # Make Redis fail
        mock_redis.get = AsyncMock(side_effect=Exception("Redis error"))
        
        @cached(expire=300)
        async def test_function(x: int) -> int:
            return x * 2
        
        # Should still work even if cache fails
        result = await test_function(5)
        assert result == 10


class TestTransactionContext:
    """Test transaction context manager."""
    
    @pytest.mark.asyncio
    async def test_transaction_context_success(self, async_db_session):
        """Test successful transaction."""
        mock_transaction = AsyncMock()
        async_db_session.begin = AsyncMock(return_value=mock_transaction)
        
        async with transaction_context(async_db_session) as session:
            assert session == async_db_session
        
        # Verify transaction was started
        async_db_session.begin.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transaction_context_rollback(self, async_db_session):
        """Test transaction rollback on error."""
        mock_transaction = AsyncMock()
        async_db_session.begin = AsyncMock(return_value=mock_transaction)
        
        with pytest.raises(ValueError):
            async with transaction_context(async_db_session) as session:
                raise ValueError("Test error")
        
        # Verify rollback was called
        mock_transaction.__aexit__.assert_called_once()


class TestDatabaseUtilsDesign:
    """Test that database utils follow design principles."""
    
    def test_no_sync_operations(self):
        """Verify no synchronous database operations."""
        import inspect
        from code_ai.utils import database
        
        # Get all functions in the module
        functions = [
            obj for name, obj in inspect.getmembers(database)
            if inspect.isfunction(obj) and obj.__module__ == database.__name__
        ]
        
        # Check that all database operations are async
        for func in functions:
            # Skip internal functions and decorators
            if func.__name__.startswith("_") or func.__name__ in ["cached"]:
                continue
            
            # Database operations should be async
            if "db" in func.__name__ or "database" in func.__name__:
                assert inspect.iscoroutinefunction(func), f"{func.__name__} should be async"
    
    def test_connection_pooling_config(self):
        """Verify proper connection pooling configuration."""
        import inspect
        from code_ai.utils import database
        
        source = inspect.getsource(database.init_database)
        
        # Verify connection pool settings
        assert "pool_size=" in source
        assert "max_overflow=" in source
        assert "pool_pre_ping=True" in source
        assert "pool_recycle=" in source
