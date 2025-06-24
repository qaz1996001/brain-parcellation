import asyncio
from contextlib import asynccontextmanager
from typing import Optional, TypeVar, Generic, Type, AsyncGenerator, Callable, Any
from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionTransaction, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from advanced_alchemy.extensions.fastapi import service
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SessionManager:
    """
    Manages database sessions with support for nested transactions and proper cleanup.

    This class provides a centralized way to manage SQLAlchemy async sessions,
    ensuring proper transaction handling, error recovery, and resource cleanup.
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
            self,
            session: Optional[AsyncSession] = None
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
            self,
            session: AsyncSession,
            nested: bool = True
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
            **kwargs
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

        raise last_exception

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


ModelT = TypeVar('ModelT')


class BaseRepositoryService(service.SQLAlchemyAsyncRepositoryService[ModelT], Generic[ModelT]):
    """
    Enhanced Service base class providing unified Session management.

    This base class extends the SQLAlchemyAsyncRepositoryService with
    additional session management capabilities, making it easier to handle
    complex transactional operations across multiple repository calls.
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
        if hasattr(self.repository, '_sessionmaker'):
            session_factory = self.repository._sessionmaker
        elif hasattr(self.repository, 'session_factory'):
            session_factory = self.repository.session_factory
        else:
            # If not found directly, we need to create one from the engine
            # The repository.session is an AsyncSession instance
            # We can get the bind (engine) from it and create a sessionmaker
            if hasattr(self.repository, 'session') and hasattr(self.repository.session, 'bind'):
                from sqlalchemy.ext.asyncio import async_sessionmaker
                engine = self.repository.session.bind

                # Create a new session factory with the same engine
                session_factory = async_sessionmaker(
                    bind=engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
            else:
                # Last resort: try to get from the class attribute if available
                if hasattr(self.repository.__class__, 'session_factory'):
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
            self,
            func: Callable,
            *args,
            session: Optional[AsyncSession] = None,
            **kwargs
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
            stop_on_error: bool = True
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
                        if 'session' not in kwargs:
                            kwargs['session'] = db_session

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
            "session_factory": str(self._session_factory)
        }