from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionmaker
from advanced_alchemy.extensions.fastapi import service


class BaseRepositoryService(service.SQLAlchemyAsyncRepositoryService):
    """增強的 Service 基類，提供統一的 Session 管理"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 從 repository 獲取 session factory
        self._session_factory = self.repository.session_factory
        self._session_manager = SessionManager(self._session_factory)

    @property
    def session_manager(self) -> SessionManager:
        """獲取 session 管理器"""
        return self._session_manager

    async def execute_in_transaction(self, func, *args, session: Optional[AsyncSession] = None, **kwargs):
        """
        在交易中執行函數
        如果提供了 session，則使用該 session
        否則創建新的 session
        """
        async with self.session_manager.use_session(session) as db_session:
            return await func(*args, session=db_session, **kwargs)
