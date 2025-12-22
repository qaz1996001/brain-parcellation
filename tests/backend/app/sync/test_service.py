from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from backend.app.sync.service import DCOPEventDicomService


class SpySessionManager:
    def __init__(self):
        self.called = False

    @asynccontextmanager
    async def get_session(self):
        self.called = True
        yield None


class StaticSessionManager:
    def __init__(self, session):
        self._session = session

    @asynccontextmanager
    async def get_session(self):
        yield self._session


class DummySession:
    def __init__(self):
        self.execute = AsyncMock()
        self.committed = False
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_schedule_new_studies_deduplicates_and_calls_once():
    service = object.__new__(DCOPEventDicomService)
    service.add_study_new = AsyncMock(return_value=["ok"])

    result = await service.schedule_new_studies(["a", "a", None])

    assert result == ["ok"]
    service.add_study_new.assert_awaited_once_with(data_list=["a"])


@pytest.mark.asyncio
async def test_link_prev_study_guard_closes_early():
    service = object.__new__(DCOPEventDicomService)
    spy_manager = SpySessionManager()
    service._session_manager = spy_manager

    await service.link_prev_study(None, "prev")

    assert spy_manager.called is False


@pytest.mark.asyncio
async def test_link_prev_study_persists_when_valid():
    service = object.__new__(DCOPEventDicomService)
    dummy_session = DummySession()
    service._session_manager = StaticSessionManager(dummy_session)

    await service.link_prev_study("current", "previous")

    assert dummy_session.execute.await_count == 1
    assert dummy_session.committed is True
    assert dummy_session.added[0].study_uid == "current"
    assert dummy_session.added[0].prev_study_uid == "previous"

