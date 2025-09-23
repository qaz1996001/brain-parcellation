"""
Global pytest fixtures and configuration.
"""
import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Set test environment
os.environ["TESTING"] = "1"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"

# Import after setting environment variables
from backend.app.server import app
from code_ai.utils.database import Base


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_db_engine():
    """Create async database engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture
async def async_db_session(async_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for testing."""
    async_session_maker = sessionmaker(
        async_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def test_client() -> TestClient:
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_dicom_path() -> Path:
    """Return path to sample DICOM file."""
    return Path(__file__).parent / "fixtures" / "sample.dcm"


@pytest.fixture
def sample_nifti_path() -> Path:
    """Return path to sample NIfTI file."""
    return Path(__file__).parent / "fixtures" / "sample.nii.gz"


@pytest.fixture
def pipeline_config() -> dict:
    """Return sample pipeline configuration."""
    return {
        "segmentation": {
            "model_path": "/models/synthseg_2.0.h5",
            "gpu_memory_limit": 4.0,
            "batch_size": 1,
        },
        "parcellation": {
            "depth_number": 5,
            "atlas_path": "/models/atlas",
        },
        "wmh_detection": {
            "threshold": 0.5,
            "min_size": 10,
        },
    }


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis client for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        async def get(self, key):
            return self.data.get(key)
        
        async def setex(self, key, expire, value):
            self.data[key] = value
            return True
        
        async def delete(self, key):
            if key in self.data:
                del self.data[key]
            return True
    
    mock = MockRedis()
    monkeypatch.setattr("code_ai.utils.database._redis_client", mock)
    return mock
