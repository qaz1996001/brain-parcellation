"""
Pytest fixtures for contract testing.

Provides comprehensive test fixtures for validating backward compatibility
and behavioral equivalence between old and new implementations.
"""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest


# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """
    Provide a clean environment for testing.

    Saves current environment variables and restores them after test.
    """
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_env_backend(clean_env) -> dict:
    """
    Provide test environment variables for backend.

    Returns:
        dict: Test environment variables
    """
    test_vars = {
        "UPLOAD_DATA_API_URL": "http://test-api.example.com/upload",
        "UPLOAD_DATA_DICOM_SEG_URL": "http://test-orthanc.example.com:8042",
        "PATH_PROCESS": "/test/process",
        "PATH_JSON": "/test/json",
        "PATH_LOG": "/test/logs",
        "PATH_ROOT": "/test/root",
        "PATH_RENAME_DICOM": "/test/rename_dicom",
        "PATH_RAW_DICOM": "/test/raw_dicom",
        "PATH_RENAME_NIFTI": "/test/rename_nifti",
        "AI_APP_CONNECTION_STRING": "postgresql+asyncpg://test:test@localhost:5432/test_db",
    }
    os.environ.update(test_vars)
    return test_vars


@pytest.fixture
def test_env_code_ai(clean_env) -> dict:
    """
    Provide test environment variables for code_ai.

    Returns:
        dict: Test environment variables
    """
    test_vars = {
        "PATH_CODE": "/test/code",
        "PATH_PROCESS": "/test/process",
        "PATH_JSON": "/test/json",
        "PATH_LOG": "/test/logs",
        "PATH_SYNTHSEG": "/test/models/synthseg",
        "GPU_N": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_ENABLE_AUTO_MIXED_PRECISION": "0"
    }
    os.environ.update(test_vars)
    return test_vars


@pytest.fixture
def empty_env(clean_env) -> None:
    """
    Provide empty environment for testing default behavior.

    All environment variables are cleared to test fail-safe mode.
    """
    # Clear all config-related environment variables to test defaults
    config_vars = [
        "UPLOAD_DATA_API_URL", "UPLOAD_DATA_DICOM_SEG_URL",
        "PATH_PROCESS", "PATH_JSON", "PATH_LOG", "PATH_ROOT",
        "PATH_RENAME_DICOM", "PATH_RAW_DICOM", "PATH_RENAME_NIFTI",
        "AI_APP_CONNECTION_STRING", "PATH_CODE", "PATH_SYNTHSEG",
        "GPU_N", "TF_CPP_MIN_LOG_LEVEL", "TF_ENABLE_AUTO_MIXED_PRECISION",
    ]
    for var in config_vars:
        os.environ.pop(var, None)
