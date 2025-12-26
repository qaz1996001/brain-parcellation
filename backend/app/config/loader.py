"""
Configuration loader for backend application.

This module provides the ONLY entry point for environment variable reads.
Following the Configuration Centralization Pattern: 350+ → 2 os.getenv() calls.

Design Principles:
- Knuth: Precise type conversion, explicit validation, fail-safe defaults
- Linus (Data): Clean data structure eliminates special cases
- Fail-Safe Production: Never crashes, uses safe defaults
- Strict Testing: Fails fast on missing/invalid configuration

Usage:
    Production mode (default):
        config = load_backend_config_from_env(fail_safe=True)
        # Missing env vars → uses defaults, logs warnings

    Testing/CI mode:
        config = load_backend_config_from_env(fail_safe=False)
        # Missing env vars → raises ValueError
"""

import logging
import os
from pathlib import Path
from typing import Optional

from .models import APIConfig, BackendConfig, DatabaseConfig, PathConfig

logger = logging.getLogger(__name__)


# Default configuration for fail-safe production mode
# Ensures application can always start, even without environment variables
DEFAULT_CONFIG = BackendConfig(
    api=APIConfig(
        upload_data_url="http://localhost:8000/upload"
    ),
    paths=PathConfig(
        path_process=Path("/tmp/process"),
        path_json=Path("/tmp/json"),
        path_log=Path("/tmp/logs"),
        path_root=Path("/tmp/root"),
        path_rename_dicom=Path("/tmp/rename_dicom")
    ),
    database=DatabaseConfig(
        connection_string="postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom"
    )
)


def _get_env_or_default(
    key: str,
    default: Optional[str] = None,
    fail_safe: bool = True
) -> str:
    """
    Get environment variable with fail-safe/strict mode support.

    Args:
        key: Environment variable name
        default: Default value if variable not set
        fail_safe: True (production) = use default, False (testing) = raise error

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If fail_safe=False and variable not set
    """
    value = os.getenv(key)

    if value is None:
        if fail_safe:
            if default is None:
                logger.warning(
                    f"Environment variable '{key}' not set, using None. "
                    f"Consider setting this variable for production."
                )
            else:
                logger.warning(
                    f"Environment variable '{key}' not set, using default: {default}"
                )
            return default
        else:
            raise ValueError(
                f"Required environment variable '{key}' is not set. "
                f"Set this variable in your environment or .env file."
            )

    return value


def _convert_to_path(
    path_str: Optional[str],
    default: Path,
    fail_safe: bool = True
) -> Path:
    """
    Convert string to Path object with validation.

    Args:
        path_str: Path string from environment variable
        default: Default Path if path_str is None
        fail_safe: Whether to use default on error

    Returns:
        Path object

    Raises:
        TypeError: If fail_safe=False and path_str is invalid
    """
    if path_str is None:
        return default

    try:
        path = Path(path_str)
        return path
    except (TypeError, ValueError) as e:
        if fail_safe:
            logger.warning(
                f"Invalid path '{path_str}': {e}. Using default: {default}"
            )
            return default
        else:
            raise TypeError(
                f"Invalid path value '{path_str}': {e}"
            ) from e


def load_backend_config_from_env(fail_safe: bool = True) -> BackendConfig:
    """
    Load backend configuration from environment variables.

    This is the SINGLE SOURCE OF TRUTH for environment configuration.
    All os.getenv() calls are concentrated here.

    Operating Modes:
        Production (fail_safe=True):
            - Missing variables → use safe defaults
            - Invalid values → log warning, use defaults
            - Always returns valid config
            - Application always starts successfully

        Testing/CI (fail_safe=False):
            - Missing required variables → raises ValueError
            - Invalid values → raises TypeError/ValueError
            - Strict validation for early error detection

    Args:
        fail_safe: True for production (graceful), False for testing (strict)

    Returns:
        BackendConfig: Immutable configuration object

    Raises:
        ValueError: If fail_safe=False and required env var missing
        TypeError: If fail_safe=False and env var has invalid type

    Examples:
        >>> # Production mode
        >>> config = load_backend_config_from_env(fail_safe=True)
        >>> config.api.upload_data_url
        'http://localhost:8000/upload'

        >>> # Testing mode with all env vars set
        >>> os.environ['UPLOAD_DATA_API_URL'] = 'http://api.example.com'
        >>> config = load_backend_config_from_env(fail_safe=False)
        >>> config.api.upload_data_url
        'http://api.example.com'
    """
    # API Configuration
    upload_data_url = _get_env_or_default(
        "UPLOAD_DATA_API_URL",
        default=DEFAULT_CONFIG.api.upload_data_url,
        fail_safe=fail_safe
    )

    api_config = APIConfig(
        upload_data_url=upload_data_url
    )

    # Path Configuration
    path_process = _convert_to_path(
        _get_env_or_default("PATH_PROCESS", None, fail_safe),
        default=DEFAULT_CONFIG.paths.path_process,
        fail_safe=fail_safe
    )

    path_json = _convert_to_path(
        _get_env_or_default("PATH_JSON", None, fail_safe),
        default=DEFAULT_CONFIG.paths.path_json,
        fail_safe=fail_safe
    )

    path_log = _convert_to_path(
        _get_env_or_default("PATH_LOG", None, fail_safe),
        default=DEFAULT_CONFIG.paths.path_log,
        fail_safe=fail_safe
    )

    path_root = _convert_to_path(
        _get_env_or_default("PATH_ROOT", None, fail_safe),
        default=DEFAULT_CONFIG.paths.path_root,
        fail_safe=fail_safe
    )

    path_rename_dicom = _convert_to_path(
        _get_env_or_default("PATH_RENAME_DICOM", None, fail_safe),
        default=DEFAULT_CONFIG.paths.path_rename_dicom,
        fail_safe=fail_safe
    )

    paths_config = PathConfig(
        path_process=path_process,
        path_json=path_json,
        path_log=path_log,
        path_root=path_root,
        path_rename_dicom=path_rename_dicom
    )

    # Database Configuration
    connection_string = _get_env_or_default(
        "AI_APP_CONNECTION_STRING",
        default=DEFAULT_CONFIG.database.connection_string,
        fail_safe=fail_safe
    )

    database_config = DatabaseConfig(
        connection_string=connection_string
    )

    # Root Configuration
    config = BackendConfig(
        api=api_config,
        paths=paths_config,
        database=database_config
    )

    logger.info(
        f"Backend configuration loaded successfully (fail_safe={fail_safe})"
    )

    return config