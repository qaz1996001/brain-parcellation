"""
Configuration loader for code_ai application.

This module provides the ONLY entry point for environment variable reads.
Following the Configuration Centralization Pattern: 200+ → 1 os.getenv() calls.

Design Principles:
- Knuth: Precise type conversion, explicit validation, fail-safe defaults
- Linus (Data): Clean data structure eliminates special cases
- Fail-Safe Production: Never crashes, uses safe defaults
- Strict Testing: Fails fast on missing/invalid configuration

Usage:
    Production mode (default):
        config = load_code_ai_config_from_env(fail_safe=True)
        # Missing env vars → uses defaults, logs warnings

    Testing/CI mode:
        config = load_code_ai_config_from_env(fail_safe=False)
        # Missing env vars → raises ValueError
"""

import logging
import os
from pathlib import Path
from typing import Optional

from .models import CodeAIConfig, ModelConfig, PathConfig, TensorFlowConfig

logger = logging.getLogger(__name__)


# Default configuration for fail-safe production mode
# Ensures application can always start, even without environment variables
DEFAULT_CONFIG = CodeAIConfig(
    model=ModelConfig(
        path_synthseg=Path("/tmp/models/synthseg"),
        gpu_n=0
    ),
    paths=PathConfig(
        path_code=Path.cwd(),
        path_process=Path("/tmp/process"),
        path_json=Path("/tmp/json"),
        path_log=Path("/tmp/logs")
    ),
    tensorflow=TensorFlowConfig(
        cpp_min_log_level="3",
        enable_auto_mixed_precision="0"
    )
)


def _get_env_or_default(
    key: str,
    default: Optional[str] = None,
    fail_safe: bool = True
) -> Optional[str]:
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
            if default is not None:
                logger.warning(
                    f"Environment variable '{key}' not set, using default: {default}"
                )
            return default
        else:
            if default is None:
                raise ValueError(
                    f"Required environment variable '{key}' is not set. "
                    f"Set this variable in your environment or .env file."
                )
            return default

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


def _convert_to_int(
    int_str: Optional[str],
    default: int,
    fail_safe: bool = True
) -> int:
    """
    Convert string to int with validation.

    Args:
        int_str: Integer string from environment variable
        default: Default int if int_str is None
        fail_safe: Whether to use default on error

    Returns:
        Integer value

    Raises:
        ValueError: If fail_safe=False and int_str is invalid
    """
    if int_str is None:
        return default

    try:
        return int(int_str)
    except (TypeError, ValueError) as e:
        if fail_safe:
            logger.warning(
                f"Invalid integer '{int_str}': {e}. Using default: {default}"
            )
            return default
        else:
            raise ValueError(
                f"Invalid integer value '{int_str}': {e}"
            ) from e


def load_code_ai_config_from_env(fail_safe: bool = True) -> CodeAIConfig:
    """
    Load code_ai configuration from environment variables.

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
        CodeAIConfig: Immutable configuration object

    Raises:
        ValueError: If fail_safe=False and required env var missing
        TypeError: If fail_safe=False and env var has invalid type

    Examples:
        >>> # Production mode
        >>> config = load_code_ai_config_from_env(fail_safe=True)
        >>> config.model.gpu_n
        0

        >>> # Testing mode with env vars set
        >>> os.environ['GPU_N'] = '1'
        >>> config = load_code_ai_config_from_env(fail_safe=False)
        >>> config.model.gpu_n
        1
    """
    # Model Configuration
    path_synthseg = _convert_to_path(
        _get_env_or_default("PATH_SYNTHSEG", None, fail_safe),
        default=DEFAULT_CONFIG.model.path_synthseg,
        fail_safe=fail_safe
    )

    gpu_n = _convert_to_int(
        _get_env_or_default("GPU_N", None, fail_safe),
        default=DEFAULT_CONFIG.model.gpu_n,
        fail_safe=fail_safe
    )

    model_config = ModelConfig(
        path_synthseg=path_synthseg,
        gpu_n=gpu_n
    )

    # Path Configuration
    path_code = _convert_to_path(
        _get_env_or_default("PATH_CODE", None, fail_safe),
        default=DEFAULT_CONFIG.paths.path_code,
        fail_safe=fail_safe
    )

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

    paths_config = PathConfig(
        path_code=path_code,
        path_process=path_process,
        path_json=path_json,
        path_log=path_log
    )

    # TensorFlow Configuration
    cpp_min_log_level = _get_env_or_default(
        "TF_CPP_MIN_LOG_LEVEL",
        default=DEFAULT_CONFIG.tensorflow.cpp_min_log_level,
        fail_safe=fail_safe
    )

    enable_auto_mixed_precision = _get_env_or_default(
        "TF_ENABLE_AUTO_MIXED_PRECISION",
        default=DEFAULT_CONFIG.tensorflow.enable_auto_mixed_precision,
        fail_safe=fail_safe
    )

    tensorflow_config = TensorFlowConfig(
        cpp_min_log_level=cpp_min_log_level,
        enable_auto_mixed_precision=enable_auto_mixed_precision
    )

    # Root Configuration
    config = CodeAIConfig(
        model=model_config,
        paths=paths_config,
        tensorflow=tensorflow_config
    )

    logger.info(
        f"Code_AI configuration loaded successfully (fail_safe={fail_safe})"
    )

    return config