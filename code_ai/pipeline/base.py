"""
Base utilities for AI pipeline configuration.

Phase C: Pure Function Refactoring - Pipeline Configuration Support

This module provides the dual-mode configuration pattern for pipelines:
- Old pattern: Pipelines read from environment directly (backward compatible)
- New pattern: Pipelines receive config via parameter (pure function)

Design Principles:
- Knuth: Precise type definitions, clear interfaces
- Linus: "We do not break userspace" - old patterns work forever
- Fail-Safe: Production never crashes on config issues

Usage:
    # Old pattern (backward compatible, still works forever)
    result = pipeline_cmb(ID, input_file, output_folder)

    # New pattern (pure function with explicit config)
    config = load_code_ai_config_from_env()
    result = pipeline_cmb(ID, input_file, output_folder, config=config)
"""

import logging
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from code_ai.config import CodeAIConfig

logger = logging.getLogger(__name__)

# Cached config instance for fail-safe mode
_cached_config: Optional["CodeAIConfig"] = None


def get_config(config: Optional["CodeAIConfig"] = None) -> "CodeAIConfig":
    """
    Get configuration with dual-mode support.

    This function implements the backward compatibility pattern:
    - If config is provided, use it (pure function pattern)
    - If config is None, load from environment (legacy pattern)

    The result is cached to avoid repeated environment reads.

    Args:
        config: Optional explicit configuration object.
                If None, loads from environment.

    Returns:
        CodeAIConfig: Configuration object (immutable)

    Examples:
        >>> # New pattern: explicit config (testable, deterministic)
        >>> config = load_code_ai_config_from_env()
        >>> cfg = get_config(config)
        >>> cfg is config
        True

        >>> # Old pattern: load from environment (backward compatible)
        >>> cfg = get_config()  # Loads and caches from env
        >>> cfg.model.gpu_n
        0
    """
    global _cached_config

    if config is not None:
        # New pattern: use explicitly provided config
        return config

    # Old pattern: load from environment (cached)
    if _cached_config is None:
        from code_ai.config import load_code_ai_config_from_env
        _cached_config = load_code_ai_config_from_env(fail_safe=True)
        logger.info("Loaded pipeline config from environment (cached)")

    return _cached_config


def clear_config_cache() -> None:
    """
    Clear the cached configuration.

    Used for testing to reset state between tests.
    """
    global _cached_config
    _cached_config = None
    logger.debug("Pipeline config cache cleared")


def get_path_with_fallback(
    config: Optional["CodeAIConfig"],
    env_var: str,
    config_path: Optional[Path] = None,
    default: Optional[str] = None
) -> Optional[str]:
    """
    Get path from config or environment with fallback support.

    This function provides the dual-mode path resolution:
    1. If config is provided and has the path, use it
    2. If no config, try environment variable
    3. If neither, use default

    Args:
        config: Optional configuration object
        env_var: Environment variable name for legacy mode
        config_path: Path from config (if available)
        default: Default value if nothing else works

    Returns:
        Path string or None

    Examples:
        >>> # With config (pure function pattern)
        >>> path = get_path_with_fallback(config, "PATH_LOG", config.paths.path_log)

        >>> # Without config (legacy pattern)
        >>> path = get_path_with_fallback(None, "PATH_LOG", None, "/tmp/logs")
    """
    if config is not None and config_path is not None:
        return str(config_path)

    # Fallback to environment (legacy pattern)
    return os.getenv(env_var, default)


def get_gpu_n(config: Optional["CodeAIConfig"] = None) -> int:
    """
    Get GPU number from config or environment.

    Args:
        config: Optional configuration object

    Returns:
        GPU device number (default: 0)
    """
    if config is not None:
        return config.model.gpu_n

    try:
        return int(os.getenv("GPU_N", "0"))
    except (TypeError, ValueError):
        logger.warning("Invalid GPU_N value, using default: 0")
        return 0
