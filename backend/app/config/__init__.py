"""
Environment configuration module for brain-parcellation backend.

This module provides environment-aware configuration following the Twelve-Factor App
methodology and design principles from Ken Thompson, Linus Torvalds, Martin Fowler,
and Donald Knuth.

Phase A: Pure Function Refactoring - Configuration Infrastructure
New exports for centralized, type-safe configuration management.
"""

from .environments import (
    get_environment,
    get_config,
    EnvironmentConfig,
    validate_environment,
)

# Phase A: Configuration Centralization Pattern
from .models import (
    APIConfig,
    PathConfig,
    DatabaseConfig,
    BackendConfig,
)
from .loader import (
    load_backend_config_from_env,
    DEFAULT_CONFIG,
)

__all__ = [
    # Legacy environment management
    "get_environment",
    "get_config",
    "EnvironmentConfig",
    "validate_environment",
    # Phase A: Configuration models (immutable dataclasses)
    "APIConfig",
    "PathConfig",
    "DatabaseConfig",
    "BackendConfig",
    # Phase A: Configuration loader (single source of truth)
    "load_backend_config_from_env",
    "DEFAULT_CONFIG",
]
