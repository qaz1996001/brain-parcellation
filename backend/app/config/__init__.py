"""
Environment configuration module for brain-parcellation backend.

This module provides environment-aware configuration following the Twelve-Factor App
methodology and design principles from Ken Thompson, Linus Torvalds, Martin Fowler,
and Donald Knuth.
"""

from .environments import (
    get_environment,
    get_config,
    EnvironmentConfig,
    validate_environment,
)

__all__ = [
    "get_environment",
    "get_config",
    "EnvironmentConfig",
    "validate_environment",
]
