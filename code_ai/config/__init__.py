"""
Configuration module for code_ai application.

Phase A: Pure Function Refactoring - Configuration Infrastructure
Provides centralized, type-safe configuration management for AI pipelines.

Design Principles:
- Knuth: Precise type definitions, self-documenting configuration
- Linus (Data Structure): Good structure makes code naturally simple
- Immutability: Configuration cannot be modified after creation
- Fail-Safe Production: Never crashes, uses safe defaults
"""

from .models import (
    ModelConfig,
    PathConfig,
    TensorFlowConfig,
    CodeAIConfig,
)
from .loader import (
    load_code_ai_config_from_env,
    DEFAULT_CONFIG,
)

__all__ = [
    # Configuration models (immutable dataclasses)
    "ModelConfig",
    "PathConfig",
    "TensorFlowConfig",
    "CodeAIConfig",
    # Configuration loader (single source of truth)
    "load_code_ai_config_from_env",
    "DEFAULT_CONFIG",
]