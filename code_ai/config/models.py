"""
Configuration dataclasses for code_ai application.

All configuration is represented as immutable, type-annotated dataclasses
following Knuth's precision principle: explicit types, boundaries, and validation.

Design Pattern: Configuration Centralization
- All environment variable reads concentrated in loader.py
- Immutable frozen dataclasses ensure no runtime modification
- Type hints provide compile-time safety
- Hierarchical structure mirrors system architecture
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    """
    AI Model configuration.

    Contains paths to trained models and inference parameters.

    Attributes:
        path_synthseg: Path to SynthSeg model directory
        gpu_n: GPU device number (0 for first GPU, -1 for CPU)
    """
    path_synthseg: Path
    gpu_n: int = 0


@dataclass(frozen=True)
class PathConfig:
    """
    File system path configuration for AI processing.

    All paths are represented as pathlib.Path objects for:
    - Type safety (Path vs str)
    - Platform independence (automatic path separator handling)
    - Validation capabilities

    Attributes:
        path_code: Code/source directory path
        path_process: Directory for processing temporary files
        path_json: Directory for JSON output files
        path_log: Directory for log files
    """
    path_code: Path
    path_process: Path
    path_json: Path
    path_log: Path


@dataclass(frozen=True)
class TensorFlowConfig:
    """
    TensorFlow-specific configuration.

    Controls TensorFlow behavior and logging.

    Attributes:
        cpp_min_log_level: TensorFlow C++ logging level (0=all, 3=errors only)
        enable_auto_mixed_precision: Enable automatic mixed precision (0=off, 1=on)
    """
    cpp_min_log_level: str = "3"
    enable_auto_mixed_precision: str = "0"


@dataclass(frozen=True)
class CodeAIConfig:
    """
    Root configuration aggregating all code_ai subsystems.

    This is the single source of truth for code_ai configuration.
    Immutable by design (frozen=True) to prevent runtime modification.

    Design Principles:
    - Knuth: Each field has explicit type and clear purpose
    - Linus (Data Structure): Good structure makes code naturally simple
    - Immutability: Configuration cannot be modified after creation

    Attributes:
        model: AI model configuration
        paths: File system path configuration
        tensorflow: TensorFlow-specific settings
    """
    model: ModelConfig
    paths: PathConfig
    tensorflow: TensorFlowConfig