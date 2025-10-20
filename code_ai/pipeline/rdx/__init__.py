#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDX AI Medical Imaging Pipeline Framework

Redesigned pipeline architecture with:
    - Unified configuration system (CLI > ENV > Defaults)
    - Model-organized output directories
    - Shared base classes for common functionality
    - GPU management and monitoring
    - Comprehensive logging

Modules:
    base: Base classes and infrastructure components

Author: Architecture Redesign Team
Created: 2025-10-20
"""

from .base import (
    BasePipeline,
    GPUManager,
    LoggingManager,
    OutputManager,
    PipelineConfig,
    PipelineLogger,
    create_pipeline_parser,
)

__all__ = [
    # Base classes
    'BasePipeline',
    'PipelineConfig',

    # Infrastructure
    'OutputManager',
    'GPUManager',
    'LoggingManager',
    'PipelineLogger',

    # Utilities
    'create_pipeline_parser',
]

__version__ = '1.0.0'
