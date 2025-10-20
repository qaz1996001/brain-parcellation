#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base classes for medical imaging AI pipelines

This module provides common infrastructure for all pipeline implementations:
    - PipelineConfig: Configuration management with CLI > ENV > Defaults hierarchy
    - BasePipeline: Abstract base class for pipeline execution
    - OutputManager: Output directory organization by model
    - GPUManager: GPU device management and memory monitoring
    - LoggingManager: Unified logging with daily rotation

Author: Architecture Redesign Team
Created: 2025-10-20
"""

from .config import PipelineConfig, create_pipeline_parser
from .gpu import GPUManager
from .logging_manager import LoggingManager, PipelineLogger
from .output import OutputManager
from .pipeline import BasePipeline

__all__ = [
    # Configuration
    'PipelineConfig',
    'create_pipeline_parser',

    # Pipeline base class
    'BasePipeline',

    # Infrastructure components
    'OutputManager',
    'GPUManager',
    'LoggingManager',
    'PipelineLogger',
]

__version__ = '1.0.0'
