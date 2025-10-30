#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline configuration with CLI > ENV > Defaults hierarchy

Author: Architecture Redesign Team
Created: 2025-10-20
"""
import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution with CLI > ENV > Default priority

    Configuration hierarchy:
        1. CLI arguments (highest priority)
        2. Environment variables
        3. Default values (lowest priority)

    Mandatory parameters:
        - study_id: Must be provided via CLI --id

    Optional parameters with fallbacks:
        - input_files: Input file paths
        - input_dicom_dir: Source DICOM directory for DICOM-seg generation
        - output_dir: Base output directory (CLI > ENV[PATH_OUTPUT] > ./output)
        - working_dir: Temporary working directory (CLI > ENV[PATH_WORK] > /tmp/pipeline_work)
        - gpu_id: GPU device ID (CLI > ENV[GPU_N] > 0)
        - gpu_memory_threshold: GPU usage threshold (ENV[GPU_MEMORY_THRESHOLD] > 0.6)

    Infrastructure paths (from ENV):
        - path_code: Code directory (ENV[PATH_CODE])
        - path_process: Processing directory (ENV[PATH_PROCESS])
        - path_json: JSON output directory (ENV[PATH_JSON])
        - path_log: Log directory (ENV[PATH_LOG] > ./logs)
    """

    # Mandatory parameters
    study_id: str

    # Input/Output paths
    input_files: List[Path] = field(default_factory=list)
    input_dicom_dir: Optional[Path] = None
    output_dir: Path = Path('./output')
    working_dir: Path = Path('/tmp/pipeline_work')

    # GPU configuration
    gpu_id: int = 0
    gpu_memory_threshold: float = 0.6

    # Infrastructure paths (from ENV)
    path_code: Optional[Path] = None
    path_process: Optional[Path] = None
    path_json: Optional[Path] = None
    path_log: Path = Path('./logs')
    path_python: Optional[Path] = None

    # Advanced options
    legacy_output: bool = False  # Use old output structure
    keep_intermediate: bool = False  # Keep temporary files
    verbose: bool = False  # Enable verbose logging

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'PipelineConfig':
        """Create configuration with CLI > ENV > Default priority

        Args:
            args: Parsed command-line arguments from argparse

        Returns:
            PipelineConfig instance with resolved configuration

        Raises:
            ValueError: If mandatory parameters are missing or invalid
        """
        # Mandatory: study_id
        study_id = args.id
        if not study_id:
            raise ValueError("Study ID (--id) is mandatory")

        # Input files
        input_files = [Path(f) for f in args.input] if hasattr(args, 'input') and args.input else []

        # Input DICOM directory (optional)
        input_dicom_dir = None
        if hasattr(args, 'input_dicom_dir') and args.input_dicom_dir:
            input_dicom_dir = Path(args.input_dicom_dir)

        # Output directory: CLI > ENV > Default
        output_dir = Path('./output')
        if hasattr(args, 'output_dir') and args.output_dir:
            output_dir = Path(args.output_dir)
        elif os.getenv('PATH_OUTPUT'):
            output_dir = Path(os.getenv('PATH_OUTPUT'))

        # Working directory: CLI > ENV > Default
        working_dir = Path('/tmp/pipeline_work')
        if hasattr(args, 'working_dir') and args.working_dir:
            working_dir = Path(args.working_dir)
        elif os.getenv('PATH_WORK'):
            working_dir = Path(os.getenv('PATH_WORK'))

        # GPU settings: CLI > ENV > Default
        gpu_id = 0
        if hasattr(args, 'gpu') and args.gpu is not None:
            gpu_id = args.gpu
        elif os.getenv('GPU_N'):
            gpu_id = int(os.getenv('GPU_N'))

        gpu_memory_threshold = float(os.getenv('GPU_MEMORY_THRESHOLD', '0.6'))

        # Infrastructure paths (from ENV)
        path_code = Path(os.getenv('PATH_CODE')) if os.getenv('PATH_CODE') else None
        path_process = Path(os.getenv('PATH_PROCESS')) if os.getenv('PATH_PROCESS') else None
        path_json = Path(os.getenv('PATH_JSON')) if os.getenv('PATH_JSON') else None
        path_python = Path(os.getenv('PYTHON3')) if os.getenv('PYTHON3') else None

        # Log directory: ENV > Default
        path_log = Path('./logs')
        if os.getenv('PATH_LOG'):
            path_log = Path(os.getenv('PATH_LOG'))

        # Advanced options
        legacy_output = getattr(args, 'legacy_output', False)
        keep_intermediate = getattr(args, 'keep_intermediate', False)
        verbose = getattr(args, 'verbose', False)

        return cls(
            study_id=study_id,
            input_files=input_files,
            input_dicom_dir=input_dicom_dir,
            output_dir=output_dir,
            working_dir=working_dir,
            gpu_id=gpu_id,
            gpu_memory_threshold=gpu_memory_threshold,
            path_code=path_code,
            path_process=path_process,
            path_json=path_json,
            path_log=path_log,
            path_python = path_python,
            legacy_output=legacy_output,
            keep_intermediate=keep_intermediate,
            verbose=verbose,
        )

    def validate(self) -> None:
        """Validate configuration and raise ValueError if invalid

        Raises:
            ValueError: If configuration is invalid with detailed error message
        """
        # Validate study_id
        if not self.study_id or not self.study_id.strip():
            raise ValueError("study_id is mandatory and cannot be empty")

        # Validate input files exist
        if not self.input_files:
            raise ValueError("At least one input file is required")

        for input_file in self.input_files:
            if not input_file.exists():
                raise ValueError(f"Input file not found: {input_file}")
            if not input_file.is_file():
                raise ValueError(f"Input path is not a file: {input_file}")

        # Validate input DICOM directory if provided
        if self.input_dicom_dir is not None:
            if not self.input_dicom_dir.exists():
                raise ValueError(f"Input DICOM directory not found: {self.input_dicom_dir}")
            if not self.input_dicom_dir.is_dir():
                raise ValueError(f"Input DICOM path is not a directory: {self.input_dicom_dir}")

        # Validate GPU settings
        if self.gpu_id < 0:
            raise ValueError(f"Invalid GPU ID: {self.gpu_id} (must be >= 0)")

        if not 0.0 < self.gpu_memory_threshold <= 1.0:
            raise ValueError(
                f"Invalid GPU memory threshold: {self.gpu_memory_threshold} "
                f"(must be between 0.0 and 1.0)"
            )

    def __str__(self) -> str:
        """String representation for logging"""
        return (
            f"PipelineConfig(\n"
            f"  study_id='{self.study_id}',\n"
            f"  input_files={[str(f) for f in self.input_files]},\n"
            f"  input_dicom_dir={self.input_dicom_dir},\n"
            f"  output_dir={self.output_dir},\n"
            f"  working_dir={self.working_dir},\n"
            f"  gpu_id={self.gpu_id},\n"
            f"  gpu_memory_threshold={self.gpu_memory_threshold}\n"
            f")"
        )


def create_pipeline_parser(description: str = "Medical imaging AI pipeline") -> argparse.ArgumentParser:
    """Create standardized argument parser for pipelines

    Args:
        description: Pipeline description for help text

    Returns:
        Configured ArgumentParser with standard pipeline arguments

    Example:
        >>> parser = create_pipeline_parser("Aneurysm Detection Pipeline")
        >>> parser.add_argument('--custom-arg', help='Pipeline-specific argument')
        >>> args = parser.parse_args()
        >>> config = PipelineConfig.from_cli_and_env(args)
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Priority:
  CLI arguments > Environment variables > Default values

Examples:
  # Minimal usage (output to default ./output/)
  python pipeline.py --id Study_12345 --input scan.nii.gz

  # With custom output directory
  python pipeline.py --id Study_12345 --input scan.nii.gz --output-dir /results

  # With environment variables
  export PATH_OUTPUT=/results
  export GPU_N=1
  python pipeline.py --id Study_12345 --input scan.nii.gz

  # Multiple inputs (CMB pipeline)
  python pipeline.py --id Study_12345 --input swan.nii.gz --input t1.nii.gz

  # With DICOM-seg generation
  python pipeline.py --id Study_12345 --input scan.nii.gz --input-dicom-dir /dicom/source
        """
    )

    # Mandatory arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--id',
        type=str,
        required=True,
        help='Study/Patient ID (mandatory)'
    )
    required.add_argument(
        '--input',
        action='append',
        required=True,
        help='Input file path (can be specified multiple times for multi-input models)'
    )

    # Optional arguments
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '--input-dicom-dir',
        type=str,
        help='Input DICOM directory for DICOM-seg generation (optional)'
    )
    optional.add_argument(
        '--output-dir',
        type=str,
        help='Base output directory (default: ENV[PATH_OUTPUT] or ./output)'
    )
    optional.add_argument(
        '--working-dir',
        type=str,
        help='Temporary working directory (default: ENV[PATH_WORK] or /tmp/pipeline_work)'
    )
    optional.add_argument(
        '--gpu',
        type=int,
        help='GPU device ID (default: ENV[GPU_N] or 0)'
    )

    # Advanced options
    advanced = parser.add_argument_group('advanced options')
    advanced.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='Keep intermediate/temporary files (default: False)'
    )
    advanced.add_argument(
        '--legacy-output',
        action='store_true',
        help='Use legacy output structure instead of model-organized (default: False)'
    )
    advanced.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (default: False)'
    )

    # Backward compatibility (hidden from help)
    parser.add_argument('--ID', dest='id', help=argparse.SUPPRESS)
    parser.add_argument('--Inputs', dest='input', action='append', help=argparse.SUPPRESS)
    parser.add_argument('--DicomDir', dest='input_dicom_dir', help=argparse.SUPPRESS)
    parser.add_argument('--InputsDicomDir', dest='input_dicom_dir', help=argparse.SUPPRESS)
    parser.add_argument('--Output_folder', dest='output_dir', help=argparse.SUPPRESS)

    return parser
