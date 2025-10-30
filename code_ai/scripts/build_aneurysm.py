#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to build RDX platform JSON for aneurysm detection

This script processes existing aneurysm detection results and generates
RDX platform JSON format using the AneurysmDetectionBuilder.

Usage:
    python build_aneurysm.py --id <study_id> --path-root <data_directory> [--model-id <uuid>]

Example:
    python build_aneurysm.py \
        --id 1C95C88E_20171122_MR_1C95C88E \
        --path-root /results/1C95C88E_20171122_MR_1C95C88E/aneurysm_model \
        --model-id 5d7b5e3a-9c1f-4a2b-8d6e-3f9a1c2b4e5f

Expected Directory Structure:
    path_root/
    ├── Dicom/
    │   ├── MRA_BRAIN/
    │   ├── MIP_Pitch/
    │   └── MIP_Yaw/
    ├── Image_nii/
    │   ├── MRA_BRAIN.nii.gz
    │   ├── Pred.nii.gz
    │   └── Vessel.nii.gz
    ├── Image_reslice/
    │   ├── MIP_Pitch_pred.nii.gz
    │   └── MIP_Yaw_pred.nii.gz
    └── excel/
        └── Aneurysm_Pred_list.xlsx

Author: Architecture Redesign Team
Created: 2025-10-20
"""
import argparse
import logging
import pathlib
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_ai.pipeline.rdx.build.aneurysm import AneurysmDetectionBuilder


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration

    Args:
        verbose: Enable verbose logging

    Returns:
        Logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def validate_directory_structure(path_root: pathlib.Path, logger: logging.Logger) -> bool:
    """Validate required directory structure exists

    Args:
        path_root: Root directory path
        logger: Logger instance

    Returns:
        True if valid, False otherwise
    """
    required_dirs = [
        path_root / "Dicom" / "MRA_BRAIN",
        path_root / "Image_nii",
        path_root / "excel"
    ]

    required_files = [
        path_root / "excel" / "Aneurysm_Pred_list.xlsx",
        path_root / "Image_nii" / "Pred.nii.gz"
    ]

    missing_dirs = [d for d in required_dirs if not d.exists()]
    missing_files = [f for f in required_files if not f.exists()]

    if missing_dirs:
        logger.error(f"Missing required directories: {missing_dirs}")
        return False

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build RDX platform JSON for aneurysm detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python build_aneurysm.py \\
        --id Study_12345 \\
        --path-root /results/Study_12345/aneurysm_model

Expected directory structure:
    path_root/
    ├── Dicom/MRA_BRAIN/        # DICOM files
    ├── Image_nii/Pred.nii.gz   # Prediction NIfTI
    └── excel/Aneurysm_Pred_list.xlsx  # Detection results
        """
    )

    parser.add_argument(
        "--id",
        required=True,
        help="Study/Patient ID"
    )

    parser.add_argument(
        "--path-root",
        required=True,
        type=pathlib.Path,
        help="Root directory containing all processing results"
    )

    parser.add_argument(
        "--model-id",
        default="5d7b5e3a-9c1f-4a2b-8d6e-3f9a1c2b4e5f",
        help="Model UUID (default: aneurysm detection model)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("RDX Platform JSON Builder - Aneurysm Detection")
    logger.info("=" * 60)
    logger.info(f"Study ID: {args.id}")
    logger.info(f"Path Root: {args.path_root}")
    logger.info(f"Model ID: {args.model_id}")

    # Validate directory structure
    if not args.path_root.exists():
        logger.error(f"Path root does not exist: {args.path_root}")
        return 1

    if not validate_directory_structure(args.path_root, logger):
        logger.error("Directory structure validation failed")
        logger.error("Please ensure all required directories and files exist")
        return 1

    # Build platform JSON
    try:
        logger.info("Starting platform JSON generation...")

        platform_json = AneurysmDetectionBuilder.execute_rdx_platform_json(
            _id=args.id,
            path_root=args.path_root,
            model_id=args.model_id
        )

        # Check output
        json_path = args.path_root / 'rdx_aneurysm_json.json'
        if json_path.exists():
            logger.info(f"✅ Platform JSON created successfully: {json_path}")
            logger.info(f"File size: {json_path.stat().st_size} bytes")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("❌ Platform JSON was not created")
            return 1

    except Exception as e:
        logger.error(f"❌ Platform JSON generation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
