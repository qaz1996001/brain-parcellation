#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to build RDX platform JSON for vessel dilated segmentation

This script processes existing vessel segmentation results, applies 3D dilation,
and generates RDX platform JSON format using the VesselDilatedBuilder.

Usage:
    python build_vessel.py --id <study_id> --path-root <data_directory> [--model-id <uuid>]

Example:
    python build_vessel.py \
        --id 1C95C88E_20171122_MR_1C95C88E \
        --path-root /results/1C95C88E_20171122_MR_1C95C88E/vessel_dilated_model \
        --model-id 289fc383-34ff-46b7-bf0a-fbb22a104a18

Expected Directory Structure:
    path_root/
    ├── Dicom/
    │   └── MRA_BRAIN/
    └── Image_nii/
        └── Vessel.nii.gz

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

from code_ai.pipeline.rdx.build.vessel import VesselDilatedBuilder


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
        path_root / "Image_nii"
    ]

    required_files = [
        path_root / "Image_nii" / "Vessel.nii.gz"
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
        description="Build RDX platform JSON for vessel dilated segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python build_vessel.py \\
        --id Study_12345 \\
        --path-root /results/Study_12345/vessel_dilated_model

Expected directory structure:
    path_root/
    ├── Dicom/MRA_BRAIN/      # DICOM files
    └── Image_nii/Vessel.nii.gz  # Vessel segmentation NIfTI

The script will:
1. Load vessel segmentation from Vessel.nii.gz
2. Apply 3D dilation (kernel 3x3x3, 15 iterations)
3. Create DICOM-seg file with dilated vessel
4. Generate RDX platform JSON
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
        default="289fc383-34ff-46b7-bf0a-fbb22a104a18",
        help="Model UUID (default: vessel dilated model)"
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
    logger.info("RDX Platform JSON Builder - Vessel Dilated Segmentation")
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
        logger.info("This will apply 3D dilation to vessel segmentation")

        platform_json = VesselDilatedBuilder.execute_rdx_platform_json(
            _id=args.id,
            path_root=args.path_root,
            model_id=args.model_id
        )

        # Check output
        json_path = args.path_root / 'rdx_vessel_dilated_json.json'
        if json_path.exists():
            logger.info(f"✅ Platform JSON created successfully: {json_path}")
            logger.info(f"File size: {json_path.stat().st_size} bytes")

            # Check for DICOM-seg
            dicom_seg_dir = args.path_root / "Dicom" / "Dicom-Seg"
            if dicom_seg_dir.exists():
                dicom_files = list(dicom_seg_dir.glob("*.dcm"))
                logger.info(f"✅ Created {len(dicom_files)} DICOM-seg file(s)")

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
