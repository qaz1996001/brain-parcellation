#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vessel Dilated Segmentation Pipeline

Redesigned pipeline using base classes with:
- Unified configuration (CLI > ENV > Defaults)
- Model-organized output directory
- GPU management and logging

Author: Architecture Redesign Team
Created: 2025-10-20

Python 3.10.13
tensorflow==2.14.0
numpy==1.26.0
SimpleITK==2.3.1
nibabel==5.1.0
scikit-image==0.22.0
pynvml==12.0.0
scipy==1.11.3
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from code_ai.pipeline.rdx import BasePipeline, PipelineConfig, create_pipeline_parser
from code_ai.pipeline.rdx.build.vessel import VesselDilatedBuilder


class VesselDilatedPipeline(BasePipeline):
    """Vessel dilated segmentation pipeline

    Input:
        - MRA_BRAIN.nii.gz: MR Angiography brain scan
        - (Optional) DICOM directory for DICOM-seg generation

    Outputs (in {output_dir}/{study_id}/vessel_dilated_model/):
        - {study_id}.json: Metadata and detection results
        - {study_id}_A01.dcm: DICOM-seg file
        - Vessel.nii.gz: Vessel segmentation

    Process:
        1. Run vessel segmentation inference (reuse from aneurysm pipeline)
        2. Apply 3D dilation to vessel mask
        3. Create DICOM-seg file
        4. Generate platform JSON with results
    """

    @property
    def model_name(self) -> str:
        return "vessel_dilated"

    def required_inputs(self) -> List[str]:
        return ["MRA_BRAIN"]

    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Run vessel dilated segmentation pipeline

        Args:
            inputs: Dictionary with key "MRA_BRAIN"

        Returns:
            Dictionary mapping output names to file paths

        Raises:
            subprocess.CalledProcessError: If GPU inference fails
            FileNotFoundError: If required outputs not created
        """
        mra_path = inputs["MRA_BRAIN"]

        self.logger.info("Starting Vessel Dilated segmentation pipeline")
        self.logger.info(f"  MRA_BRAIN: {mra_path}")

        # Step 1: Copy MRA to working directory
        self.logger.info("Step 1: Preparing input...")
        work_mra = self.copy_to_working_dir(mra_path, "MRA_BRAIN.nii.gz")

        # Step 2: Run GPU inference (vessel segmentation)
        self.logger.info("Step 2: Running vessel segmentation inference...")
        self._run_vessel_segmentation()

        # Verify vessel output was created
        vessel_output = self.working_dir / "Vessel.nii.gz"
        if not vessel_output.exists():
            raise FileNotFoundError(
                f"Vessel segmentation not created: {vessel_output}\n"
                f"Working directory contents: {list(self.working_dir.iterdir())}"
            )

        # Step 3: Setup output directories
        self.logger.info("Step 3: Setting up output directories...")
        path_dcm, path_nii = self._setup_output_dirs()

        # Copy vessel output to nii directory
        shutil.copy2(vessel_output, path_nii / "Vessel.nii.gz")

        # Step 4: Create DICOM-seg files (if DICOM directory provided)
        dicom_seg_path = None
        if self.config.input_dicom_dir:
            self.logger.info("Step 4: Creating dilated vessel DICOM-seg...")
            # Copy DICOM directory
            dicom_dest = path_dcm / "MRA_BRAIN"
            if not dicom_dest.exists():
                shutil.copytree(self.config.input_dicom_dir, dicom_dest)

            # Generate platform JSON will handle DICOM-seg creation via builder
        else:
            self.logger.info("Step 4: Skipping DICOM-seg (no DICOM directory)")

        # Step 5: Generate platform JSON using VesselDilatedBuilder
        self.logger.info("Step 5: Generating platform JSON...")
        json_path = self._generate_platform_json()

        # Collect outputs
        outputs = {
            "Vessel": vessel_output,
            "JSON": json_path,
        }

        # Add DICOM-seg if created
        dicom_seg_dir = path_dcm / "Dicom-Seg"
        if dicom_seg_dir.exists():
            dicom_files = list(dicom_seg_dir.glob("*.dcm"))
            for i, dcm_path in enumerate(dicom_files, start=1):
                outputs[f"DICOM_A{i:02d}"] = dcm_path

        self.logger.info(f"Vessel dilated inference completed: {len(outputs)} outputs")
        return outputs

    def _run_vessel_segmentation(self) -> None:
        """Execute GPU-based vessel segmentation inference

        Uses same GPU script as aneurysm pipeline to generate Vessel.nii.gz

        Raises:
            subprocess.CalledProcessError: If inference fails
            FileNotFoundError: If gpu_aneurysm.py not found
        """
        # Find gpu_aneurysm.py (which generates vessel segmentation)
        gpu_script = self._find_script("gpu_aneurysm.py")

        # Build command
        cmd = [
            "python", str(gpu_script),
            "--path_code", str(self.config.path_code or Path.cwd()),
            "--path_process", str(self.working_dir),
            "--case", self.config.study_id,
            "--path_log", str(self.config.path_log or Path("./logs")),
            "--gpu_n", str(self.config.gpu_id)
        ]

        self.logger.info(f"Running vessel segmentation: {' '.join(cmd)}")

        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Vessel segmentation failed with return code {result.returncode}")
            self.logger.error(f"STDOUT: {result.stdout}")
            self.logger.error(f"STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)

        self.logger.info("Vessel segmentation completed successfully")

    def _setup_output_dirs(self) -> tuple[Path, Path]:
        """Setup organized output directories

        Returns:
            Tuple of (dicom_dir, nii_dir)
        """
        path_dcm = self.working_dir / "Dicom"
        path_nii = self.working_dir / "Image_nii"

        for path in [path_dcm, path_nii]:
            path.mkdir(parents=True, exist_ok=True)

        return path_dcm, path_nii

    def _generate_platform_json(self) -> Path:
        """Generate platform JSON using VesselDilatedBuilder

        Returns:
            Path to created JSON file
        """
        try:
            self.logger.info("Generating platform JSON using VesselDilatedBuilder...")

            # Call builder to generate platform JSON
            # This will apply 3D dilation and create DICOM-seg
            platform_json = VesselDilatedBuilder.execute_rdx_platform_json(
                _id=self.config.study_id,
                path_root=self.working_dir,
                model_id='289fc383-34ff-46b7-bf0a-fbb22a104a18'  # Vessel dilated model ID
            )

            # Find generated JSON file
            json_file = self.working_dir / 'rdx_vessel_dilated_json.json'

            if not json_file.exists():
                # Fallback: create basic JSON
                self.logger.warning("Platform JSON not created by builder, generating basic version")
                json_file = self._create_basic_json()

            return json_file

        except Exception as e:
            self.logger.error(f"Platform JSON generation failed: {e}", exc_info=True)
            # Create basic JSON as fallback
            return self._create_basic_json()

    def _create_basic_json(self) -> Path:
        """Create basic JSON metadata (fallback)

        Returns:
            Path to created JSON file
        """
        json_file = self.working_dir / f"{self.config.study_id}.json"

        import datetime

        metadata = {
            "study_id": self.config.study_id,
            "model": "vessel_dilated_segmentation",
            "model_name": self.model_name,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return json_file

    def _find_script(self, script_name: str) -> Path:
        """Find script file in chuan directory

        Args:
            script_name: Name of script file

        Returns:
            Path to script

        Raises:
            FileNotFoundError: If script not found
        """
        # Try chuan directory
        chuan_dir = Path(__file__).parent.parent / "chuan"
        script_path = chuan_dir / script_name

        if script_path.exists():
            return script_path

        # Try code directory from config
        if self.config.path_code:
            script_path = self.config.path_code / script_name
            if script_path.exists():
                return script_path

        raise FileNotFoundError(f"Script not found: {script_name}")

    def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
        """Map output files to destination paths

        Args:
            output_name: Name from run_inference() output dict
            output_path: Source file path

        Returns:
            Destination path in model-organized directory
        """
        if output_name == "JSON":
            return self.output_manager.get_json_path(self.model_name)

        elif output_name.startswith("DICOM_A"):
            # Extract slice number
            slice_num = int(output_name[-2:])
            return self.output_manager.get_dicom_path(self.model_name, slice_num)

        elif output_name == "Vessel":
            return self.output_manager.get_nifti_path(self.model_name, "Vessel")

        else:
            # Default: keep original filename
            return self.output_manager.get_output_path(self.model_name, output_path.name)


def main():
    """Main entry point for Vessel Dilated pipeline"""
    # Create argument parser
    parser = create_pipeline_parser(
        description="Vessel Dilated Segmentation Pipeline"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig.from_cli_and_env(args)

    # Create and execute pipeline
    pipeline = VesselDilatedPipeline(config)
    success = pipeline.execute()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
