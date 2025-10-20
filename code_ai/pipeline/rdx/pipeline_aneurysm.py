#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aneurysm Detection and Vessel Segmentation Pipeline

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
from code_ai.pipeline.rdx.build.aneurysm import AneurysmDetectionBuilder


class AneurysmPipeline(BasePipeline):
    """Aneurysm detection and vessel segmentation pipeline

    Input:
        - MRA_BRAIN.nii.gz: MR Angiography brain scan
        - (Optional) DICOM directory for DICOM-seg generation

    Outputs (in {output_dir}/{study_id}/aneurysm_model/):
        - {study_id}.json: Metadata and detection results
        - {study_id}_A01.dcm, A02.dcm: DICOM-seg slices
        - Pred_Aneurysm.nii.gz: Aneurysm prediction mask
        - Prob_Aneurysm.nii.gz: Aneurysm probability map
        - Pred_Aneurysm_Vessel.nii.gz: Vessel segmentation
        - Pred_Aneurysm_Vessel16.nii.gz: 16-label vessel segmentation

    Process:
        1. Run GPU-based aneurysm detection inference
        2. Generate MIP (Maximum Intensity Projection) images
        3. Calculate aneurysm metrics (size, location, etc.)
        4. Create DICOM-seg files
        5. Generate platform JSON with results
    """

    @property
    def model_name(self) -> str:
        return "aneurysm"

    def required_inputs(self) -> List[str]:
        return ["MRA_BRAIN"]

    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Run aneurysm detection pipeline

        Args:
            inputs: Dictionary with key "MRA_BRAIN"

        Returns:
            Dictionary mapping output names to file paths

        Raises:
            subprocess.CalledProcessError: If GPU inference fails
            FileNotFoundError: If required outputs not created
        """
        mra_path = inputs["MRA_BRAIN"]

        self.logger.info("Starting Aneurysm detection pipeline")
        self.logger.info(f"  MRA_BRAIN: {mra_path}")

        # Step 1: Copy MRA to working directory
        self.logger.info("Step 1: Preparing input...")
        work_mra = self.copy_to_working_dir(mra_path, "MRA_BRAIN.nii.gz")

        # Step 2: Run GPU inference
        self.logger.info("Step 2: Running aneurysm detection inference...")
        self._run_gpu_inference()

        # Verify core outputs were created
        self._verify_core_outputs()

        # Step 3: Setup output directories
        self.logger.info("Step 3: Setting up output directories...")
        path_dcm, path_nii, path_reslice, path_excel = self._setup_output_dirs()

        # Copy core outputs to organized directories
        self._copy_core_outputs(path_nii)

        # Step 4: Generate MIP images (if DICOM directory provided)
        mip_outputs = []
        if self.config.input_dicom_dir:
            self.logger.info("Step 4: Generating MIP images...")
            mip_outputs = self._generate_mip(path_dcm, path_reslice)
        else:
            self.logger.info("Step 4: Skipping MIP generation (no DICOM directory)")

        # Step 5: Calculate aneurysm metrics
        self.logger.info("Step 5: Calculating aneurysm metrics...")
        self._calculate_metrics(path_dcm, path_excel)

        # Step 6: Create DICOM-seg files (if DICOM directory provided)
        dicom_seg_paths = []
        if self.config.input_dicom_dir:
            self.logger.info("Step 6: Creating DICOM-seg files...")
            dicom_seg_paths = self._create_dicom_seg(path_dcm, path_nii, path_reslice)
        else:
            self.logger.info("Step 6: Skipping DICOM-seg (no DICOM directory)")

        # Step 7: Generate platform JSON
        self.logger.info("Step 7: Generating platform JSON...")
        json_path = self._generate_platform_json(path_dcm, path_nii, path_reslice, path_excel)

        # Collect outputs
        outputs = {
            "Pred": self.working_dir / "Pred.nii.gz",
            "Prob": self.working_dir / "Prob.nii.gz",
            "Vessel": self.working_dir / "Vessel.nii.gz",
            "Vessel16": self.working_dir / "Vessel_16.nii.gz",
            "JSON": json_path,
        }

        # Add DICOM-seg slices
        for i, dcm_path in enumerate(dicom_seg_paths, start=1):
            outputs[f"DICOM_A{i:02d}"] = dcm_path

        # Add MIP outputs
        for i, mip_path in enumerate(mip_outputs, start=1):
            outputs[f"MIP_{i}"] = mip_path

        self.logger.info(f"Aneurysm inference completed: {len(outputs)} outputs")
        return outputs

    def _run_gpu_inference(self) -> None:
        """Execute GPU-based aneurysm detection inference

        Calls gpu_aneurysm.py via subprocess

        Raises:
            subprocess.CalledProcessError: If inference fails
            FileNotFoundError: If gpu_aneurysm.py not found
        """
        # Find gpu_aneurysm.py
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

        self.logger.info(f"Running GPU inference: {' '.join(cmd)}")

        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"GPU inference failed with return code {result.returncode}")
            self.logger.error(f"STDOUT: {result.stdout}")
            self.logger.error(f"STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)

        self.logger.info("GPU inference completed successfully")

    def _verify_core_outputs(self) -> None:
        """Verify that core inference outputs were created

        Raises:
            FileNotFoundError: If required outputs not found
        """
        required_outputs = [
            "Pred.nii.gz",
            "Prob.nii.gz",
            "Vessel.nii.gz",
        ]

        for output_file in required_outputs:
            output_path = self.working_dir / output_file
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Required output not created: {output_path}\n"
                    f"Working directory: {self.working_dir}\n"
                    f"Contents: {list(self.working_dir.iterdir())}"
                )

    def _setup_output_dirs(self) -> tuple[Path, Path, Path, Path]:
        """Setup organized output directories

        Returns:
            Tuple of (dicom_dir, nii_dir, reslice_dir, excel_dir)
        """
        path_dcm = self.working_dir / "Dicom"
        path_nii = self.working_dir / "Image_nii"
        path_reslice = self.working_dir / "Image_reslice"
        path_excel = self.working_dir / "excel"

        for path in [path_dcm, path_nii, path_reslice, path_excel]:
            path.mkdir(parents=True, exist_ok=True)

        return path_dcm, path_nii, path_reslice, path_excel

    def _copy_core_outputs(self, path_nii: Path) -> None:
        """Copy core outputs to nii directory

        Args:
            path_nii: Directory for NIfTI files
        """
        core_files = ["MRA_BRAIN.nii.gz", "Pred.nii.gz", "Vessel.nii.gz"]

        for filename in core_files:
            src = self.working_dir / filename
            dst = path_nii / filename
            if src.exists():
                shutil.copy2(src, dst)

    def _generate_mip(self, path_dcm: Path, path_reslice: Path) -> List[Path]:
        """Generate MIP (Maximum Intensity Projection) images

        Args:
            path_dcm: DICOM directory
            path_reslice: Reslice directory

        Returns:
            List of paths to generated MIP files
        """
        try:
            # Import utility function
            from code_ai.pipeline.chuan.util_aneurysm import create_MIP_pred

            # Copy DICOM directory if provided
            if self.config.input_dicom_dir:
                dicom_dest = path_dcm / "MRA_BRAIN"
                if not dicom_dest.exists():
                    shutil.copytree(self.config.input_dicom_dir, dicom_dest)

            # Find PNG directory (for reference images)
            png_dir = self._find_png_directory()

            # Generate MIP
            create_MIP_pred(str(path_dcm), str(path_reslice), str(png_dir), self.config.gpu_id)

            # Find generated MIP files
            mip_files = list(path_dcm.glob("MIP_*.dcm"))

            self.logger.info(f"Generated {len(mip_files)} MIP files")
            return mip_files

        except Exception as e:
            self.logger.error(f"MIP generation failed: {e}", exc_info=True)
            return []

    def _calculate_metrics(self, path_dcm: Path, path_excel: Path) -> None:
        """Calculate aneurysm metrics (size, location, etc.)

        Args:
            path_dcm: DICOM directory
            path_excel: Excel output directory
        """
        try:
            from code_ai.pipeline.chuan.util_aneurysm import (
                make_aneurysm_vessel_location_16labels_pred,
                calculate_aneurysm_long_axis_make_pred,
                make_table_row_patient_pred,
                make_table_add_location
            )

            # Calculate vessel locations and labels
            make_aneurysm_vessel_location_16labels_pred(str(self.working_dir))

            # Calculate aneurysm long axis
            calculate_aneurysm_long_axis_make_pred(
                str(path_dcm),
                str(self.working_dir),
                str(path_excel),
                self.config.study_id
            )

            # Create summary table
            make_table_row_patient_pred(str(path_excel), self.config.study_id)

            # Add location information
            make_table_add_location(str(self.working_dir), str(path_excel))

            self.logger.info("Aneurysm metrics calculation completed")

        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}", exc_info=True)

    def _create_dicom_seg(
        self,
        path_dcm: Path,
        path_nii: Path,
        path_reslice: Path
    ) -> List[Path]:
        """Create DICOM-seg files

        Args:
            path_dcm: DICOM directory
            path_nii: NIfTI directory
            path_reslice: Reslice directory

        Returns:
            List of paths to created DICOM-seg files
        """
        try:
            from code_ai.pipeline.chuan.create_dicomseg_multi_file_json_claude import (
                create_dicomseg_multi_file
            )

            # Create DICOM-seg directory
            path_dicomseg = path_dcm / "Dicom-Seg"
            path_dicomseg.mkdir(parents=True, exist_ok=True)

            # Create DICOM-seg files
            create_dicomseg_multi_file(
                str(self.config.path_code or Path.cwd()),
                str(path_dcm),
                str(path_nii),
                str(path_reslice),
                str(path_dicomseg),
                self.config.study_id
            )

            # Find created DICOM-seg files
            dicom_seg_files = sorted(path_dicomseg.glob("*.dcm"))

            self.logger.info(f"Created {len(dicom_seg_files)} DICOM-seg files")
            return dicom_seg_files

        except Exception as e:
            self.logger.error(f"DICOM-seg creation failed: {e}", exc_info=True)
            return []

    def _generate_platform_json(
        self,
        path_dcm: Path,
        path_nii: Path,
        path_reslice: Path,
        path_excel: Path
    ) -> Path:
        """Generate platform JSON with results using AneurysmDetectionBuilder

        Args:
            path_dcm: DICOM directory
            path_nii: NIfTI directory
            path_reslice: Reslice directory
            path_excel: Excel directory

        Returns:
            Path to created JSON file
        """
        try:
            self.logger.info("Generating platform JSON using AneurysmDetectionBuilder...")

            # Call builder to generate platform JSON
            platform_json = AneurysmDetectionBuilder.execute_rdx_platform_json(
                _id=self.config.study_id,
                path_root=self.working_dir,
                model_id='5d7b5e3a-9c1f-4a2b-8d6e-3f9a1c2b4e5f'  # Aneurysm model ID
            )

            # Find generated JSON file
            json_file = self.working_dir / 'rdx_aneurysm_json.json'

            if not json_file.exists():
                # Fallback: create basic JSON
                self.logger.warning("Platform JSON not created by builder, generating basic version")
                json_file = self._create_basic_json(self.working_dir)

            return json_file

        except Exception as e:
            self.logger.error(f"Platform JSON generation failed: {e}", exc_info=True)
            # Create basic JSON as fallback
            return self._create_basic_json(self.working_dir)

    def _create_basic_json(self, json_dir: Path) -> Path:
        """Create basic JSON metadata (fallback)

        Args:
            json_dir: Directory for JSON file

        Returns:
            Path to created JSON file
        """
        json_dir.mkdir(parents=True, exist_ok=True)
        json_file = json_dir / f"{self.config.study_id}.json"

        import datetime

        metadata = {
            "study_id": self.config.study_id,
            "model": "aneurysm_detection",
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

    def _find_png_directory(self) -> Path:
        """Find PNG reference directory

        Returns:
            Path to PNG directory

        Note:
            Creates empty directory if not found
        """
        # Try code directory
        if self.config.path_code:
            png_dir = self.config.path_code / "png"
            if png_dir.exists():
                return png_dir

        # Try chuan directory
        chuan_dir = Path(__file__).parent.parent / "chuan" / "code" / "png"
        if chuan_dir.exists():
            return chuan_dir

        # Create temporary PNG directory
        png_dir = self.working_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        return png_dir

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

        elif output_name == "Pred":
            return self.output_manager.get_nifti_path(self.model_name, "Pred_Aneurysm")

        elif output_name == "Prob":
            return self.output_manager.get_nifti_path(self.model_name, "Prob_Aneurysm")

        elif output_name == "Vessel":
            return self.output_manager.get_nifti_path(self.model_name, "Pred_Aneurysm_Vessel")

        elif output_name == "Vessel16":
            return self.output_manager.get_nifti_path(self.model_name, "Pred_Aneurysm_Vessel16")

        elif output_name.startswith("MIP_"):
            return self.output_manager.get_output_path(self.model_name, output_path.name)

        else:
            # Default: keep original filename
            return self.output_manager.get_output_path(self.model_name, output_path.name)


def main():
    """Main entry point for Aneurysm pipeline"""
    # Create argument parser
    parser = create_pipeline_parser(
        description="Aneurysm Detection and Vessel Segmentation Pipeline"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig.from_cli_and_env(args)

    # Create and execute pipeline
    pipeline = AneurysmPipeline(config)
    success = pipeline.execute()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
