#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMB (Cerebral Microbleed) Detection Pipeline

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
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from code_ai.pipeline.rdx import BasePipeline, PipelineConfig, create_pipeline_parser
from code_ai.pipeline.cmb import CMBServiceTF
from code_ai.pipeline.dicomseg import dicom_seg_cmb_file
from code_ai.pipeline import get_study_id


class CMBPipeline(BasePipeline):
    """Cerebral Microbleed detection pipeline

    Inputs:
        - SWAN.nii.gz: Susceptibility-weighted imaging
        - T1.nii.gz: T1-weighted imaging (template)

    Outputs (in {output_dir}/{study_id}/cmb_model/):
        - {study_id}.json: Metadata and results
        - {study_id}_A01.dcm through A05.dcm: DICOM-seg slices
        - Pred_CMB.nii.gz: CMB prediction mask
        - synthseg_SWAN_original_CMB.nii.gz: Synthseg intermediate result

    Process:
        1. Run synthseg preprocessing on SWAN and T1
        2. Run CMB classification using CMBServiceTF
        3. Generate DICOM-seg files (5 slices)
        4. Create metadata JSON
    """

    @property
    def model_name(self) -> str:
        return "cmb"

    def required_inputs(self) -> List[str]:
        return ["SWAN", "T1"]

    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Run CMB detection pipeline

        Args:
            inputs: Dictionary with keys "SWAN" and "T1"

        Returns:
            Dictionary mapping output names to file paths

        Raises:
            FileNotFoundError: If synthseg output not found
            subprocess.CalledProcessError: If synthseg fails
        """
        swan_path = inputs["SWAN"]
        t1_path = inputs["T1"]

        self.logger.info("Starting CMB detection pipeline")
        self.logger.info(f"  SWAN: {swan_path}")
        self.logger.info(f"  T1: {t1_path}")

        # Step 1: Run synthseg preprocessing
        self.logger.info("Step 1: Running synthseg preprocessing...")
        self._run_synthseg(swan_path, t1_path)

        # Step 2: Find synthseg output
        self.logger.info("Step 2: Locating synthseg output...")
        temp_path = self._find_synthseg_output()
        self.logger.info(f"  Found: {temp_path.name}")

        # Step 3: Run CMB classification
        self.logger.info("Step 3: Running CMB classification...")
        pred_path, json_path = self._run_cmb_classification(swan_path, temp_path)
        self.logger.info(f"  Prediction: {pred_path}")
        self.logger.info(f"  JSON: {json_path}")

        # Step 4: Create DICOM-seg files (if DICOM directory provided)
        dicom_paths = []
        if self.config.input_dicom_dir:
            self.logger.info("Step 4: Creating DICOM-seg files...")
            dicom_paths = self._create_dicom_seg(pred_path)
            self.logger.info(f"  Created {len(dicom_paths)} DICOM-seg files")
        else:
            self.logger.info("Step 4: Skipping DICOM-seg (no input DICOM directory)")

        # Collect outputs
        outputs = {
            "synthseg_temp": temp_path,
            "Pred_CMB": pred_path,
            "JSON": json_path,
        }

        # Add DICOM-seg slices (CMB typically has 5 slices)
        for i, dcm_path in enumerate(dicom_paths[:5], start=1):
            outputs[f"DICOM_A{i:02d}"] = dcm_path

        self.logger.info(f"CMB inference completed: {len(outputs)} outputs")
        return outputs

    def _run_synthseg(self, swan_path: Path, t1_path: Path) -> None:
        """Run synthseg preprocessing

        Args:
            swan_path: Path to SWAN NIfTI file
            t1_path: Path to T1 NIfTI file

        Raises:
            subprocess.CalledProcessError: If synthseg fails
            FileNotFoundError: If main.py not found
        """
        # Find main.py in pipeline directory
        main_py = Path(__file__).parent / "main.py"
        if not main_py.exists():
            # Try parent directory
            main_py = Path(__file__).parent.parent / "main.py"

        if not main_py.exists():
            raise FileNotFoundError(f"main.py not found in {Path(__file__).parent}")

        # Get Python executable
        python_exe = self.config.path_code / "python3" if self.config.path_code else "python3"

        # Build command
        cmd = [
            str(python_exe),
            str(main_py),
            "-i", str(swan_path),
            "--template", str(t1_path),
            "--output", str(self.working_dir),
            "--all", "False",
            "--CMB", "TRUE"
        ]

        self.logger.info(f"Running synthseg: {' '.join(cmd)}")

        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )

        if result.returncode != 0:
            self.logger.error(f"Synthseg failed with return code {result.returncode}")
            self.logger.error(f"STDOUT: {result.stdout}")
            self.logger.error(f"STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)

        self.logger.info("Synthseg preprocessing completed")

    def _find_synthseg_output(self) -> Path:
        """Find synthseg output file

        Returns:
            Path to synthseg output NIfTI file

        Raises:
            FileNotFoundError: If output not found
        """
        pattern = str(self.working_dir / 'synthseg_*SWAN_original_CMB*.nii.gz')
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(
                f"Synthseg output not found matching pattern: {pattern}\n"
                f"Working directory contents: {list(self.working_dir.iterdir())}"
            )

        temp_path = Path(matches[0])

        if not temp_path.exists():
            raise FileNotFoundError(f"Synthseg output file not found: {temp_path}")

        return temp_path

    def _run_cmb_classification(
        self,
        swan_path: Path,
        temp_path: Path
    ) -> tuple[Path, Path]:
        """Run CMB classification using CMBServiceTF

        Args:
            swan_path: Path to SWAN NIfTI file
            temp_path: Path to synthseg intermediate result

        Returns:
            Tuple of (prediction_path, json_path)
        """
        output_nii = self.working_dir / "Pred_CMB.nii.gz"
        output_json = self.working_dir / "Pred_CMB.json"

        self.logger.info("Running CMB classification service...")

        cmb_service = CMBServiceTF()
        cmb_service.cmb_classify(
            swan_path_str=str(swan_path),
            temp_path_str=str(temp_path),
            output_nii_path_str=str(output_nii),
            output_json_path_str=str(output_json)
        )

        # Verify outputs were created
        if not output_nii.exists():
            raise FileNotFoundError(f"CMB prediction not created: {output_nii}")
        if not output_json.exists():
            raise FileNotFoundError(f"CMB JSON not created: {output_json}")

        return output_nii, output_json

    def _create_dicom_seg(self, pred_path: Path) -> List[Path]:
        """Create DICOM-seg files from prediction

        Args:
            pred_path: Path to prediction NIfTI file

        Returns:
            List of paths to created DICOM-seg files

        Note:
            Requires config.input_dicom_dir to be set
        """
        if not self.config.input_dicom_dir:
            self.logger.warning("No input DICOM directory, skipping DICOM-seg creation")
            return []

        try:
            self.logger.info(f"Creating DICOM-seg from {pred_path}")
            self.logger.info(f"  Source DICOM: {self.config.input_dicom_dir}")

            # Call DICOM-seg creation function
            stdout, stderr = dicom_seg_cmb_file(
                self.config.study_id,
                str(self.config.input_dicom_dir),
                str(pred_path),
                str(self.working_dir)
            )

            if stderr:
                self.logger.warning(f"DICOM-seg stderr: {stderr}")

            # Find generated DICOM files
            dicom_files = sorted(self.working_dir.glob("*.dcm"))

            self.logger.info(f"Created {len(dicom_files)} DICOM-seg files")

            return dicom_files

        except Exception as e:
            self.logger.error(f"DICOM-seg creation failed: {e}", exc_info=True)
            return []

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
            # Extract slice number from output_name (e.g., "DICOM_A01" -> 1)
            slice_num = int(output_name[-2:])
            return self.output_manager.get_dicom_path(self.model_name, slice_num)

        elif output_name == "Pred_CMB":
            return self.output_manager.get_nifti_path(self.model_name, "Pred_CMB")

        elif output_name == "synthseg_temp":
            # Clean up filename: remove study_id and double underscores
            basename = output_path.name
            if self.config.study_id in basename:
                basename = basename.replace(self.config.study_id, '').replace('__', '_')
                # Remove leading underscore if present
                if basename.startswith('_'):
                    basename = basename[1:]

            return self.output_manager.get_output_path(self.model_name, basename)

        else:
            # Default: keep original filename
            return self.output_manager.get_output_path(self.model_name, output_path.name)


def main():
    """Main entry point for CMB pipeline"""
    # Create argument parser
    parser = create_pipeline_parser(
        description="CMB (Cerebral Microbleed) Detection Pipeline"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig.from_cli_and_env(args)

    # Create and execute pipeline
    pipeline = CMBPipeline(config)
    success = pipeline.execute()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
