#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SynthSeg 5-Class Segmentation Pipeline

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
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from code_ai.pipeline.rdx import BasePipeline, PipelineConfig, create_pipeline_parser
from code_ai.pipeline.dicomseg import dicom_seg_multi_file
from code_ai.pipeline import get_study_id


class SynthsegPipeline(BasePipeline):
    """SynthSeg 5-class brain segmentation pipeline

    Input:
        - Single NIfTI file (any brain MRI modality)

    Outputs (in {output_dir}/{study_id}/vessel_model/):
        - {study_id}.json: Metadata
        - {study_id}.dcm: DICOM-seg file (single)
        - synthseg_*_original_synthseg*.nii.gz: Segmentation results

    Process:
        1. Run synthseg 5-class segmentation
        2. Find output files
        3. Create DICOM-seg (if DICOM directory provided)
        4. Generate metadata JSON
    """

    @property
    def model_name(self) -> str:
        return "vessel"

    def required_inputs(self) -> List[str]:
        return ["SCAN"]

    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Run synthseg 5-class segmentation

        Args:
            inputs: Dictionary with key "SCAN"

        Returns:
            Dictionary mapping output names to file paths

        Raises:
            FileNotFoundError: If synthseg output not found
            subprocess.CalledProcessError: If synthseg fails
        """
        scan_path = inputs["SCAN"]

        self.logger.info("Starting SynthSeg 5-class segmentation pipeline")
        self.logger.info(f"  Input: {scan_path}")

        # Step 1: Run synthseg 5-class segmentation
        self.logger.info("Step 1: Running synthseg 5-class segmentation...")
        self._run_synthseg5class(scan_path)

        # Step 2: Find output files
        self.logger.info("Step 2: Locating synthseg outputs...")
        output_files = self._find_synthseg_outputs()
        self.logger.info(f"  Found {len(output_files)} output file(s)")

        if not output_files:
            raise FileNotFoundError("No synthseg output files found")

        # Use first output as primary
        primary_output = output_files[0]
        self.logger.info(f"  Primary output: {primary_output.name}")

        # Step 3: Create DICOM-seg (if DICOM directory provided)
        dicom_path = None
        if self.config.input_dicom_dir:
            self.logger.info("Step 3: Creating DICOM-seg file...")
            dicom_path = self._create_dicom_seg(primary_output)
            if dicom_path:
                self.logger.info(f"  Created: {dicom_path.name}")
        else:
            self.logger.info("Step 3: Skipping DICOM-seg (no input DICOM directory)")

        # Step 4: Generate metadata JSON
        self.logger.info("Step 4: Generating metadata JSON...")
        json_path = self._generate_json(primary_output, scan_path)
        self.logger.info(f"  Created: {json_path.name}")

        # Collect outputs
        outputs = {
            "JSON": json_path,
        }

        if dicom_path:
            outputs["DICOM"] = dicom_path

        # Add all synthseg outputs
        for i, out_file in enumerate(output_files):
            outputs[f"synthseg_{i}"] = out_file

        self.logger.info(f"SynthSeg inference completed: {len(outputs)} outputs")
        return outputs

    def _run_synthseg5class(self, scan_path: Path) -> None:
        """Run synthseg 5-class segmentation

        Args:
            scan_path: Path to input NIfTI file

        Raises:
            subprocess.CalledProcessError: If synthseg fails
            FileNotFoundError: If main5class.py not found
        """
        # Find main5class.py in pipeline directory
        main5class_py = Path(__file__).parent / "main5class.py"
        if not main5class_py.exists():
            # Try parent directory
            main5class_py = Path(__file__).parent.parent / "main5class.py"

        if not main5class_py.exists():
            raise FileNotFoundError(f"main5class.py not found in {Path(__file__).parent}")

        # Get Python executable
        python_exe = self.config.path_code / "python3" if self.config.path_code else "python3"

        # Build command
        cmd = [
            str(python_exe),
            str(main5class_py),
            "-i", str(scan_path),
            "--output", str(self.working_dir)
        ]

        self.logger.info(f"Running synthseg5class: {' '.join(cmd)}")

        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )

        if result.returncode != 0:
            self.logger.error(f"Synthseg5class failed with return code {result.returncode}")
            self.logger.error(f"STDOUT: {result.stdout}")
            self.logger.error(f"STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)

        self.logger.info("Synthseg5class segmentation completed")

    def _find_synthseg_outputs(self) -> List[Path]:
        """Find all synthseg output files

        Returns:
            List of paths to synthseg output NIfTI files

        Note:
            Returns empty list if no outputs found
        """
        pattern = str(self.working_dir / 'synthseg*_original_synthseg*.nii.gz')
        matches = glob.glob(pattern)

        output_files = [Path(f) for f in matches if Path(f).exists()]

        return sorted(output_files)

    def _create_dicom_seg(self, pred_path: Path) -> Optional[Path]:
        """Create DICOM-seg file from segmentation

        Args:
            pred_path: Path to segmentation NIfTI file

        Returns:
            Path to created DICOM-seg file or None if failed

        Note:
            Requires config.input_dicom_dir to be set
        """
        if not self.config.input_dicom_dir:
            self.logger.warning("No input DICOM directory, skipping DICOM-seg creation")
            return None

        try:
            self.logger.info(f"Creating DICOM-seg from {pred_path}")
            self.logger.info(f"  Source DICOM: {self.config.input_dicom_dir}")

            # Call DICOM-seg creation function
            stdout, stderr = dicom_seg_multi_file(
                self.config.study_id,
                str(self.config.input_dicom_dir),
                str(pred_path),
                str(self.working_dir)
            )

            if stderr:
                self.logger.warning(f"DICOM-seg stderr: {stderr}")

            # Find generated DICOM file (should be single file)
            dicom_files = list(self.working_dir.glob("*.dcm"))

            if dicom_files:
                self.logger.info(f"Created DICOM-seg file: {dicom_files[0].name}")
                return dicom_files[0]
            else:
                self.logger.warning("No DICOM-seg file created")
                return None

        except Exception as e:
            self.logger.error(f"DICOM-seg creation failed: {e}", exc_info=True)
            return None

    def _generate_json(self, pred_path: Path, scan_path: Path) -> Path:
        """Generate metadata JSON file

        Args:
            pred_path: Path to primary segmentation output
            scan_path: Path to original input scan

        Returns:
            Path to created JSON file
        """
        json_path = self.working_dir / f"{self.config.study_id}.json"

        metadata = {
            "study_id": self.config.study_id,
            "model": "synthseg5class",
            "model_name": self.model_name,
            "input_file": str(scan_path.name),
            "prediction_file": str(pred_path.name),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return json_path

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

        elif output_name == "DICOM":
            return self.output_manager.get_dicom_path(self.model_name)

        elif output_name.startswith("synthseg_"):
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
    """Main entry point for SynthSeg pipeline"""
    import datetime

    # Create argument parser
    parser = create_pipeline_parser(
        description="SynthSeg 5-Class Brain Segmentation Pipeline"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig.from_cli_and_env(args)

    # Create and execute pipeline
    pipeline = SynthsegPipeline(config)
    success = pipeline.execute()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
