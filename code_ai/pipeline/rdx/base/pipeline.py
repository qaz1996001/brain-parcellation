#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract base class for medical imaging AI pipelines

Author: Architecture Redesign Team
Created: 2025-10-20
"""
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from .config import PipelineConfig
from .gpu import GPUManager
from .logging_manager import LoggingManager
from .output import OutputManager


class BasePipeline(ABC):
    """Abstract base class for all medical imaging pipelines

    Provides common infrastructure:
        - Configuration management
        - GPU allocation and monitoring
        - Logging setup with daily rotation
        - Output directory organization
        - Error handling and cleanup
        - Execution workflow

    Subclasses must implement:
        - model_name property: Model identifier (e.g., 'aneurysm', 'cmb', 'vessel')
        - required_inputs(): List of required input types
        - run_inference(): Model-specific inference logic
        - _get_output_destination(): Output file path mapping

    Example:
        >>> class MyPipeline(BasePipeline):
        ...     @property
        ...     def model_name(self) -> str:
        ...         return "my_model"
        ...
        ...     def required_inputs(self) -> List[str]:
        ...         return ["SCAN"]
        ...
        ...     def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        ...         # Model-specific inference
        ...         return {"output": Path("/tmp/result.nii.gz")}
        ...
        ...     def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
        ...         return self.output_manager.get_nifti_path(self.model_name, output_name)
        ...
        >>> config = PipelineConfig.from_cli_and_env(args)
        >>> pipeline = MyPipeline(config)
        >>> success = pipeline.execute()
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration

        Args:
            config: Pipeline configuration

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.config.validate()

        # Setup logging
        log_dir = config.path_log or Path('./logs')
        self.logger = LoggingManager.setup_logger(
            log_dir=log_dir,
            pipeline_name=self.model_name,
            verbose=config.verbose
        )

        # Setup GPU manager
        self.gpu_manager = GPUManager(
            gpu_id=config.gpu_id,
            memory_threshold=config.gpu_memory_threshold
        )

        # Setup output manager
        self.output_manager = OutputManager(
            base_output_dir=config.output_dir,
            study_id=config.study_id
        )

        # Setup working directory
        self.working_dir = config.working_dir / self.model_name / config.study_id
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized {self.model_name} pipeline")
        self.logger.info(f"Configuration: {config}")

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier for output organization

        Returns:
            Model name (e.g., 'aneurysm', 'cmb', 'vessel')

        Note:
            Used for:
            - Output directory naming: {study_id}/{model_name}_model/
            - Logger naming
            - Working directory organization
        """
        pass

    @abstractmethod
    def required_inputs(self) -> List[str]:
        """Return list of required input file types/names

        Returns:
            List of input names in expected order

        Example:
            >>> # Single input
            >>> return ["MRA_BRAIN"]
            >>>
            >>> # Multiple inputs
            >>> return ["SWAN", "T1"]

        Note:
            Used to validate and map input files to expected types
        """
        pass

    @abstractmethod
    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Execute model-specific inference

        Args:
            inputs: Dictionary mapping input names to file paths
                   (from required_inputs() mapped to config.input_files)

        Returns:
            Dictionary mapping output names to file paths
            Output files should be in working_dir

        Example:
            >>> def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
            ...     mra_path = inputs["MRA_BRAIN"]
            ...     # Run inference
            ...     pred_path = self.working_dir / "Pred.nii.gz"
            ...     # ... inference code ...
            ...     return {
            ...         "Pred": pred_path,
            ...         "Prob": self.working_dir / "Prob.nii.gz",
            ...         "JSON": self.working_dir / f"{self.config.study_id}.json"
            ...     }

        Raises:
            Should raise appropriate exceptions if inference fails
        """
        pass

    @abstractmethod
    def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
        """Determine destination path for output file

        Args:
            output_name: Name/key from run_inference() output dictionary
            output_path: Source path to output file

        Returns:
            Destination path in model-organized output directory

        Example:
            >>> def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
            ...     if output_name == "JSON":
            ...         return self.output_manager.get_json_path(self.model_name)
            ...     elif output_name == "Pred":
            ...         return self.output_manager.get_nifti_path(self.model_name, "Pred")
            ...     else:
            ...         return self.output_manager.get_output_path(self.model_name, output_path.name)

        Note:
            Use OutputManager methods:
            - get_json_path()
            - get_dicom_path()
            - get_nifti_path()
            - get_output_path()
        """
        pass

    def validate_inputs(self) -> Dict[str, Path]:
        """Validate input files match requirements

        Returns:
            Dictionary mapping input types to file paths

        Raises:
            ValueError: If validation fails with detailed error message

        Note:
            Maps config.input_files to required_inputs() in order
        """
        required = self.required_inputs()
        provided = self.config.input_files

        if len(provided) != len(required):
            raise ValueError(
                f"{self.model_name} pipeline requires {len(required)} input(s) {required}, "
                f"but {len(provided)} were provided: {[str(f) for f in provided]}"
            )

        # Map inputs to names
        input_mapping = {}
        for name, path in zip(required, provided):
            if not path.exists():
                raise ValueError(f"Input file not found: {path}")
            if not path.is_file():
                raise ValueError(f"Input path is not a file: {path}")
            input_mapping[name] = path

        self.logger.info(f"Input validation passed: {input_mapping}")
        return input_mapping

    def execute(self) -> bool:
        """Main execution pipeline with error handling

        Execution flow:
            1. Validate inputs
            2. Check GPU availability
            3. Configure TensorFlow
            4. Run inference (model-specific)
            5. Organize outputs
            6. Cleanup temporary files (unless keep_intermediate=True)

        Returns:
            True if successful, False otherwise

        Note:
            All exceptions are caught, logged, and result in False return
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info(f"Starting {self.model_name} pipeline for {self.config.study_id}")
            self.logger.info("=" * 80)

            # 1. Validate inputs
            self.logger.info("Step 1/6: Validating inputs...")
            inputs = self.validate_inputs()
            self.logger.info(f"✓ Inputs validated: {len(inputs)} file(s)")

            # 2. Check GPU availability
            self.logger.info("Step 2/6: Checking GPU availability...")
            if not self.gpu_manager.check_gpu_available():
                self.logger.error("✗ Insufficient GPU memory")
                return False
            self.logger.info("✓ GPU available")

            # Log GPU info
            gpu_info = self.gpu_manager.get_gpu_info()
            if gpu_info:
                self.logger.info(
                    f"GPU: {gpu_info['name']}, "
                    f"Memory: {gpu_info['memory_used'] / 1e9:.2f}GB / "
                    f"{gpu_info['memory_total'] / 1e9:.2f}GB "
                    f"({gpu_info['memory_usage_ratio']:.1%})"
                )

            # 3. Configure TensorFlow
            self.logger.info("Step 3/6: Configuring TensorFlow GPU...")
            if not self.gpu_manager.configure_tensorflow():
                self.logger.error("✗ TensorFlow GPU configuration failed")
                return False
            self.logger.info("✓ TensorFlow configured")

            # 4. Run inference (model-specific)
            self.logger.info("Step 4/6: Running inference...")
            self.logger.info(f"Working directory: {self.working_dir}")
            outputs = self.run_inference(inputs)
            self.logger.info(f"✓ Inference completed: {len(outputs)} output(s)")

            # 5. Organize outputs
            self.logger.info("Step 5/6: Organizing outputs...")
            self._organize_outputs(outputs)
            self.logger.info("✓ Outputs organized")

            # 6. Cleanup
            if not self.config.keep_intermediate:
                self.logger.info("Step 6/6: Cleaning up temporary files...")
                self._cleanup()
                self.logger.info("✓ Cleanup completed")
            else:
                self.logger.info("Step 6/6: Skipping cleanup (--keep-intermediate enabled)")
                self.logger.info(f"Temporary files kept in: {self.working_dir}")

            self.logger.info("=" * 80)
            self.logger.info(f"✓ {self.model_name} pipeline completed successfully")
            self.logger.info(f"Output directory: {self.output_manager.get_model_dir(self.model_name)}")
            self.logger.info("=" * 80)

            return True

        except KeyboardInterrupt:
            self.logger.warning("Pipeline interrupted by user")
            return False

        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error(f"✗ {self.model_name} pipeline failed: {e}", exc_info=True)
            self.logger.error("=" * 80)
            return False

        finally:
            # Cleanup GPU manager
            self.gpu_manager.cleanup()

    def _organize_outputs(self, outputs: Dict[str, Path]) -> None:
        """Organize output files into model-specific directories

        Args:
            outputs: Dictionary mapping output names to source file paths

        Note:
            - Creates model directory if it doesn't exist
            - Copies files from working directory to organized output directory
            - Uses _get_output_destination() to determine target paths
            - Logs each file copy operation
        """
        copied_count = 0

        for output_name, output_path in outputs.items():
            if not output_path.exists():
                self.logger.warning(
                    f"Output file not found, skipping: {output_name} -> {output_path}"
                )
                continue

            # Determine destination
            dest = self._get_output_destination(output_name, output_path)

            # Ensure destination directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(output_path, dest)
            copied_count += 1

            self.logger.info(f"  {output_name}: {output_path.name} -> {dest}")

        self.logger.info(f"Copied {copied_count}/{len(outputs)} output files")

    def _cleanup(self) -> None:
        """Clean up temporary working directory

        Note:
            - Only called if config.keep_intermediate is False
            - Removes entire working directory tree
            - Logs cleanup operation
        """
        if self.working_dir.exists():
            try:
                shutil.rmtree(self.working_dir)
                self.logger.info(f"Removed working directory: {self.working_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to remove working directory: {e}")

    def copy_to_working_dir(self, source: Path, dest_name: Optional[str] = None) -> Path:
        """Copy file to working directory (helper method)

        Args:
            source: Source file path
            dest_name: Destination filename (default: use source filename)

        Returns:
            Path to copied file in working directory

        Example:
             mra_work = self.copy_to_working_dir(
                 inputs["MRA_BRAIN"], "MRA_BRAIN.nii.gz")
        """
        if dest_name is None:
            dest_name = source.name

        dest = self.working_dir / dest_name
        shutil.copy2(source, dest)
        self.logger.debug(f"Copied to working dir: {source} -> {dest}")

        return dest

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"study_id='{self.config.study_id}', "
            f"gpu_id={self.config.gpu_id})"
        )
