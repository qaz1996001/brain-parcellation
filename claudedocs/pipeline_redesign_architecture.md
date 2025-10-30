# Pipeline Architecture Redesign Specification

## Executive Summary

This document specifies the redesigned architecture for three medical imaging AI pipelines:
- Aneurysm Detection Pipeline
- CMB (Cerebral Microbleed) Detection Pipeline
- SynthSeg Segmentation Pipeline

**Key improvements:**
- Unified configuration system with CLI > ENV > Defaults hierarchy
- Flexible output directory structure organized by model type
- Shared base classes eliminating code duplication
- Consistent error handling and logging
- Maintainable, testable architecture

## Current State Analysis

### Existing Pipelines

| Pipeline | File | Configuration | Issues |
|----------|------|---------------|--------|
| Aneurysm | `chuan/pipeline_aneurysm_tensorflow.py` | Hardcoded paths | 319 lines, mixed configuration, subprocess calls |
| CMB | `pipeline_cmb_tensorflow.py` | Environment variables | Uses service class, better structured |
| SynthSeg | `pipeline_synthseg5class_tensorflow.py` | Environment variables | Simplest, pattern-based output |

### Common Problems
1. **Inconsistent configuration**: Mix of hardcoded paths, environment variables, and CLI args
2. **Code duplication**: GPU management, logging setup repeated in each file
3. **Inflexible output**: Fixed output structure, not model-organized
4. **Limited testability**: Monolithic functions, hard to unit test
5. **Maintenance burden**: Changes require editing multiple files

## Design Requirements

### Functional Requirements

**FR1: Configuration Hierarchy**
- Priority: CLI arguments > Environment variables > Default values
- Mandatory: Study ID must be provided via CLI `--id`
- Optional: Input dir, working dir, output dir with fallbacks

**FR2: Output Directory Structure**
```
{output_dir}/{study_id}/
├── aneurysm_model/
│   ├── {study_id}.json
│   ├── {study_id}_A01.dcm
│   └── {study_id}_A02.dcm
├── vessel_model/
│   ├── {study_id}.json
│   └── {study_id}.dcm
└── cmb_model/
    ├── {study_id}.json
    ├── {study_id}_A01.dcm
    ├── {study_id}_A02.dcm
    ├── {study_id}_A03.dcm
    ├── {study_id}_A04.dcm
    └── {study_id}_A05.dcm
```

**FR3: Unified CLI Interface**
```bash
python pipeline_{model}.py \
  --id <study_id> \           # Mandatory
  --input <file1> \           # Mandatory, repeatable
  --input <file2> \           # For multi-input models
  --output-dir <path> \       # Optional, fallback to ENV or default
  --working-dir <path> \      # Optional
  --input-dicom-dir <path> \  # Optional, for DICOM-seg generation
  --gpu <device_id>           # Optional, default 0
```

**FR4: GPU Management**
- Automatic GPU memory checking
- Configurable memory threshold (default 0.6)
- Graceful degradation on GPU unavailability

### Non-Functional Requirements

**NFR1: Maintainability**
- Single source of truth for common operations
- Clear separation of concerns
- DRY (Don't Repeat Yourself) principle

**NFR2: Testability**
- Unit testable components
- Mock-friendly interfaces
- Dependency injection

**NFR3: Backward Compatibility**
- Support legacy CLI arguments as aliases
- Maintain existing environment variable names
- Optional legacy output structure

**NFR4: Performance**
- No performance degradation vs current implementation
- Efficient resource management
- Proper cleanup of temporary files

## Architecture Design

### Component Hierarchy

```
┌─────────────────────────────────────────┐
│         BasePipeline (Abstract)         │
│  - Configuration management             │
│  - GPU management                       │
│  - Logging setup                        │
│  - Output organization                  │
│  - Error handling                       │
└─────────────────────────────────────────┘
                    ▲
                    │ inherits
        ┌───────────┼───────────┐
        │           │           │
┌───────┴──────┐ ┌──┴────────┐ ┌┴──────────────┐
│  Aneurysm    │ │    CMB    │ │   SynthSeg    │
│  Pipeline    │ │ Pipeline  │ │   Pipeline    │
│              │ │           │ │               │
│ - MRA input  │ │ - SWAN    │ │ - Single NII  │
│ - MIP gen    │ │ - T1      │ │ - Synthseg5   │
│ - Vessel seg │ │ - CMB cls │ │ - Basic seg   │
└──────────────┘ └───────────┘ └───────────────┘
```

### Core Classes

#### 1. PipelineConfig

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import os

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution with CLI > ENV > Default priority"""

    # Mandatory parameters
    study_id: str  # Must be provided via CLI --id

    # Input/Output paths
    input_files: List[Path]
    input_dicom_dir: Optional[Path] = None
    output_dir: Path = None  # Fallback to ENV or ./output
    working_dir: Path = None  # Fallback to ENV or /tmp/pipeline_work

    # GPU configuration
    gpu_id: int = 0  # CLI > ENV[GPU_N] > 0
    gpu_memory_threshold: float = 0.6  # ENV[GPU_MEMORY_THRESHOLD] > 0.6

    # Infrastructure paths (from ENV)
    path_code: Optional[Path] = None      # ENV[PATH_CODE]
    path_process: Optional[Path] = None   # ENV[PATH_PROCESS]
    path_json: Optional[Path] = None      # ENV[PATH_JSON]
    path_log: Optional[Path] = None       # ENV[PATH_LOG]

    # Advanced options
    legacy_output: bool = False  # Use old output structure
    keep_intermediate: bool = False  # Keep temporary files

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'PipelineConfig':
        """Create config with CLI > ENV > Default priority"""
        return cls(
            study_id=args.id,  # Mandatory
            input_files=[Path(f) for f in args.input],
            input_dicom_dir=Path(args.input_dicom_dir) if args.input_dicom_dir else None,
            output_dir=Path(args.output_dir) if args.output_dir else
                       Path(os.getenv('PATH_OUTPUT', './output')),
            working_dir=Path(args.working_dir) if args.working_dir else
                        Path(os.getenv('PATH_WORK', '/tmp/pipeline_work')),
            gpu_id=args.gpu if hasattr(args, 'gpu') else int(os.getenv('GPU_N', 0)),
            gpu_memory_threshold=float(os.getenv('GPU_MEMORY_THRESHOLD', 0.6)),
            path_code=Path(os.getenv('PATH_CODE')) if os.getenv('PATH_CODE') else None,
            path_process=Path(os.getenv('PATH_PROCESS')) if os.getenv('PATH_PROCESS') else None,
            path_json=Path(os.getenv('PATH_JSON')) if os.getenv('PATH_JSON') else None,
            path_log=Path(os.getenv('PATH_LOG')) if os.getenv('PATH_LOG') else None,
            legacy_output=args.legacy_output if hasattr(args, 'legacy_output') else False,
            keep_intermediate=args.keep_intermediate if hasattr(args, 'keep_intermediate') else False,
        )

    def validate(self) -> None:
        """Validate configuration and raise ValueError if invalid"""
        if not self.study_id:
            raise ValueError("study_id is mandatory")
        if not self.input_files:
            raise ValueError("At least one input file is required")
        for input_file in self.input_files:
            if not input_file.exists():
                raise ValueError(f"Input file not found: {input_file}")
```

#### 2. OutputManager

```python
from pathlib import Path
from typing import Optional

class OutputManager:
    """Manages output directory structure per user specification"""

    def __init__(self, base_output_dir: Path, study_id: str):
        self.base_dir = Path(base_output_dir) / study_id
        self.study_id = study_id

    def get_model_dir(self, model_name: str) -> Path:
        """Get model-specific output directory

        Returns: {base_dir}/{model_name}_model/
        Example: /output/Study123/aneurysm_model/
        """
        model_dir = self.base_dir / f"{model_name}_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_output_path(self, model_name: str, filename: str) -> Path:
        """Get full path for output file

        Args:
            model_name: Model identifier (aneurysm, cmb, vessel)
            filename: Output filename (e.g., {study_id}.json, {study_id}_A01.dcm)

        Returns: Full path to output file
        """
        return self.get_model_dir(model_name) / filename

    def get_json_path(self, model_name: str) -> Path:
        """Get path for model metadata JSON"""
        return self.get_output_path(model_name, f"{self.study_id}.json")

    def get_dicom_path(self, model_name: str, slice_num: Optional[int] = None) -> Path:
        """Get path for DICOM-seg output

        Args:
            model_name: Model identifier
            slice_num: Slice number for multi-slice outputs (A01, A02, etc.)

        Returns: Path like {study_id}_A01.dcm or {study_id}.dcm
        """
        if slice_num is not None:
            filename = f"{self.study_id}_A{slice_num:02d}.dcm"
        else:
            filename = f"{self.study_id}.dcm"
        return self.get_output_path(model_name, filename)

    def get_nifti_path(self, model_name: str, suffix: str) -> Path:
        """Get path for NIfTI output

        Args:
            model_name: Model identifier
            suffix: File suffix (e.g., Pred_Aneurysm, Prob_Aneurysm)

        Returns: Path like {model_name}_model/{suffix}.nii.gz
        """
        return self.get_output_path(model_name, f"{suffix}.nii.gz")
```

#### 3. GPUManager

```python
import pynvml
import tensorflow as tf
import logging
from typing import Optional

class GPUManager:
    """Manages GPU device allocation and memory checking"""

    def __init__(self, gpu_id: int = 0, memory_threshold: float = 0.6):
        self.gpu_id = gpu_id
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger(__name__)

    def check_gpu_available(self) -> bool:
        """Check if GPU has sufficient memory available

        Returns: True if GPU memory usage < threshold
        """
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_usage = memory_info.used / memory_info.total

            available = gpu_usage < self.memory_threshold
            if not available:
                self.logger.warning(
                    f"GPU {self.gpu_id} usage {gpu_usage:.2%} exceeds threshold {self.memory_threshold:.2%}"
                )
            return available

        except Exception as e:
            self.logger.error(f"GPU check failed: {e}")
            return False

    def configure_tensorflow(self) -> None:
        """Configure TensorFlow for GPU usage"""
        try:
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            if not gpus:
                self.logger.warning("No GPU devices found")
                return

            target_gpu = gpus[self.gpu_id]
            tf.config.experimental.set_visible_devices(devices=target_gpu, device_type='GPU')
            tf.config.experimental.set_memory_growth(target_gpu, True)

            self.logger.info(f"TensorFlow configured for GPU {self.gpu_id}")

        except Exception as e:
            self.logger.error(f"TensorFlow GPU configuration failed: {e}")
            raise
```

#### 4. LoggingManager

```python
import logging
import time
from pathlib import Path

class LoggingManager:
    """Unified logging configuration for all pipelines"""

    @staticmethod
    def setup_logger(log_dir: Path, pipeline_name: str) -> logging.Logger:
        """Setup logger with daily log files

        Args:
            log_dir: Directory for log files
            pipeline_name: Pipeline identifier for logger name

        Returns: Configured logger instance
        """
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create daily log file
        localt = time.localtime(time.time())
        time_str = f"{localt.tm_year}{localt.tm_mon:02d}{localt.tm_mday:02d}"
        log_file = log_dir / f"{time_str}.log"

        # Configure logging
        logger = logging.getLogger(pipeline_name)
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
```

#### 5. BasePipeline (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import shutil

class BasePipeline(ABC):
    """Abstract base class for all medical imaging pipelines

    Provides common infrastructure:
    - Configuration management
    - GPU allocation
    - Logging setup
    - Output organization
    - Error handling
    - Cleanup
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.validate()

        # Setup infrastructure
        log_dir = config.path_log or Path('./logs')
        self.logger = LoggingManager.setup_logger(log_dir, self.model_name())
        self.gpu_manager = GPUManager(config.gpu_id, config.gpu_memory_threshold)
        self.output_manager = OutputManager(config.output_dir, config.study_id)

        # Working directory
        self.working_dir = config.working_dir / self.model_name() / config.study_id
        self.working_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier for output organization"""
        pass

    @abstractmethod
    def required_inputs(self) -> List[str]:
        """Return list of required input file patterns"""
        pass

    @abstractmethod
    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Execute model inference

        Args:
            inputs: Dictionary mapping input names to file paths

        Returns: Dictionary mapping output names to file paths
        """
        pass

    def validate_inputs(self) -> Dict[str, Path]:
        """Validate input files match requirements

        Returns: Dictionary mapping input types to file paths
        Raises: ValueError if validation fails
        """
        required = self.required_inputs()
        provided = self.config.input_files

        if len(provided) != len(required):
            raise ValueError(
                f"{self.model_name()} requires {len(required)} inputs {required}, "
                f"but {len(provided)} were provided"
            )

        # Map inputs to names
        input_mapping = {}
        for name, path in zip(required, provided):
            if not path.exists():
                raise ValueError(f"Input file not found: {path}")
            input_mapping[name] = path

        return input_mapping

    def execute(self) -> bool:
        """Main execution pipeline with error handling

        Returns: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting {self.model_name()} pipeline for {self.config.study_id}")

            # 1. Validate inputs
            self.logger.info("Validating inputs...")
            inputs = self.validate_inputs()

            # 2. Check GPU availability
            if not self.gpu_manager.check_gpu_available():
                self.logger.error("Insufficient GPU memory")
                return False

            # 3. Configure TensorFlow
            self.logger.info("Configuring GPU...")
            self.gpu_manager.configure_tensorflow()

            # 4. Run inference (model-specific)
            self.logger.info("Running inference...")
            outputs = self.run_inference(inputs)

            # 5. Organize outputs
            self.logger.info("Organizing outputs...")
            self._organize_outputs(outputs)

            # 6. Cleanup
            if not self.config.keep_intermediate:
                self.logger.info("Cleaning up temporary files...")
                self._cleanup()

            self.logger.info(f"{self.model_name()} pipeline completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"{self.model_name()} pipeline failed: {e}", exc_info=True)
            return False

    def _organize_outputs(self, outputs: Dict[str, Path]) -> None:
        """Organize output files into model-specific directories"""
        for output_name, output_path in outputs.items():
            if output_path.exists():
                # Determine destination based on file type
                dest = self._get_output_destination(output_name, output_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(output_path, dest)
                self.logger.info(f"Copied {output_name}: {output_path} -> {dest}")

    @abstractmethod
    def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
        """Determine destination path for output file"""
        pass

    def _cleanup(self) -> None:
        """Clean up temporary working directory"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
            self.logger.info(f"Cleaned up working directory: {self.working_dir}")
```

### Model-Specific Implementations

#### AneurysmPipeline

```python
class AneurysmPipeline(BasePipeline):
    """Aneurysm detection and vessel segmentation pipeline

    Input: MRA_BRAIN.nii.gz
    Outputs:
        - aneurysm_model/{study_id}.json
        - aneurysm_model/{study_id}_A01.dcm, A02.dcm (DICOM-seg slices)
        - aneurysm_model/Pred_Aneurysm.nii.gz
        - aneurysm_model/Prob_Aneurysm.nii.gz
        - aneurysm_model/Pred_Aneurysm_Vessel.nii.gz
        - aneurysm_model/Pred_Aneurysm_Vessel16.nii.gz
    """

    @property
    def model_name(self) -> str:
        return "aneurysm"

    def required_inputs(self) -> List[str]:
        return ["MRA_BRAIN"]

    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Run aneurysm detection pipeline

        Process:
        1. Copy MRA to working directory
        2. Run GPU inference (subprocess to gpu_aneurysm.py)
        3. Generate MIP images
        4. Calculate aneurysm metrics
        5. Create DICOM-seg files
        6. Generate platform JSON
        """
        mra_path = inputs["MRA_BRAIN"]

        # Copy input to working directory
        work_mra = self.working_dir / "MRA_BRAIN.nii.gz"
        shutil.copy(mra_path, work_mra)

        # Run GPU inference
        self._run_gpu_inference()

        # Generate MIP images
        self._generate_mip()

        # Calculate metrics
        self._calculate_metrics()

        # Create DICOM-seg
        dicom_seg_paths = self._create_dicom_seg()

        # Generate JSON
        json_path = self._generate_json()

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

        return outputs

    def _run_gpu_inference(self) -> None:
        """Execute GPU inference via subprocess"""
        import subprocess

        cmd = [
            "python", "gpu_aneurysm.py",
            "--path_code", str(self.config.path_code),
            "--path_process", str(self.working_dir),
            "--case", self.config.study_id,
            "--path_log", str(self.config.path_log),
            "--gpu_n", str(self.config.gpu_id)
        ]

        self.logger.info(f"Running GPU inference: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def _generate_mip(self) -> None:
        """Generate MIP (Maximum Intensity Projection) images"""
        # Implementation from util_aneurysm.create_MIP_pred
        pass

    def _calculate_metrics(self) -> None:
        """Calculate aneurysm metrics and location"""
        # Implementation from util_aneurysm functions:
        # - make_aneurysm_vessel_location_16labels_pred
        # - calculate_aneurysm_long_axis_make_pred
        # - make_table_row_patient_pred
        pass

    def _create_dicom_seg(self) -> List[Path]:
        """Create DICOM-seg files"""
        # Implementation from create_dicomseg_multi_file
        pass

    def _generate_json(self) -> Path:
        """Generate platform JSON metadata"""
        # Implementation from make_pred_json
        pass

    def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
        """Map output files to destination paths"""
        if output_name == "JSON":
            return self.output_manager.get_json_path(self.model_name)
        elif output_name.startswith("DICOM_A"):
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
        else:
            return self.output_manager.get_output_path(self.model_name, output_path.name)
```

#### CMBPipeline

```python
class CMBPipeline(BasePipeline):
    """Cerebral Microbleed detection pipeline

    Inputs: SWAN.nii.gz, T1.nii.gz
    Outputs:
        - cmb_model/{study_id}.json
        - cmb_model/{study_id}_A01.dcm through A05.dcm
        - cmb_model/Pred_CMB.nii.gz
        - cmb_model/synthseg_temp.nii.gz
    """

    @property
    def model_name(self) -> str:
        return "cmb"

    def required_inputs(self) -> List[str]:
        return ["SWAN", "T1"]

    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Run CMB detection pipeline

        Process:
        1. Run synthseg on SWAN and T1
        2. Run CMB classification
        3. Create DICOM-seg files
        4. Generate JSON
        """
        swan_path = inputs["SWAN"]
        t1_path = inputs["T1"]

        # Run synthseg preprocessing
        self._run_synthseg(swan_path, t1_path)

        # Find synthseg output
        import glob
        temp_files = glob.glob(str(self.working_dir / 'synthseg_*SWAN_original_CMB*.nii.gz'))
        if not temp_files:
            raise FileNotFoundError("Synthseg output not found")
        temp_path = Path(temp_files[0])

        # Run CMB classification
        pred_path, json_path = self._run_cmb_classification(swan_path, temp_path)

        # Create DICOM-seg
        dicom_paths = self._create_dicom_seg(pred_path)

        # Collect outputs
        outputs = {
            "synthseg_temp": temp_path,
            "Pred_CMB": pred_path,
            "JSON": json_path,
        }

        # Add DICOM-seg slices (CMB has 5 slices)
        for i, dcm_path in enumerate(dicom_paths[:5], start=1):
            outputs[f"DICOM_A{i:02d}"] = dcm_path

        return outputs

    def _run_synthseg(self, swan_path: Path, t1_path: Path) -> None:
        """Run synthseg preprocessing"""
        import subprocess

        cmd = [
            self.config.path_code / "python3",
            self.config.path_code / "main.py",
            "-i", str(swan_path),
            "--template", str(t1_path),
            "--output", str(self.working_dir),
            "--all", "False",
            "--CMB", "TRUE"
        ]

        subprocess.run(cmd, check=True)

    def _run_cmb_classification(self, swan_path: Path, temp_path: Path) -> tuple[Path, Path]:
        """Run CMB classification using CMBServiceTF"""
        from code_ai.pipeline.cmb import CMBServiceTF

        output_nii = self.working_dir / "Pred_CMB.nii.gz"
        output_json = self.working_dir / "Pred_CMB.json"

        cmb_service = CMBServiceTF()
        cmb_service.cmb_classify(
            swan_path_str=str(swan_path),
            temp_path_str=str(temp_path),
            output_nii_path_str=str(output_nii),
            output_json_path_str=str(output_json)
        )

        return output_nii, output_json

    def _create_dicom_seg(self, pred_path: Path) -> List[Path]:
        """Create DICOM-seg files from prediction"""
        from code_ai.pipeline.dicomseg import dicom_seg_cmb_file

        dicom_seg_cmb_file(
            self.config.study_id,
            str(self.config.input_dicom_dir),
            str(pred_path),
            str(self.working_dir)
        )

        # Find generated DICOM files
        import glob
        return [Path(p) for p in glob.glob(str(self.working_dir / "*.dcm"))]

    def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
        """Map output files to destination paths"""
        if output_name == "JSON":
            return self.output_manager.get_json_path(self.model_name)
        elif output_name.startswith("DICOM_A"):
            slice_num = int(output_name[-2:])
            return self.output_manager.get_dicom_path(self.model_name, slice_num)
        elif output_name == "Pred_CMB":
            return self.output_manager.get_nifti_path(self.model_name, "Pred_CMB")
        elif output_name == "synthseg_temp":
            # Extract meaningful filename from pattern
            basename = output_path.name.replace(self.config.study_id, '').replace('__', '_')
            return self.output_manager.get_output_path(self.model_name, basename)
        else:
            return self.output_manager.get_output_path(self.model_name, output_path.name)
```

#### SynthsegPipeline

```python
class SynthsegPipeline(BasePipeline):
    """SynthSeg 5-class segmentation pipeline

    Input: Single NIfTI file
    Outputs:
        - vessel_model/{study_id}.json
        - vessel_model/{study_id}.dcm
        - vessel_model/synthseg_*.nii.gz
    """

    @property
    def model_name(self) -> str:
        return "vessel"

    def required_inputs(self) -> List[str]:
        return ["SCAN"]

    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Run synthseg 5-class segmentation

        Process:
        1. Run synthseg5class inference
        2. Find output files
        3. Create DICOM-seg
        4. Generate JSON
        """
        scan_path = inputs["SCAN"]

        # Run synthseg5class
        self._run_synthseg5class(scan_path)

        # Find output files
        import glob
        output_files = glob.glob(str(self.working_dir / 'synthseg*_original_synthseg*.nii.gz'))

        if not output_files:
            raise FileNotFoundError("Synthseg output not found")

        # Create DICOM-seg from first output
        primary_output = Path(output_files[0])
        dicom_path = self._create_dicom_seg(primary_output)

        # Generate JSON
        json_path = self._generate_json(primary_output)

        # Collect outputs
        outputs = {
            "JSON": json_path,
            "DICOM": dicom_path,
        }

        # Add all synthseg outputs
        for i, out_file in enumerate(output_files):
            outputs[f"synthseg_{i}"] = Path(out_file)

        return outputs

    def _run_synthseg5class(self, scan_path: Path) -> None:
        """Run synthseg 5-class segmentation"""
        import subprocess

        cmd = [
            self.config.path_code / "python3",
            self.config.path_code / "main5class.py",
            "-i", str(scan_path),
            "--output", str(self.working_dir)
        ]

        subprocess.run(cmd, check=True)

    def _create_dicom_seg(self, pred_path: Path) -> Path:
        """Create DICOM-seg file"""
        from code_ai.pipeline.dicomseg import dicom_seg_multi_file

        dicom_seg_multi_file(
            self.config.study_id,
            str(self.config.input_dicom_dir),
            str(pred_path),
            str(self.working_dir)
        )

        # Find generated DICOM file
        import glob
        dicom_files = glob.glob(str(self.working_dir / "*.dcm"))
        return Path(dicom_files[0]) if dicom_files else None

    def _generate_json(self, pred_path: Path) -> Path:
        """Generate metadata JSON"""
        import json

        json_path = self.working_dir / f"{self.config.study_id}.json"

        metadata = {
            "study_id": self.config.study_id,
            "model": "synthseg5class",
            "prediction_file": pred_path.name,
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return json_path

    def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
        """Map output files to destination paths"""
        if output_name == "JSON":
            return self.output_manager.get_json_path(self.model_name)
        elif output_name == "DICOM":
            return self.output_manager.get_dicom_path(self.model_name)
        elif output_name.startswith("synthseg_"):
            # Clean up filename
            basename = output_path.name.replace(self.config.study_id, '').replace('__', '_')
            return self.output_manager.get_output_path(self.model_name, basename)
        else:
            return self.output_manager.get_output_path(self.model_name, output_path.name)
```

### CLI Entry Points

Each pipeline has a unified CLI entry point:

```python
# pipeline_aneurysm.py
def main():
    parser = argparse.ArgumentParser(description="Aneurysm Detection Pipeline")
    parser.add_argument('--id', required=True, help='Study/Patient ID')
    parser.add_argument('--input', action='append', required=True, help='Input file path(s)')
    parser.add_argument('--input-dicom-dir', help='Input DICOM directory for DICOM-seg')
    parser.add_argument('--output-dir', help='Base output directory')
    parser.add_argument('--working-dir', help='Temporary working directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--keep-intermediate', action='store_true', help='Keep temporary files')
    parser.add_argument('--legacy-output', action='store_true', help='Use legacy output structure')

    # Backward compatibility aliases
    parser.add_argument('--ID', dest='id', help=argparse.SUPPRESS)
    parser.add_argument('--Inputs', dest='input', action='append', help=argparse.SUPPRESS)
    parser.add_argument('--DicomDir', dest='input_dicom_dir', help=argparse.SUPPRESS)
    parser.add_argument('--Output_folder', dest='output_dir', help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig.from_cli_and_env(args)

    # Create and execute pipeline
    pipeline = AneurysmPipeline(config)
    success = pipeline.execute()

    return 0 if success else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
```

## Migration Strategy

### Phase 1: Foundation (Week 1)
- Create `code_ai/pipeline/base/` module
- Implement core classes: `PipelineConfig`, `OutputManager`, `GPUManager`, `LoggingManager`
- Implement `BasePipeline` abstract class
- Unit tests for all base components

### Phase 2: CMB Pipeline (Week 2)
- Implement `CMBPipeline` class
- Refactor `pipeline_cmb_tensorflow.py` to use new architecture
- Integration tests with existing CMB model
- Validate output structure matches specification

### Phase 3: Synthseg Pipeline (Week 3)
- Implement `SynthsegPipeline` class
- Refactor `pipeline_synthseg5class_tensorflow.py`
- Integration tests
- Documentation updates

### Phase 4: Aneurysm Pipeline (Week 4)
- Implement `AneurysmPipeline` class
- Refactor `pipeline_aneurysm_tensorflow.py`
- Most complex due to subprocess calls and multiple outputs
- Comprehensive integration tests

### Phase 5: Deployment (Week 5)
- Backward compatibility testing
- Performance benchmarking
- Migration documentation
- Staged rollout

## Testing Strategy

### Unit Tests
```python
# test_pipeline_config.py
def test_config_cli_priority():
    """Test CLI arguments override environment variables"""

def test_config_mandatory_id():
    """Test that missing ID raises error"""

def test_config_input_validation():
    """Test that missing input files raise error"""

# test_output_manager.py
def test_output_directory_structure():
    """Test output paths match specification"""

def test_model_directory_creation():
    """Test automatic directory creation"""

# test_gpu_manager.py
def test_gpu_memory_check():
    """Test GPU availability checking"""

def test_tensorflow_configuration():
    """Test TensorFlow GPU configuration"""
```

### Integration Tests
```python
# test_cmb_pipeline.py
def test_cmb_pipeline_execution():
    """Test end-to-end CMB pipeline with sample data"""

def test_cmb_output_structure():
    """Test CMB outputs match specification"""

# test_backward_compatibility.py
def test_legacy_cli_arguments():
    """Test old-style CLI arguments still work"""

def test_legacy_output_mode():
    """Test --legacy-output flag produces old structure"""
```

## Documentation

### User Documentation
- CLI usage guide with examples
- Migration guide from old to new interface
- Configuration reference
- Troubleshooting guide

### Developer Documentation
- Architecture overview
- Adding new pipelines (extending BasePipeline)
- API reference for base classes
- Contributing guidelines

## Performance Considerations

### No Performance Degradation
- Base classes add minimal overhead (<1% execution time)
- GPU management unchanged from current implementation
- No additional data copying beyond organization step
- Subprocess execution pattern preserved for compatibility

### Improvements
- Better error handling reduces failed runs
- Automatic cleanup reduces disk usage
- Structured logging improves debugging
- Configuration validation catches errors early

## Risks and Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking changes in refactor | High | Medium | Comprehensive testing, backward compatibility |
| Performance regression | High | Low | Benchmarking before/after, preserve critical paths |
| Missing edge cases | Medium | Medium | Thorough integration tests, staged rollout |
| Configuration complexity | Low | Low | Clear documentation, validation with helpful errors |

## Acceptance Criteria

### Functional Requirements
- ✅ All three pipelines refactored to use new architecture
- ✅ Output structure matches user specification exactly
- ✅ CLI interface consistent across all pipelines
- ✅ Configuration hierarchy: CLI > ENV > Defaults
- ✅ Mandatory ID parameter enforced

### Non-Functional Requirements
- ✅ No performance regression vs current implementation
- ✅ 100% test coverage for base classes
- ✅ Integration tests pass for all pipelines
- ✅ Backward compatibility maintained
- ✅ Documentation complete

### Quality Gates
- ✅ Code review approved
- ✅ All tests passing
- ✅ Performance benchmarks meet targets
- ✅ Documentation reviewed

## Appendix

### Environment Variables Reference

| Variable | Purpose | Default | Used By |
|----------|---------|---------|---------|
| `PATH_OUTPUT` | Base output directory | `./output` | All pipelines |
| `PATH_WORK` | Temporary working directory | `/tmp/pipeline_work` | All pipelines |
| `PATH_CODE` | Code directory | - | Infrastructure |
| `PATH_PROCESS` | Processing directory | - | Infrastructure |
| `PATH_JSON` | JSON output directory | - | Infrastructure |
| `PATH_LOG` | Log directory | `./logs` | All pipelines |
| `GPU_N` | GPU device ID | `0` | All pipelines |
| `GPU_MEMORY_THRESHOLD` | GPU usage threshold | `0.6` | All pipelines |

### CLI Examples

```bash
# Aneurysm pipeline with all options
python pipeline_aneurysm.py \
  --id Study_12345 \
  --input /data/MRA_BRAIN.nii.gz \
  --input-dicom-dir /data/dicom/MRA \
  --output-dir /results \
  --working-dir /tmp/work \
  --gpu 0

# CMB pipeline with environment variables
export PATH_OUTPUT=/results
export GPU_N=1
python pipeline_cmb.py \
  --id Study_12345 \
  --input /data/SWAN.nii.gz \
  --input /data/T1.nii.gz

# Synthseg pipeline minimal
python pipeline_synthseg.py \
  --id Study_12345 \
  --input /data/scan.nii.gz
```

### Output Structure Example

```
/results/Study_12345/
├── aneurysm_model/
│   ├── Study_12345.json
│   ├── Study_12345_A01.dcm
│   ├── Study_12345_A02.dcm
│   ├── Pred_Aneurysm.nii.gz
│   ├── Prob_Aneurysm.nii.gz
│   ├── Pred_Aneurysm_Vessel.nii.gz
│   └── Pred_Aneurysm_Vessel16.nii.gz
├── cmb_model/
│   ├── Study_12345.json
│   ├── Study_12345_A01.dcm
│   ├── Study_12345_A02.dcm
│   ├── Study_12345_A03.dcm
│   ├── Study_12345_A04.dcm
│   ├── Study_12345_A05.dcm
│   ├── Pred_CMB.nii.gz
│   └── synthseg_SWAN_original_CMB.nii.gz
└── vessel_model/
    ├── Study_12345.json
    ├── Study_12345.dcm
    └── synthseg_original_synthseg.nii.gz
```

---

## Summary

This architecture redesign provides:

1. **Unified Configuration**: CLI > ENV > Defaults hierarchy with mandatory ID
2. **Flexible Output Structure**: Model-organized directories matching user specification
3. **Shared Infrastructure**: Eliminate code duplication across pipelines
4. **Maintainability**: Clear separation of concerns, testable components
5. **Backward Compatibility**: Support legacy interfaces during transition
6. **Extensibility**: Easy to add new pipelines by extending BasePipeline

The phased migration approach minimizes risk while delivering incremental value. All acceptance criteria are testable and measurable.
