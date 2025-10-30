# RDX AI Pipeline Framework

Redesigned medical imaging AI pipeline architecture with unified configuration, model-organized outputs, and shared infrastructure.

## Overview

This framework provides base classes for implementing medical imaging AI pipelines with:

- **Unified Configuration**: CLI arguments > Environment variables > Defaults
- **Model-Organized Outputs**: Structured directory layout per model type
- **Shared Infrastructure**: GPU management, logging, output organization
- **Error Handling**: Comprehensive logging and graceful failure handling
- **Testability**: Clean interfaces and dependency injection

## Architecture

```
code_ai/pipeline/rdx/
├── __init__.py           # Package exports
├── README.md             # This file
└── base/                 # Base classes
    ├── __init__.py       # Module exports
    ├── config.py         # PipelineConfig, create_pipeline_parser
    ├── pipeline.py       # BasePipeline (abstract base class)
    ├── output.py         # OutputManager
    ├── gpu.py            # GPUManager
    └── logging_manager.py # LoggingManager, PipelineLogger
```

## Core Components

### 1. PipelineConfig

Configuration management with hierarchy: CLI > ENV > Defaults

```python
from code_ai.pipeline.rdx import PipelineConfig, create_pipeline_parser

# Create argument parser
parser = create_pipeline_parser("My Pipeline")
parser.add_argument('--custom-arg', help='Pipeline-specific argument')
args = parser.parse_args()

# Create configuration
config = PipelineConfig.from_cli_and_env(args)

# Mandatory: study_id via --id
# Optional with fallbacks:
#   - input_files: --input (repeatable)
#   - output_dir: --output-dir > ENV[PATH_OUTPUT] > ./output
#   - working_dir: --working-dir > ENV[PATH_WORK] > /tmp/pipeline_work
#   - gpu_id: --gpu > ENV[GPU_N] > 0
```

### 2. BasePipeline

Abstract base class for pipeline implementations

```python
from code_ai.pipeline.rdx import BasePipeline
from pathlib import Path
from typing import Dict, List

class MyPipeline(BasePipeline):
    @property
    def model_name(self) -> str:
        return "my_model"  # Used for output directory: {study_id}/my_model_model/

    def required_inputs(self) -> List[str]:
        return ["SCAN"]  # Expected input types

    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Model-specific inference logic"""
        scan_path = inputs["SCAN"]

        # Copy to working directory
        work_scan = self.copy_to_working_dir(scan_path, "scan.nii.gz")

        # Run inference
        pred_path = self.working_dir / "prediction.nii.gz"
        # ... inference code ...

        return {
            "Pred": pred_path,
            "JSON": self.working_dir / f"{self.config.study_id}.json"
        }

    def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
        """Map outputs to organized directory structure"""
        if output_name == "JSON":
            return self.output_manager.get_json_path(self.model_name)
        elif output_name == "Pred":
            return self.output_manager.get_nifti_path(self.model_name, "Prediction")
        else:
            return self.output_manager.get_output_path(self.model_name, output_path.name)

# Usage
config = PipelineConfig.from_cli_and_env(args)
pipeline = MyPipeline(config)
success = pipeline.execute()
```

### 3. OutputManager

Manages model-organized output directory structure

```python
from code_ai.pipeline.rdx import OutputManager
from pathlib import Path

manager = OutputManager(Path("/results"), "Study_12345")

# Get model directory: /results/Study_12345/aneurysm_model/
model_dir = manager.get_model_dir("aneurysm")

# Get JSON path: /results/Study_12345/aneurysm_model/Study_12345.json
json_path = manager.get_json_path("aneurysm")

# Get DICOM path: /results/Study_12345/aneurysm_model/Study_12345_A01.dcm
dicom_path = manager.get_dicom_path("aneurysm", slice_num=1)

# Get NIfTI path: /results/Study_12345/aneurysm_model/Pred_Aneurysm.nii.gz
nifti_path = manager.get_nifti_path("aneurysm", "Pred_Aneurysm")
```

### 4. GPUManager

GPU device management and memory monitoring

```python
from code_ai.pipeline.rdx import GPUManager

gpu_manager = GPUManager(gpu_id=0, memory_threshold=0.6)

# Check GPU availability
if gpu_manager.check_gpu_available():
    # Configure TensorFlow
    gpu_manager.configure_tensorflow()

    # Get GPU info
    info = gpu_manager.get_gpu_info()
    print(f"GPU: {info['name']}, Memory: {info['memory_usage_ratio']:.1%}")

    # ... inference ...

    # Cleanup
    gpu_manager.cleanup()
```

### 5. LoggingManager

Unified logging with daily rotation

```python
from code_ai.pipeline.rdx import LoggingManager, PipelineLogger
from pathlib import Path

# Basic usage
logger = LoggingManager.setup_logger(
    log_dir=Path("./logs"),
    pipeline_name="my_pipeline",
    verbose=False
)
logger.info("Pipeline started")
logger.error("Error occurred", exc_info=True)

# Context manager with auto-cleanup
with PipelineLogger("my_pipeline", Path("./logs"), cleanup_days=30) as logger:
    logger.info("Processing...")
    # Exceptions logged automatically
```

## Output Directory Structure

The framework organizes outputs by model type:

```
{output_dir}/{study_id}/
├── aneurysm_model/
│   ├── {study_id}.json
│   ├── {study_id}_A01.dcm
│   ├── {study_id}_A02.dcm
│   ├── Pred_Aneurysm.nii.gz
│   └── Prob_Aneurysm.nii.gz
├── cmb_model/
│   ├── {study_id}.json
│   ├── {study_id}_A01.dcm
│   ├── {study_id}_A02.dcm
│   ├── {study_id}_A03.dcm
│   ├── {study_id}_A04.dcm
│   ├── {study_id}_A05.dcm
│   └── Pred_CMB.nii.gz
└── vessel_model/
    ├── {study_id}.json
    ├── {study_id}.dcm
    └── synthseg_*.nii.gz
```

## CLI Usage

All pipelines use a unified CLI interface:

```bash
# Minimal usage
python pipeline.py --id Study_12345 --input scan.nii.gz

# With custom output directory
python pipeline.py \
  --id Study_12345 \
  --input scan.nii.gz \
  --output-dir /results

# Multiple inputs (e.g., CMB pipeline)
python pipeline.py \
  --id Study_12345 \
  --input swan.nii.gz \
  --input t1.nii.gz

# With DICOM-seg generation
python pipeline.py \
  --id Study_12345 \
  --input scan.nii.gz \
  --input-dicom-dir /dicom/source

# With environment variables
export PATH_OUTPUT=/results
export GPU_N=1
python pipeline.py --id Study_12345 --input scan.nii.gz

# Keep intermediate files for debugging
python pipeline.py \
  --id Study_12345 \
  --input scan.nii.gz \
  --keep-intermediate \
  --verbose
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PATH_OUTPUT` | Base output directory | `./output` |
| `PATH_WORK` | Temporary working directory | `/tmp/pipeline_work` |
| `PATH_CODE` | Code directory | - |
| `PATH_PROCESS` | Processing directory | - |
| `PATH_JSON` | JSON output directory | - |
| `PATH_LOG` | Log directory | `./logs` |
| `GPU_N` | GPU device ID | `0` |
| `GPU_MEMORY_THRESHOLD` | GPU usage threshold | `0.6` |

## Implementation Example

Complete pipeline implementation:

```python
#!/usr/bin/env python3
"""Example pipeline implementation"""
import sys
from pathlib import Path
from typing import Dict, List

from code_ai.pipeline.rdx import BasePipeline, PipelineConfig, create_pipeline_parser


class ExamplePipeline(BasePipeline):
    """Example medical imaging pipeline"""

    @property
    def model_name(self) -> str:
        return "example"

    def required_inputs(self) -> List[str]:
        return ["SCAN"]

    def run_inference(self, inputs: Dict[str, Path]) -> Dict[str, Path]:
        """Run model inference"""
        scan_path = inputs["SCAN"]

        # Copy to working directory
        work_scan = self.copy_to_working_dir(scan_path)

        # Run inference (placeholder)
        self.logger.info(f"Running inference on {work_scan}")
        pred_path = self.working_dir / "prediction.nii.gz"

        # TODO: Actual inference code
        # pred_path = run_model(work_scan)

        # Generate JSON metadata
        import json
        json_path = self.working_dir / f"{self.config.study_id}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "study_id": self.config.study_id,
                "model": self.model_name,
                "input": str(scan_path)
            }, f, indent=2)

        return {
            "Prediction": pred_path,
            "JSON": json_path,
        }

    def _get_output_destination(self, output_name: str, output_path: Path) -> Path:
        """Map outputs to destination paths"""
        if output_name == "JSON":
            return self.output_manager.get_json_path(self.model_name)
        elif output_name == "Prediction":
            return self.output_manager.get_nifti_path(self.model_name, "Prediction")
        else:
            return self.output_manager.get_output_path(self.model_name, output_path.name)


def main():
    # Create parser
    parser = create_pipeline_parser("Example Pipeline")
    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig.from_cli_and_env(args)

    # Create and execute pipeline
    pipeline = ExamplePipeline(config)
    success = pipeline.execute()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
```

## Backward Compatibility

The framework supports legacy CLI arguments as hidden aliases:

- `--ID` → `--id`
- `--Inputs` → `--input`
- `--DicomDir` / `--InputsDicomDir` → `--input-dicom-dir`
- `--Output_folder` → `--output-dir`

## Testing

Example unit test structure:

```python
import pytest
from pathlib import Path
from code_ai.pipeline.rdx import PipelineConfig, OutputManager, GPUManager

def test_config_validation():
    """Test configuration validation"""
    config = PipelineConfig(
        study_id="Test_001",
        input_files=[Path("scan.nii.gz")],
        output_dir=Path("./output")
    )
    # Should raise ValueError for missing file
    with pytest.raises(ValueError):
        config.validate()

def test_output_manager():
    """Test output directory organization"""
    manager = OutputManager(Path("/tmp"), "Study_123")
    json_path = manager.get_json_path("aneurysm")
    assert str(json_path) == "/tmp/Study_123/aneurysm_model/Study_123.json"

def test_gpu_manager():
    """Test GPU availability checking"""
    gpu_manager = GPUManager(gpu_id=0, memory_threshold=0.6)
    available = gpu_manager.check_gpu_available()
    assert isinstance(available, bool)
```

## Best Practices

1. **Configuration**: Always validate configuration before execution
2. **Logging**: Use logger for all output, not print()
3. **Error Handling**: Let exceptions propagate, BasePipeline handles them
4. **Cleanup**: Use working_dir for temporary files, automatically cleaned up
5. **GPU Management**: Check availability before TensorFlow configuration
6. **Output Organization**: Use OutputManager methods for consistent paths

## Migration from Old Pipelines

1. Import base classes:
   ```python
   from code_ai.pipeline.rdx import BasePipeline, PipelineConfig, create_pipeline_parser
   ```

2. Convert main function to pipeline class:
   - Main logic → `run_inference()`
   - Output file mapping → `_get_output_destination()`
   - Define `model_name` and `required_inputs()`

3. Update CLI entry point:
   - Use `create_pipeline_parser()`
   - Create `PipelineConfig.from_cli_and_env()`
   - Instantiate pipeline and call `execute()`

4. Remove manual implementations of:
   - GPU checking (use GPUManager)
   - Logging setup (use LoggingManager)
   - Output directory creation (use OutputManager)

## Support

For issues or questions, see:
- Architecture specification: `claudedocs/pipeline_redesign_architecture.md`
- Code examples in model-specific pipeline implementations
- Unit tests in `tests/code_ai/pipeline/rdx/`
