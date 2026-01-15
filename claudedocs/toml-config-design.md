# TOML Configuration Design

## Overview

This document presents a consolidated TOML configuration structure to externalize all magic numbers and strings identified during the Linus Torvalds philosophy code review.

---

## File Structure

```
config/
├── application.toml      # Main application configuration
├── labels/               # External label mapping files
│   ├── white_matter.json
│   ├── cmb.json
│   ├── dwi.json
│   └── wmh.json
└── series_sort_order.json  # Series sorting configuration
```

---

## application.toml

```toml
# =============================================================================
# Brain Parcellation System - Central Configuration
# =============================================================================
# This file consolidates all magic numbers and configurable parameters
# following the Linus Torvalds philosophy: "Good programmers worry about
# data structures and their relationships."
# =============================================================================

[meta]
version = "1.0.0"
description = "Brain parcellation system configuration"

# =============================================================================
# HTTP & Network Configuration
# =============================================================================

[http]
# Shared timeout settings (eliminating DRY violations)
default_timeout_seconds = 180
long_timeout_seconds = 300
subprocess_timeout_seconds = 600
async_result_timeout_seconds = 3600

[http.retry]
max_attempts = 3
interval_seconds = 20

# =============================================================================
# Redis Cache Configuration
# =============================================================================

[redis]
default_ttl_seconds = 21600  # 6 hours
rpc_result_expire_seconds = 1800  # 30 minutes

# =============================================================================
# RabbitMQ / Task Queue Configuration
# =============================================================================

[rabbitmq]
broker_kind = "RABBITMQ_AMQPSTORM"
concurrent_mode = "THREADING"
default_concurrent_num = 10
ai_concurrent_num = 1  # GPU mutual exclusion

[rabbitmq.queues]
# Centralized queue names (eliminating string duplication)
pipeline_inference = "task_pipeline_inference_queue"
dcm2niix = "call_dcm2niix_queue"
dicom_2_nii_file = "dicom_2_nii_file_queue"
dicom_2_nii_series = "dicom_2_nii_series_queue"
process_instances = "process_instances_queue"
process_dir = "process_dir_queue"
dicom_to_nii = "dicom_to_nii_queue"
dicom_rename = "dicom_rename_queue"
scheduler = "add_raw_dicom_to_nii_inference_queue"
delete_old_data = "delete_old_data_queue"

[rabbitmq.qps]
# Queue processing rates
pipeline_inference = 1   # GPU mutual exclusion
post_httpx = 5
dcm2niix = 10
standard = 10
high_throughput = 100

# =============================================================================
# Tool Identifiers
# =============================================================================

[tools]
# Centralized tool IDs (used across sync, inference, task modules)
dicom = "DICOM_TOOL"
nifti = "NIFTI_TOOL"
inference = "INFERENCE_TOOL"
series_inference = "SERIES_INFERENCE_TOOL"

# =============================================================================
# Status Codes
# =============================================================================

[status_codes]
pending = "100.020"
running = "200.100"
complete = "200.150"
failed = "300.055"

# =============================================================================
# File System Configuration
# =============================================================================

[files]
min_nifti_size_bytes = 500
meta_folder = ".meta"
prediction_filename = "prediction.json"
cmd_tools_dir = "Deep_cmd_tools"
datetime_format = "%Y-%m-%d %H:%M:%S"

[files.extensions]
nifti_gz = ".nii.gz"
nifti = ".nii"
json = ".json"
mgz = ".mgz"
npz = ".npz"
txt = ".txt"
csv = ".csv"

[files.output_suffixes]
david = "_david.nii.gz"
cmb = "_CMB.nii.gz"
dwi = "_DWI.nii.gz"
wmh = "_WMH.nii.gz"

# =============================================================================
# Pagination Defaults
# =============================================================================

[pagination]
default_limit = 50
default_offset = 0
batch_size = 20
max_limit = 1000

# =============================================================================
# Validation Configuration
# =============================================================================

[validation]
uuid_length = 36
uuid_hyphen_count = 4
dicom_completeness_threshold = 0.9
dicom_sample_limit = 100
recent_file_threshold_seconds = 300

# =============================================================================
# DWI Processing
# =============================================================================

[dwi]
source_series = "DWI"
target_series = ["DWI0", "DWI1000"]

# =============================================================================
# Infarct Model
# =============================================================================

[models.infarct]
target_series = ["ADC", "DWI0", "DWI1000"]

# =============================================================================
# Legacy UUID to Model Mapping
# =============================================================================
# CRITICAL: Moved from hardcoded values in task_pipeline.py

[models.legacy_uuid]
"48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6" = "CMB"
"924d1538-597c-41d6-bc27-4b0b359111cf" = "Aneurysm"
"7e94d381-3f5d-46b6-b440-e5d44ebc48d2" = "WMH"
"97abe75d-34de-4e91-80c2-ce74b6c70438" = "Infarct"

# =============================================================================
# SynthSeg Configuration
# =============================================================================

[synthseg]
input_size = 192  # Fixed typo from "intput_size"
crop = 192
min_pad = 128
n_neutral_labels = 19
target_resolution_mm = 1.0
resolution_tolerance = 0.05

[synthseg.config_flags]
robust = true
parc = true
fast = false
v1 = false
ct = false

[synthseg.normalization]
ct_clip_min = 0
ct_clip_max = 80
rescale_min = 0.0
rescale_max = 1.0
percentile_min = 0.5
percentile_max = 99.5

[synthseg.thresholds]
posterior = 0.25
posterior_fast = 0.2
parcellation_mask = 0.1
gaussian_sigma = 0.5
lr_average_weight = 0.5
volume_precision = 3
verbose_threshold = 10

[synthseg.unet]
levels = 5
conv_per_level = 2
conv_size = 3
features = 24
feat_mult = 2
activation = "elu"
batch_norm = -1

[synthseg.unet_denoiser]
conv_size = 5
features = 16
skip_n_concatenations = 2

[synthseg.models]
standard = "synthseg_2.0.h5"
robust = "synthseg_robust_2.0.h5"
parcellation = "synthseg_parc_2.0.h5"

[synthseg.labels]
segmentation = "synthseg_segmentation_labels_2.0.npy"
denoiser = "synthseg_denoiser_labels_2.0.npy"
parcellation = "synthseg_parcellation_labels.npy"
names = "synthseg_parcellation_names.npy"
topology = "synthseg_topological_classes_2.0.npy"
david = "synthseg_segmentation_labels_2.0_david2.npy"

[synthseg.dtypes]
segmentation = "int32"
posteriors = "float32"
mask = "bool"
volume = "int"

[synthseg.layer_names]
split = "split"
concat = "concat"
average_lr = "average_lr"
gradient_operator = "sobel"

# =============================================================================
# Parcellation Configuration
# =============================================================================

[parcellation]
depth_number = 5
depth_number_min = 4
depth_number_max = 10
decimal_places = 8
inner_size = 2
outer_size = 5

[parcellation.thresholds]
x_axis_boundary = 0.97
ventricle_percentile_low = 20
ventricle_percentile_high = 80
prerequisite = 0.5
hemi_revision_markers = [100, 200]
hemisphere_boundaries = [1000, 2000, 3000]

[parcellation.labels]
brain_stem = 16
brain_stem_output = 301
cortex_left = 3
cortex_right = 42
mask_indices = [2, 20]

[parcellation.label_files]
# External JSON files for large label mappings
white_matter = "config/labels/white_matter.json"
cmb = "config/labels/cmb.json"
dwi = "config/labels/dwi.json"
wmh = "config/labels/wmh.json"

# =============================================================================
# Evaluate Configuration
# =============================================================================

[evaluate]
histogram_bin_offset = 0.1
dice_epsilon = 1e-5
default_hausdorff_percentile = 100
default_crop_margin = 10
progress_update_interval = 10
default_hypothesis = "two-sided"
affine_matrix_dim = 4

# =============================================================================
# FSL Configuration (External Tool)
# =============================================================================

[fsl]
# CRITICAL: Use environment variable, not hardcoded path
flirt_path = "${FSL_DIR}/bin/flirt"
output_type = "NIFTI_GZ"
dof = 6
cost = "corratio"
interp = "nearestneighbour"

# =============================================================================
# DICOM Tags (Reference)
# =============================================================================

[dicom.tags]
instance_number = [0x20, 0x13]
images_in_acquisition = [0x20, 0x1002]
modality = [0x08, 0x60]
patient_id = [0x10, 0x20]
accession_number = [0x08, 0x50]
study_date = [0x08, 0x20]
series_instance_uid = [0x0020, 0x000E]

# =============================================================================
# Logging Configuration
# =============================================================================

[logging]
default_level = "INFO"
scheduler_logger = "add_raw_dicom_to_nii_inference"
pipeline_log_file = "task_pipeline_inference_queue.log"
```

---

## Configuration Loader Design

### Recommended Loader Implementation

```python
# config/loader.py
"""
Configuration Loader - Single Source of Truth

Follows Linus principles:
- Single responsibility
- Pure function design
- Fail-safe with explicit defaults
"""

import os
import tomllib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class HttpConfig:
    default_timeout: int
    long_timeout: int
    subprocess_timeout: int
    async_timeout: int

@dataclass(frozen=True)
class QueueConfig:
    name: str
    qps: int

@dataclass(frozen=True)
class ApplicationConfig:
    http: HttpConfig
    queues: dict[str, QueueConfig]
    tools: dict[str, str]
    # ... other config sections

def load_config(
    config_path: Optional[Path] = None,
    fail_safe: bool = True
) -> ApplicationConfig:
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to TOML file. If None, uses default.
        fail_safe: If True, use defaults on error. If False, raise.

    Returns:
        Frozen ApplicationConfig dataclass
    """
    default_path = Path(__file__).parent / "application.toml"
    path = config_path or default_path

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return _parse_config(data)
    except Exception as e:
        if fail_safe:
            logger.warning(f"Config load failed: {e}. Using defaults.")
            return _default_config()
        raise

def _parse_config(data: dict) -> ApplicationConfig:
    """Parse TOML data into typed config objects."""
    http = HttpConfig(
        default_timeout=data["http"]["default_timeout_seconds"],
        long_timeout=data["http"]["long_timeout_seconds"],
        subprocess_timeout=data["http"]["subprocess_timeout_seconds"],
        async_timeout=data["http"]["async_result_timeout_seconds"],
    )
    # ... parse other sections
    return ApplicationConfig(http=http, ...)

def _default_config() -> ApplicationConfig:
    """Return safe default configuration."""
    return ApplicationConfig(
        http=HttpConfig(
            default_timeout=180,
            long_timeout=300,
            subprocess_timeout=600,
            async_timeout=3600,
        ),
        # ... other defaults
    )
```

---

## Usage Pattern

```python
# In any module
from config.loader import load_config

config = load_config()

# Use typed, centralized configuration
timeout = config.http.default_timeout
queue_name = config.queues["pipeline_inference"].name
tool_id = config.tools["inference"]
```

---

## Migration Strategy

### Phase 1: Create Config Files
1. Create `config/application.toml` with this structure
2. Create `config/labels/*.json` with externalized label mappings
3. Add loader module

### Phase 2: Gradual Replacement
1. Replace magic numbers one module at a time
2. Start with highest-impact duplicates (timeout=180, tool IDs)
3. Add tests to verify config loading

### Phase 3: Cleanup
1. Remove hardcoded values from source
2. Delete duplicated constants
3. Update documentation
