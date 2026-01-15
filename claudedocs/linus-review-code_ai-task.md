# Code_AI Task Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `code_ai/task/`
- **Files Reviewed**: `task_pipeline.py`, `task_dicom2nii.py`, `params.py`, `schema/intput_params.py`
- **Total Lines**: ~3300+

---

## Critical Issues

### 1. CRITICAL: God Function - 594 Lines
**File**: `code_ai/task/task_pipeline.py`
**Function**: `_task_series_pipeline_inference`

This is the most severe violation:
- **Lines**: 594
- **Nesting**: 5+ levels
- **Responsibilities**: Validation, path construction, conversion, inference, error handling, event posting

**Must decompose into**:
- `_validate_series_params()`
- `_prepare_nifti_paths()`
- `_execute_batch_inference()`
- `_execute_single_inference()`
- `_post_completion_event()`

### 2. CRITICAL: Hardcoded UUID-to-Model Mapping
**File**: `code_ai/task/task_pipeline.py`
**Lines**: 620-625

```python
# This is configuration data hardcoded as code
LEGACY_UUID_MAPPING = {
    "48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6": "CMB",
    "924d1538-597c-41d6-bc27-4b0b359111cf": "Aneurysm",
    "7e94d381-3f5d-46b6-b440-e5d44ebc48d2": "WMH",
    "97abe75d-34de-4e91-80c2-ce74b6c70438": "Infarct",
}
```

**Linus quote**: "Bad programmers worry about the code. Good programmers worry about data structures."

This data should be in TOML configuration.

### 3. Filename Typo
**File**: `code_ai/task/schema/intput_params.py`

Should be `input_params.py` (missing 'n' in "input")

---

## Function Length Violations

| Function | Lines | File |
|----------|-------|------|
| `_task_series_pipeline_inference` | 594 | task_pipeline.py |
| `_build_series_inference_cmd` | 187 | task_pipeline.py |
| `_reorder_series_by_config` | 147 | task_pipeline.py |
| `_convert_single_series_to_nifti` | 123 | task_pipeline.py |
| `_task_study_pipeline_inference` | 108 | task_pipeline.py |
| `_batch_convert_series_to_nifti` | 98 | task_pipeline.py |
| `process_dir` | 89 | task_dicom2nii.py |
| `dicom_2_nii_series` | 65 | task_dicom2nii.py |
| `_execute_dcm2niix` | 62 | task_dicom2nii.py |

---

## Magic Numbers Inventory

| File | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| task_pipeline.py | `qps=1` | GPU mutex | `task.pipeline.qps` |
| task_pipeline.py | `600` | Subprocess timeout | `task.timeouts.subprocess` |
| task_pipeline.py | `36` | UUID length check | `validation.uuid_length` |
| task_dicom2nii.py | `qps=5,10,100` | Queue rates | `task.qps.*` |
| task_dicom2nii.py | `300` | HTTP timeout | `task.timeouts.http` |
| task_dicom2nii.py | `500` | Min file size | `task.files.min_size` |
| task_dicom2nii.py | `3600` | Async timeout | `task.timeouts.async` |
| params.py | `10` | Concurrent threads | `rabbitmq.concurrent` |
| params.py | `3` | Max retries | `rabbitmq.max_retries` |
| params.py | `20` | Retry interval | `rabbitmq.retry_interval` |
| params.py | `1800` | RPC expire | `rabbitmq.rpc_expire` |

---

## Magic Strings Inventory

| File | String | Purpose | TOML Key |
|------|--------|---------|----------|
| task_pipeline.py | `"task_pipeline_inference_queue"` | Queue name | `queues.pipeline` |
| task_pipeline.py | `"INFERENCE_TOOL"` | Tool ID | `tools.inference` |
| task_pipeline.py | `"SERIES_INFERENCE_TOOL"` | Tool ID | `tools.series_inference` |
| task_pipeline.py | `"Deep_cmd_tools"` | Subdir name | `paths.cmd_tools_dir` |
| task_pipeline.py | `"prediction.json"` | Output file | `files.prediction` |
| task_dicom2nii.py | `"NIFTI_TOOL"` | Tool ID | `tools.nifti` |
| task_dicom2nii.py | `"DICOM_TOOL"` | Tool ID | `tools.dicom` |
| task_dicom2nii.py | `".nii.gz"` | Extension | `files.nifti_ext` |
| task_dicom2nii.py | `".meta"` | Folder name | `files.meta_folder` |

---

## Duplicate `FILE_SIZE = 500`
Defined at:
- `task_dicom2nii.py` line 282
- `task_dicom2nii.py` line 340

Should be single constant or in TOML.

---

## Suggested TOML Configuration

```toml
[task]
# Queue Processing Rates (qps)
[task.qps]
pipeline_inference = 1  # GPU mutual exclusion
post_httpx = 5
dcm2niix = 10
dicom_2_nii = 10
process_instances = 100

[task.timeouts]
subprocess_seconds = 600
http_seconds = 300
async_seconds = 3600
dcm2niix_seconds = 300

[task.files]
min_nifti_size_bytes = 500
meta_folder = ".meta"
nifti_extension = ".nii.gz"
prediction_filename = "prediction.json"

[task.tools]
inference = "INFERENCE_TOOL"
series_inference = "SERIES_INFERENCE_TOOL"
nifti = "NIFTI_TOOL"
dicom = "DICOM_TOOL"

[task.queues]
pipeline_inference = "task_pipeline_inference_queue"
dcm2niix = "call_dcm2niix_queue"
dicom_2_nii_file = "dicom_2_nii_file_queue"
dicom_2_nii_series = "dicom_2_nii_series_queue"
process_dir = "process_dir_queue"

[task.rabbitmq]
concurrent_num = 10
ai_concurrent_num = 1  # GPU constraint
max_retry_times = 3
retry_interval_seconds = 20
rpc_result_expire_seconds = 1800

[task.models]
# Infarct model requirements
infarct_target_series = ["ADC", "DWI0", "DWI1000"]

# Legacy UUID mapping (move from code to config!)
[task.models.legacy_uuid]
"48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6" = "CMB"
"924d1538-597c-41d6-bc27-4b0b359111cf" = "Aneurysm"
"7e94d381-3f5d-46b6-b440-e5d44ebc48d2" = "WMH"
"97abe75d-34de-4e91-80c2-ce74b6c70438" = "Infarct"

[task.validation]
uuid_length = 36
uuid_hyphen_count = 4
```

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | FAIL (severe) |
| Nesting < 3 levels | FAIL |
| No magic numbers | FAIL |
| No hardcoded UUIDs | FAIL |
| DRY principle | FAIL |
| Correct file naming | FAIL |
