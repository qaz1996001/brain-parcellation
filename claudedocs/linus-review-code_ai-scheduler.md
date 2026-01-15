# Code_AI Scheduler Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `code_ai/scheduler/`
- **Files Reviewed**: `scheduler_check_add_task.py`, `scheduler_database.py`, `__init__.py`
- **Total Lines**: ~960

---

## Critical Issues

### 1. Dead Code: 185 Lines of Comments
**File**: `code_ai/scheduler/scheduler_check_add_task.py`
**Lines**: 646-831

A massive block of commented-out code that should be deleted.

```python
# @Booster(BoosterParams(
#     queue_name='add_raw_dicom_to_nii_inference_queue',
#     ...
# ))
# def add_raw_dicom_to_nii_inference(tasks_file_path: Optional[str] = None) -> Dict[str, Any]:
#     """..."""
#     # 185 lines of commented code
```

### 2. Placeholder Function (Does Nothing)
**File**: `code_ai/scheduler/scheduler_database.py`

```python
@Booster(...)
def delete_old_date():  # Note: "date" should be "data"
    pass  # Empty function!
```

### 3. Typo: "date" vs "data"
**File**: `scheduler_database.py`

- Queue name: `'delete_old_date_queue'` → should be `'delete_old_data_queue'`
- Function name: `delete_old_date` → should be `delete_old_data`

### 4. Variable Typo
**File**: `scheduler_check_add_task.py`

```python
clinet = httpx.Client(timeout=300)  # Should be "client"
```

---

## Function Length Violations

| Function | Lines | Status |
|----------|-------|--------|
| `add_raw_dicom_to_nii_inference` | 95 | FAIL |
| `process_pipeline_tasks` | 86 | FAIL |
| `check_dicom_completeness` | 82 | FAIL |
| `is_task_waiting_in_queue` | 90 | FAIL |
| `check_output_files_and_rerun` | 96 | FAIL |

---

## Magic Numbers Inventory

| File | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| scheduler_check_add_task.py | `300` | HTTP timeout | `scheduler.http_timeout` |
| scheduler_check_add_task.py | `100` | DICOM sample limit | `scheduler.dicom_sample_limit` |
| scheduler_check_add_task.py | `0.9` | Completeness threshold | `scheduler.completeness_threshold` |
| scheduler_check_add_task.py | `300` | Recent file threshold | `scheduler.recent_file_seconds` |
| scheduler_check_add_task.py | `qps=1` | Queue rate | `scheduler.qps` |
| scheduler_check_add_task.py | `log_level=20` | INFO level | `scheduler.log_level` |

---

## Magic Strings Inventory

| File | String | Purpose | TOML Key |
|------|--------|---------|----------|
| scheduler_check_add_task.py | `"PATH_RAW_DICOM"` | Env var | N/A |
| scheduler_check_add_task.py | `"PATH_RENAME_DICOM"` | Env var | N/A |
| scheduler_check_add_task.py | `"PATH_RENAME_NIFTI"` | Env var | N/A |
| scheduler_check_add_task.py | `"UPLOAD_DATA_API_URL"` | Env var | N/A |
| scheduler_check_add_task.py | `"task_pipeline_inference_queue"` | Queue (4x) | `queues.pipeline` |
| scheduler_check_add_task.py | `"dicom_to_nii_queue"` | Queue | `queues.dicom_to_nii` |
| scheduler_check_add_task.py | `"add_raw_dicom_to_nii_inference_queue"` | Queue | `queues.scheduler` |
| scheduler_check_add_task.py | `"%Y-%m-%d %H:%M:%S"` | Date format | `scheduler.datetime_format` |

---

## DRY Violations

1. `"task_pipeline_inference_queue"` appears 4 times
2. HTTP timeout `300` appears 2 times
3. DICOM tag names repeated in validation

---

## Unused Import
**File**: `scheduler_database.py`

```python
from nb_log.monkey_print import LOCAL_DB  # Imported but never used
```

---

## Print Statement Instead of Logging
**File**: `scheduler_check_add_task.py`

```python
print(rep)  # Line ~91 - Should use logger
print(study_dir_name_list)  # Line ~909 - Should use logger
```

---

## Suggested TOML Configuration

```toml
[scheduler]
datetime_format = "%Y-%m-%d %H:%M:%S"
log_level = "INFO"  # Instead of magic number 20

[scheduler.timeouts]
http_seconds = 300
task_execution_seconds = 600
recent_file_threshold_seconds = 300

[scheduler.dicom_validation]
completeness_threshold = 0.9
max_sample_files = 100
required_tags = ["StudyInstanceUID", "SeriesInstanceUID"]

[scheduler.queues]
scheduler = "add_raw_dicom_to_nii_inference_queue"
dicom_to_nii = "dicom_to_nii_queue"
pipeline_inference = "task_pipeline_inference_queue"
delete_old_data = "delete_old_data_queue"  # Fixed typo

[scheduler.queues.config]
qps = 1
concurrent_mode = "SOLO"
broker_kind = "RABBITMQ_AMQPSTORM"
```

---

## Files to Fix

1. **Delete**: 185 lines of dead commented code (lines 646-831)
2. **Fix typo**: `delete_old_date` → `delete_old_data`
3. **Fix typo**: `clinet` → `client`
4. **Remove**: Unused import `LOCAL_DB`
5. **Implement or remove**: Empty `delete_old_date()` function

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | FAIL |
| Nesting < 3 levels | FAIL |
| No magic numbers | FAIL |
| No dead code | FAIL |
| No typos | FAIL |
| DRY principle | FAIL |
