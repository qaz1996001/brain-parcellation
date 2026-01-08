# API Reference: Task Pipeline Path Parameterization

## Configuration Helper API

### `get_task_execution_paths()`

**Module**: `backend.app.config.task_paths`

Retrieves task execution path configuration for parameter injection into GPU inference tasks.

#### Signature
```python
def get_task_execution_paths(override: Optional[Dict[str, str]] = None) -> Dict[str, str]
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `override` | `Dict[str, str]` | No | Optional path overrides. If provided, these paths are used instead of reading from environment. Useful for testing or dynamic routing. |

#### Returns

`Dict[str, str]` with keys:
- `path_process`: Base process directory path (from `PATH_PROCESS` env var)
- `path_json`: JSON output directory path (from `PATH_JSON` env var)
- `path_log`: Log file directory path (from `PATH_LOG` env var)

#### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If required path is not configured (neither override nor environment variable set) |
| `ValueError` | If path is not an absolute path |

#### Examples

**Standard Usage (Production)**:
```python
from backend.app.config.task_paths import get_task_execution_paths

# Read from environment variables
paths = get_task_execution_paths()
# Returns: {
#     'path_process': 'D:/00_Chen/Task04_git/process',
#     'path_json': 'D:/00_Chen/Task04_git/json',
#     'path_log': 'D:/00_Chen/Task04_git/logs'
# }

# Use in task dispatch
task_dict['path_process'] = paths['path_process']
task_dict['path_json'] = paths['path_json']
task_dict['path_log'] = paths['path_log']
task_pipeline_inference.push(task_dict)
```

**Testing with Override**:
```python
# Testing environment with custom paths
test_paths = {
    'path_process': 'D:/00_Chen/Task04_git_test/process',
    'path_json': 'D:/00_Chen/Task04_git_test/json',
    'path_log': 'D:/00_Chen/Task04_git_test/logs'
}
paths = get_task_execution_paths(override=test_paths)
task_dict.update(paths)
task_pipeline_inference.push(task_dict)
```

**Partial Override**:
```python
# Override only path_process, use environment for others
partial_override = {'path_process': '/custom/process/path'}
paths = get_task_execution_paths(override=partial_override)
# path_process from override, path_json/path_log from environment
```

#### Path Validation

The function performs the following validations:

1. **Existence Check**: All three paths must be configured (either via override or environment)
2. **Absolute Path Check**: All paths must be absolute (e.g., `D:/...` or `/...`), relative paths are rejected
3. **Clear Error Messages**: Provides specific error messages indicating which path is missing or invalid

**Note**: The function does NOT validate:
- Path existence (paths may not exist yet and will be created by tasks)
- Write permissions (checked at execution time, not dispatch time)
- Disk space availability

---

## Task Function APIs

### `task_pipeline_inference()`

**Module**: `code_ai.task.task_pipeline`

GPU inference pipeline task that processes DICOM and NIfTI data.

#### Signature
```python
@Booster(BoosterParamsMyAI(queue_name='task_pipeline_inference_queue', ...))
def task_pipeline_inference(func_params: Dict[str, any])
```

#### Parameters

`func_params` dictionary must contain:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `nifti_study_path` | `str` | Yes | Path to NIfTI study directory |
| `dicom_study_path` | `str` | Yes | Path to DICOM study directory |
| `upload_data_api_url` | `str` | Yes* | Base URL for upload data API |
| `path_process` | `str` | Yes* | Base process directory path |
| `path_json` | `str` | Yes* | JSON output directory path |
| `path_log` | `str` | Yes* | Log file directory path |
| `study_uid` | `str` | No | Study UID for tracking |
| `study_id` | `str` | No | Study ID for tracking |

*Required in parameters OR environment variable must be set (fallback mechanism)

#### Path Parameter Behavior

```python
# Priority: Parameter > Environment Variable > Error

# 1. If path_process in func_params: Use it
# 2. Else if PATH_PROCESS env var set: Use it (logs warning)
# 3. Else: Raise ValueError
```

#### Example Usage

**Backend Dispatcher**:
```python
from backend.app.config.task_paths import get_task_execution_paths
from backend.app.config.api_urls import get_upload_data_api_url
from code_ai.task.task_pipeline import task_pipeline_inference

# Get configuration
base_api_url = get_upload_data_api_url()
task_paths = get_task_execution_paths()

# Prepare task parameters
params_data = {
    'nifti_study_path': '/data/nifti/study_001',
    'dicom_study_path': '/data/dicom/study_001',
    'study_uid': '1.2.3.4.5',
    'study_id': 'STU001',
    'upload_data_api_url': base_api_url,
    'path_process': task_paths['path_process'],
    'path_json': task_paths['path_json'],
    'path_log': task_paths['path_log']
}

# Dispatch task to GPU worker
result = task_pipeline_inference.push(params_data)
```

**CLI Script** (Environment Fallback):
```python
# CLI can rely on environment variables
params_data = {
    'nifti_study_path': '/data/nifti/study_001',
    'dicom_study_path': '/data/dicom/study_001',
    # path_process, path_json, path_log will use environment fallback
}
task_pipeline_inference.push(params_data)
```

#### Returns

`str`: JSON-serialized list of inference results

#### Side Effects

- Creates directories: `path_json`, `path_log`, `path_process/Deep_cmd_tools`
- Writes command JSON: `path_process/Deep_cmd_tools/{study_id}_cmd.json`
- Executes subprocess inference commands
- Sends HTTP callbacks to `upload_data_api_url` with status updates

---

### `task_subprocess_inference()`

**Module**: `code_ai.task.task_pipeline`

Subprocess execution task for running shell commands.

#### Signature
```python
@Booster(BoosterParamsMyRABBITMQ(queue_name='task_subprocess_queue', ...))
def task_subprocess_inference(func_params: Dict[str, any])
```

#### Parameters

`func_params` dictionary must contain:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `cmd_str` | `str` | Yes | Shell command to execute |
| `path_process` | `str` | Yes* | Base process directory path |

*Required in parameters OR environment variable `PATH_PROCESS` must be set (fallback mechanism)

#### Example Usage

```python
from code_ai.task.task_pipeline import task_subprocess_inference
from backend.app.config.task_paths import get_task_execution_paths

# Get paths
task_paths = get_task_execution_paths()

# Prepare subprocess task
params_data = {
    'cmd_str': 'python /path/to/script.py --input data.nii',
    'path_process': task_paths['path_process']
}

# Dispatch task
result = task_subprocess_inference.push(params_data)
```

#### Returns

`str`: Decoded stdout from subprocess execution

#### Side Effects

- Creates directory: `path_process/Deep_cmd_tools`
- Executes subprocess with `shell=True`
- Logs stdout (info level) and stderr (warning level)

---

## Internal Utility APIs

### `_extract_path_from_params()`

**Module**: `code_ai.task.task_pipeline`

**Visibility**: Internal (prefix `_` indicates private)

Utility function for extracting path parameters with environment fallback.

#### Signature
```python
def _extract_path_from_params(func_params: Dict[str, any],
                               param_name: str,
                               env_var_name: str) -> str
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `func_params` | `Dict[str, any]` | Task parameters from dispatcher |
| `param_name` | `str` | Parameter key to extract (e.g., `'path_process'`) |
| `env_var_name` | `str` | Environment variable name for fallback (e.g., `'PATH_PROCESS'`) |

#### Returns

`str`: The path value from parameters or environment

#### Raises

`ValueError`: If path not found in parameters or environment

#### Behavior

1. **Check Parameters**: `func_params.get(param_name)`
2. **Fallback to Environment**: If not found, `os.getenv(env_var_name)` with warning log
3. **Error**: If both are None, raise `ValueError`

#### Example

```python
# Internal usage in task functions
path_process = _extract_path_from_params(
    func_params=func_params,
    param_name='path_process',
    env_var_name='PATH_PROCESS'
)
```

---

## Migration Patterns

### Pattern 1: Backend Service Task Dispatch

```python
from backend.app.config.task_paths import get_task_execution_paths
from backend.app.config.api_urls import get_upload_data_api_url

class MyService:
    async def dispatch_inference_task(self, study_data):
        # Get configuration
        base_api_url = get_upload_data_api_url()
        task_paths = get_task_execution_paths()

        # Build params_data
        params_data = {
            'nifti_study_path': str(study_data.nifti_path),
            'dicom_study_path': str(study_data.dicom_path),
            'study_uid': study_data.uid,
            'study_id': study_data.id,
            'upload_data_api_url': base_api_url,
            **task_paths  # Unpack: path_process, path_json, path_log
        }

        # Dispatch
        task_pipeline_inference.push(params_data)
```

### Pattern 2: CLI Script with Environment Fallback

```python
from code_ai.task.task_pipeline import task_pipeline_inference

# Simple CLI dispatch - uses environment variables
params_data = {
    'nifti_study_path': args.nifti_path,
    'dicom_study_path': args.dicom_path,
}

task_pipeline_inference.push(params_data)
# Will log warnings about environment fallback
```

### Pattern 3: Testing with Override

```python
import pytest
from backend.app.config.task_paths import get_task_execution_paths

@pytest.fixture
def test_paths(tmp_path):
    """Provide temporary test paths."""
    return {
        'path_process': str(tmp_path / 'process'),
        'path_json': str(tmp_path / 'json'),
        'path_log': str(tmp_path / 'logs')
    }

def test_inference_task(test_paths):
    paths = get_task_execution_paths(override=test_paths)

    params_data = {
        'nifti_study_path': '/test/nifti',
        'dicom_study_path': '/test/dicom',
        **paths
    }

    result = task_pipeline_inference.push(params_data)
    # Task executes in temporary test paths
```

---

## Environment Variables

| Variable | Purpose | Example | Required |
|----------|---------|---------|----------|
| `PATH_PROCESS` | Base process directory | `D:/00_Chen/Task04_git/process` | Yes* |
| `PATH_JSON` | JSON output directory | `D:/00_Chen/Task04_git/json` | Yes* |
| `PATH_LOG` | Log file directory | `D:/00_Chen/Task04_git/logs` | Yes* |
| `UPLOAD_DATA_API_URL` | Upload API base URL | `http://backend:8000` | Yes* |

*Required if not provided as task parameters

---

## Error Handling

### Common Errors

**ValueError: "path_process must be configured"**
```python
# Cause: Neither parameter nor environment variable set
# Solution: Set environment variable or pass in parameters

# Option 1: Environment
export PATH_PROCESS=/path/to/process

# Option 2: Parameter
params_data['path_process'] = '/path/to/process'
```

**ValueError: "path_process must be an absolute path"**
```python
# Cause: Relative path provided
# Wrong: 'process/data'
# Right: 'D:/00_Chen/Task04_git/process'

paths = get_task_execution_paths(override={
    'path_process': 'D:/00_Chen/Task04_git/process',  # Absolute
    # Not: 'process'  # Relative - will fail
})
```

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| `get_task_execution_paths()` | <1ms | Simple environment variable reads |
| `_extract_path_from_params()` | <0.1ms | Dict lookup + optional env read |
| Parameter serialization overhead | ~50 bytes | 3 additional string fields in task queue |

---

## Related Documentation

- [Migration Guide](./MIGRATION_TASK_PATHS.md)
- [Architecture Decision Record](./adr/ADR-TASK-PATH-PARAMETERIZATION.md)
- [Dual Deployment Guide](./DUAL_FOLDER_DEPLOYMENT_GUIDE.md)
