# Migration Guide: Task Path Parameterization

**Change ID**: `parameterize-task-pipeline-paths`
**Version**: 1.0
**Date**: 2024-12-24

## Overview

This guide helps you migrate from environment-based path configuration to parameter-based path injection for GPU inference tasks. The migration enables dual deployment scenarios where multiple backend environments (Production, Testing) can share a single GPU worker with different path configurations.

## What Changed

### Before (Environment-Based)
```python
# Task function reads from its own environment
def task_pipeline_inference(func_params: Dict[str, any]):
    path_process = os.getenv("PATH_PROCESS")  # Worker's .env
    path_json = os.getenv("PATH_JSON")
    path_log = os.getenv("PATH_LOG")
```

### After (Parameter-Based)
```python
# Task function receives paths from dispatcher
def task_pipeline_inference(func_params: Dict[str, any]):
    path_process = _extract_path_from_params(func_params, 'path_process', 'PATH_PROCESS')
    path_json = _extract_path_from_params(func_params, 'path_json', 'PATH_JSON')
    path_log = _extract_path_from_params(func_params, 'path_log', 'PATH_LOG')
```

## Migration Steps

### Step 1: Update Configuration Helper (Completed)

The centralized configuration helper is already in place:

```python
# backend/app/config/task_paths.py
from backend.app.config.task_paths import get_task_execution_paths

# Get paths from environment (or override for testing)
paths = get_task_execution_paths()
# Returns: {'path_process': '...', 'path_json': '...', 'path_log': '...'}
```

### Step 2: Update Task Dispatchers (Completed)

All backend dispatchers now inject path parameters:

**Sync Service** (`backend/app/sync/service.py`):
```python
from backend.app.config.task_paths import get_task_execution_paths

task_paths = get_task_execution_paths()
params_data = {
    'nifti_study_path': str(nifti_study_path),
    'dicom_study_path': str(dicom_study_path),
    'path_process': task_paths['path_process'],  # NEW
    'path_json': task_paths['path_json'],        # NEW
    'path_log': task_paths['path_log']           # NEW
}
```

**Listen Service** (`backend/app/listen/service.py`) - Same pattern
**Study Service** (`backend/app/study/service.py`) - Same pattern
**Scheduler** (`code_ai/scheduler/scheduler_check_add_task.py`) - Same pattern

### Step 3: Verify Task Functions (Completed)

Task functions now extract paths from parameters first, then fall back to environment:

```python
# code_ai/task/task_pipeline.py
def _extract_path_from_params(func_params: Dict[str, any],
                               param_name: str,
                               env_var_name: str) -> str:
    """Extract path with fallback to environment."""
    path_value = func_params.get(param_name)
    if path_value is None:
        logger.warning(f"Falling back to environment variable {env_var_name}")
        path_value = os.getenv(env_var_name)
        if path_value is None:
            raise ValueError(f"{param_name} must be provided")
    return path_value
```

## Environment Variable Mapping

| Environment Variable | Parameter Name | Description |
|---------------------|----------------|-------------|
| `PATH_PROCESS` | `path_process` | Base process directory for inference |
| `PATH_JSON` | `path_json` | JSON output directory |
| `PATH_LOG` | `path_log` | Log file directory |

## Dual Deployment Configuration

### Production Backend (.env)
```bash
# D:\00_Chen\Task04_git\.env
PATH_PROCESS=D:/00_Chen/Task04_git/process
PATH_JSON=D:/00_Chen/Task04_git/json
PATH_LOG=D:/00_Chen/Task04_git/logs
```

### Testing Backend (.env)
```bash
# D:\00_Chen\Task04_git_test\.env
PATH_PROCESS=D:/00_Chen/Task04_git_test/process
PATH_JSON=D:/00_Chen/Task04_git_test/json
PATH_LOG=D:/00_Chen/Task04_git_test/logs
```

### GPU Worker
The worker no longer needs PATH_* variables in its .env file. It receives paths as task parameters from whichever backend dispatches the task.

## Troubleshooting

### Issue: "path_process must be provided" Error

**Cause**: Neither parameter nor environment variable is set.

**Solution**:
1. Ensure dispatcher is calling `get_task_execution_paths()`
2. Verify environment variables are set in backend's .env file
3. Check that paths are absolute (not relative)

```python
# Check if paths are configured
from backend.app.config.task_paths import get_task_execution_paths
try:
    paths = get_task_execution_paths()
    print(f"Paths OK: {paths}")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Issue: Warning Logs "Falling back to environment variable"

**Cause**: Dispatcher is not passing path parameters (legacy behavior).

**Solution**: This is expected during migration. The fallback ensures backward compatibility. To eliminate warnings:
1. Verify dispatcher imports `get_task_execution_paths`
2. Check that `task_paths` are added to `params_data` dict
3. Confirm RabbitMQ queue is consuming updated dispatchers

### Issue: Tasks Execute in Wrong Path

**Cause**: Worker is using old environment-based configuration.

**Solution**:
1. Restart GPU worker to pick up new parameter extraction logic
2. Verify `code_ai/task/task_pipeline.py` has been updated
3. Check task parameters in RabbitMQ queue contain path fields

```python
# Inspect task parameters in RabbitMQ
# Should see: {'path_process': '...', 'path_json': '...', 'path_log': '...'}
```

### Issue: Permission Denied on Path Creation

**Cause**: Path validation checks write permissions at dispatch time.

**Solution**:
1. Ensure backend process has permissions for configured paths
2. Create directories manually if needed: `mkdir -p /path/to/process`
3. Check directory ownership: `ls -la /path/to/parent`

### Issue: Relative Path Rejected

**Cause**: Configuration helper requires absolute paths.

**Error**: `ValueError: path_process must be an absolute path, got: relative/path`

**Solution**: Use absolute paths in .env configuration:
```bash
# Wrong
PATH_PROCESS=process

# Correct
PATH_PROCESS=D:/00_Chen/Task04_git/process
```

## Rollback Instructions

If you need to rollback to environment-based configuration:

### Option 1: Temporary Rollback (Use Fallback)
Simply don't pass path parameters. The fallback mechanism will use environment variables:

```python
# Dispatcher: Don't call get_task_execution_paths()
params_data = {
    'nifti_study_path': str(nifti_study_path),
    'dicom_study_path': str(dicom_study_path),
    # path_process, path_json, path_log NOT included
}
```

Task functions will log warnings but continue working with environment variables.

### Option 2: Full Rollback (Revert Code)

1. **Revert Task Functions** (`code_ai/task/task_pipeline.py`):
```python
# Remove _extract_path_from_params utility
# Restore direct os.getenv() calls
path_process = os.getenv("PATH_PROCESS")
path_json = os.getenv("PATH_JSON")
path_log = os.getenv("PATH_LOG")
```

2. **Revert Dispatchers**: Remove imports and parameter injection
3. **Remove Helper**: Delete `backend/app/config/task_paths.py`
4. **Restart Services**: Restart all backend services and GPU worker

## Validation Checklist

After migration, verify:

- [ ] Production backend dispatches tasks with Production paths
- [ ] Testing backend dispatches tasks with Testing paths
- [ ] GPU worker executes tasks in correct path context
- [ ] No "must be provided" errors in logs
- [ ] Task output appears in expected directories
- [ ] Dual deployment works (both backends → one worker)

## Testing Override Mechanism

For testing or dynamic routing, use the override parameter:

```python
from backend.app.config.task_paths import get_task_execution_paths

# Test with custom paths
test_paths = {
    'path_process': '/tmp/test/process',
    'path_json': '/tmp/test/json',
    'path_log': '/tmp/test/logs'
}
paths = get_task_execution_paths(override=test_paths)

# Use in task dispatch
params_data.update(paths)
```

## Performance Impact

The migration has minimal performance impact:

- **Dispatch Time**: +1-2ms for `get_task_execution_paths()` call
- **Execution Time**: No change (path extraction is negligible)
- **Queue Size**: +~50 bytes per task (3 additional string parameters)

## Support

For issues or questions:
1. Check logs: `logs/task_pipeline_inference_queue.log`
2. Verify configuration: `get_task_execution_paths()`
3. Review OpenSpec proposal: `openspec/changes/parameterize-task-pipeline-paths/`
4. Contact DevOps team for deployment assistance

## Related Documentation

- [Architecture Decision Record](./adr/ADR-TASK-PATH-PARAMETERIZATION.md)
- [Dual Deployment Guide](./DUAL_FOLDER_DEPLOYMENT_GUIDE.md)
- [API Reference](./API_REFERENCE.md)
