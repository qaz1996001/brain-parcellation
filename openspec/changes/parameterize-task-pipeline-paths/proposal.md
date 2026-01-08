# Proposal: Parameterize Task Pipeline Environment Paths

**Change ID**: `parameterize-task-pipeline-paths`
**Status**: Draft
**Created**: 2024-12-24

## Problem Statement

Currently, `task_pipeline_inference` and `task_subprocess_inference` read critical path configurations (`PATH_PROCESS`, `PATH_JSON`, `PATH_LOG`) directly from environment variables using `os.getenv()`. This creates inflexibility in dual deployment scenarios:

1. **Environment Coupling**: Path configuration is determined by the worker's environment, not by the task dispatcher
2. **Single Worker Limitation**: Cannot run a single GPU-enabled worker (e.g., `D:\00_Chen\Task04_git`) that serves multiple backend environments (Production: `D:\00_Chen\Task04_git`, Testing: `D:\00_Chen\Task04_git_test`)
3. **Impure Functions**: Task functions depend on external environment state, making them non-deterministic and harder to test
4. **Configuration Inflexibility**: Each backend environment's `.env` configuration cannot be independently applied to shared workers

### Current Behavior

**File**: `code_ai/task/task_pipeline.py`

```python
# Lines 36-39 in task_pipeline_inference
path_process   = os.getenv("PATH_PROCESS")
path_cmd_tools = os.path.join(path_process, 'Deep_cmd_tools')
path_json      = os.getenv("PATH_JSON")
path_log       = os.getenv("PATH_LOG")

# Lines 106-107 in task_subprocess_inference
path_process = os.getenv("PATH_PROCESS")
path_cmd_tools = os.path.join(path_process, 'Deep_cmd_tools')
```

**Impact**: When Production (`D:\00_Chen\Task04_git`) and Testing (`D:\00_Chen\Task04_git_test`) backends share a single GPU worker, they cannot specify different path configurations for task execution.

## Why

### Business Context
- **Resource Optimization**: GPU resources are expensive. Running one shared GPU worker that serves both Production and Testing environments reduces infrastructure costs.
- **Operational Efficiency**: Simplifies GPU resource management by consolidating worker infrastructure while maintaining environment isolation at the backend level.

### Technical Context (Martin Fowler Perspective)

This proposal follows **Dependency Injection** and **Pure Function** principles:

1. **Dependency Injection**: Move environment-specific configuration from implicit (environment variables) to explicit (function parameters). This makes dependencies visible and controllable.

2. **Pure Functions**: Transform task functions from impure (dependent on external state) to pure (all inputs via parameters, deterministic output). This improves:
   - **Testability**: Can test with different path configurations without modifying environment
   - **Composability**: Functions can be composed and reused in different contexts
   - **Predictability**: Same inputs always produce same outputs

3. **Separation of Concerns**: Backend services (dispatchers) own configuration decisions; workers (executors) simply execute with provided parameters.

## Proposed Solution

Convert path configuration from environment variables to task parameters, following the pattern established in `parameterize-upload-api-url`.

### Design Pattern: Parameter Injection with Fallback

```python
# Task function receives paths from func_params
path_process = func_params.get('path_process')
if path_process is None:
    # Fallback to environment variable for backward compatibility
    path_process = os.getenv("PATH_PROCESS")
    if path_process is None:
        raise ValueError("path_process must be provided")
```

### Affected Functions

**Primary Functions** (require parameterization):
1. **`task_pipeline_inference`** - GPU inference pipeline task
   - Environment variables: `PATH_PROCESS`, `PATH_JSON`, `PATH_LOG`, `UPLOAD_DATA_API_URL` (already done)
   - Queue: `task_pipeline_inference_queue`

2. **`task_subprocess_inference`** - Subprocess execution task
   - Environment variables: `PATH_PROCESS`
   - Queue: `task_subprocess_queue`

### Callers to Update

Backend services and CLI scripts that dispatch these tasks:
- `backend/app/sync/service.py` - Task dispatch for sync operations
- `backend/app/listen/service.py` - Task dispatch for listen operations
- `backend/app/study/service.py` - Task dispatch for study operations
- `code_ai/scheduler/scheduler_check_add_task.py` - Scheduled task dispatch

### Configuration Helper

Create centralized path configuration helper (similar to `backend/app/config/api_urls.py`):

```python
# backend/app/config/task_paths.py
def get_task_execution_paths() -> dict:
    """Get path configuration for task parameter injection."""
    return {
        'path_process': os.getenv("PATH_PROCESS"),
        'path_json': os.getenv("PATH_JSON"),
        'path_log': os.getenv("PATH_LOG"),
    }
```

## What Changes

### Core Changes
- **MODIFY**: `task_pipeline_inference` to accept path parameters from `func_params` with environment fallback
- **MODIFY**: `task_subprocess_inference` to accept `path_process` from `func_params` with environment fallback
- **ADD**: Configuration helper `backend/app/config/task_paths.py` for centralized path retrieval
- **MODIFY**: All task dispatchers (backend services, schedulers, CLI scripts) to inject path parameters

### Parameter Additions

**task_pipeline_inference** will accept:
- `path_process`: Base process directory path
- `path_json`: JSON output directory path
- `path_log`: Log file directory path
- `upload_data_api_url`: Already implemented in previous change

**task_subprocess_inference** will accept:
- `path_process`: Base process directory path

## Benefits

1. **Pure Function Design**: Task functions become deterministic, receiving all configuration as parameters
2. **Environment Decoupling**: Path configuration determined by task dispatcher (backend), not worker environment
3. **Dual Deployment Support**: Production and Testing backends can specify their respective paths when dispatching to shared GPU worker
4. **Testing Flexibility**: Easier to test with different path configurations without environment changes
5. **Explicit Configuration**: Task parameters make path routing visible and traceable in task queues
6. **Resource Optimization**: Single GPU worker can serve multiple environments with different configurations

## Impact

### Affected Specs
- **task-execution**: Core task execution patterns and path management

### Affected Code
- `code_ai/task/task_pipeline.py` - Task function implementation
- `backend/app/sync/service.py` - Task dispatch logic
- `backend/app/listen/service.py` - Task dispatch logic
- `backend/app/study/service.py` - Task dispatch logic
- `code_ai/scheduler/scheduler_check_add_task.py` - Scheduler task dispatch
- `backend/app/config/task_paths.py` - NEW configuration helper

## Risks and Mitigations

### Risk 1: Breaking Changes for Existing Callers
**Impact**: All existing callers must be updated to pass new parameters
**Mitigation**:
- Provide backward compatibility by falling back to environment variables if parameters not provided
- Use gradual migration: warn on env-based usage, then deprecate, then remove
- Update all callers in same change to maintain consistency

### Risk 2: Parameter Validation Complexity
**Impact**: Need to validate path existence and accessibility
**Mitigation**:
- Add validation in configuration helper
- Centralize path validation logic
- Fail fast with clear error messages

### Risk 3: Configuration Consistency
**Impact**: Multiple backends may specify conflicting path configurations
**Mitigation**:
- Document path configuration expectations
- Add validation that paths are absolute and accessible
- Provide clear error messages for misconfiguration

## Open Questions

1. **Backward Compatibility Strategy**: Should we support environment variable fallback permanently, or plan deprecation timeline?
   - **Recommendation**: Keep fallback indefinitely for CLI scripts, deprecate for backend services after migration

2. **Validation Scope**: Should we validate path accessibility at dispatch time or execution time?
   - **Recommendation**: Validate at dispatch time for fast failure, with additional validation at execution time

3. **Path Resolution**: Should we support relative paths or require absolute paths?
   - **Recommendation**: Require absolute paths for clarity and consistency

4. **CLI Script Migration**: Should CLI scripts continue using environment variables or switch to explicit parameters?
   - **Recommendation**: CLI scripts can continue using environment fallback for developer convenience

## Success Criteria

1. ✅ `task_pipeline_inference` accepts `path_process`, `path_json`, `path_log` as parameters
2. ✅ `task_subprocess_inference` accepts `path_process` as parameter
3. ✅ All backend service callers updated to pass path parameters
4. ✅ Configuration helper created with validation
5. ✅ Backward compatibility maintained with environment variable fallback
6. ✅ Tests updated to verify parameterization
7. ✅ Documentation updated with migration guide
8. ✅ Dual deployment scenario validated (Production + Testing → Single GPU worker)

## Related Changes

- **`parameterize-upload-api-url`**: This change follows the same pattern for URL parameterization
- **`integrate-dual-deployment-gpu-solution`**: This change enables dual deployment GPU sharing
- **`add-environment-support`**: Related to environment-specific configuration

## Next Steps

1. Review and approve proposal
2. Create detailed design.md with implementation approach
3. Define spec deltas for task execution patterns
4. Create tasks.md with implementation checklist
5. Validate with `openspec validate parameterize-task-pipeline-paths --strict`
