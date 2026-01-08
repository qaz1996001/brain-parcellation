# Proposal: Parameterize UPLOAD_DATA_API_URL

**Change ID**: `parameterize-upload-api-url`
**Status**: Draft
**Created**: 2024-12-24

## Problem Statement

Currently, `UPLOAD_DATA_API_URL` is hardcoded to be read from environment variables within task functions (`process_dir` and `dicom_2_nii_series`). This creates inflexibility:

1. **Environment Coupling**: The API URL is determined by the environment where the worker runs, not where the task is dispatched
2. **Testing Limitations**: Cannot easily test tasks with different API endpoints without changing environment configuration
3. **Deployment Inflexibility**: In dual-deployment scenarios (Production/Testing), workers share the same environment but may need to call different API endpoints based on task origin

### Current Behavior

**File**: `code_ai/task/task_dicom2nii.py`

```python
# Line 253 in dicom_2_nii_series
UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")

# Line 388 in process_dir
UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
```

**Impact**: When Production and Testing environments run in separate folders but share RabbitMQ/Redis for distributed frequency control, tasks cannot distinguish which API endpoint to call.

## Proposed Solution

Convert `UPLOAD_DATA_API_URL` from environment variable to task parameter, allowing callers to explicitly specify the target API endpoint when dispatching tasks.

### Affected Functions

Primary task functions that need parameterization:
1. **`dicom_2_nii_series`** (queue: `dicom_2_nii_series_queue`) - Used for per-series NIfTI conversion
2. **`process_dir`** (queue: `process_dir_queue`) - Used for DICOM renaming and transfer

Secondary functions that call the above:
3. **`dicom_to_nii`** (queue: `dicom_to_nii_queue`) - Wrapper that calls `process_dir`
4. **`raw_dicom_2_rename_dicom`** (queue: `raw_dicom_2_rename_dicom_queue`) - Wrapper that calls `process_dir`

### Callers to Update

Backend services that dispatch these tasks:
- `backend/app/sync/service.py` - `dicom_2_nii_series.push()`, `dicom_to_nii.push()`
- `backend/app/listen/service.py` - `dicom_2_nii_series.push()`, `dicom_to_nii.push()`
- `backend/app/study/service.py` - `dicom_2_nii_series.push()`, `dicom_to_nii.push()`
- `backend/app/rerun/service.py` - (if applicable)
- `code_ai/pipeline/dicom_to_nii.py` - CLI scripts
- `code_ai/pipeline/raw_diom_to_nii_inference.py` - CLI scripts

## Benefits

1. **Environment Decoupling**: API endpoint determined by task dispatcher, not worker environment
2. **Dual Deployment Support**: Production and Testing backends can specify their respective API endpoints
3. **Testing Flexibility**: Easier to test with mock API endpoints without environment changes
4. **Explicit Configuration**: Task parameters make API routing visible and traceable

## Risks and Mitigations

### Risk 1: Breaking Changes for Existing Callers
**Impact**: All existing callers must be updated to pass the new parameter
**Mitigation**:
- Provide backward compatibility by falling back to environment variable if parameter not provided
- Use gradual migration: warn on env-based usage, then deprecate, then remove

### Risk 2: Parameter Validation Complexity
**Impact**: Need to validate URL format and accessibility
**Mitigation**:
- Add Pydantic validators to task parameter schemas
- Centralize URL validation logic

### Risk 3: Security Concerns
**Impact**: Task parameters visible in logs and queue messages
**Mitigation**:
- URL parameters are not sensitive (only internal service URLs)
- Maintain environment variable fallback for sensitive deployments

## Open Questions

1. **Backward Compatibility Strategy**: Should we support environment variable fallback, or require immediate migration?
2. **Validation Scope**: Should we validate URL reachability at task dispatch time or execution time?
3. **Default Values**: Should we provide sensible defaults based on current environment?
4. **Migration Timeline**: Gradual rollout or immediate breaking change?

## Success Criteria

1. ✅ All affected task functions accept `upload_data_api_url` as parameter
2. ✅ All backend service callers updated to pass the parameter
3. ✅ Backward compatibility maintained (optional)
4. ✅ Tests updated to verify parameterization
5. ✅ Documentation updated with migration guide

## Related Changes

- **Dual Deployment Solution**: This change supports `integrate-dual-deployment-gpu-solution` by enabling environment-specific API routing
- **Environment Configuration**: Related to `add-environment-support` in making configuration more flexible

## Next Steps

1. Review and approve proposal
2. Create detailed design.md with implementation approach
3. Define spec deltas for task parameter schemas
4. Create tasks.md with implementation checklist
5. Validate with `openspec validate`
