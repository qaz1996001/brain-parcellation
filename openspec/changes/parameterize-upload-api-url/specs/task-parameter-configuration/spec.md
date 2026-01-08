# Spec: Task Parameter Configuration

**Capability**: `task-parameter-configuration`
**Version**: 1.0.0
**Status**: Draft

## Overview

This specification defines how task functions accept runtime configuration parameters (specifically API endpoints) rather than relying solely on environment variables, enabling flexible task routing in multi-environment deployments.

## ADDED Requirements

### Requirement: Task Parameter Schema Extension

Task parameter schemas MUST support optional `upload_data_api_url` field with validation and environment variable fallback.

#### Scenario: Explicit API URL provided

**Given** a task dispatcher has a specific API endpoint to use
**When** the dispatcher creates task parameters with `upload_data_api_url="http://api.example.com"`
**Then** the task parameter schema accepts the URL
**And** validates it as a proper HTTP/HTTPS URL
**And** stores it for use by the task function

**Example**:
```python
params = Dicom2NiiSeriesParams(
    study_uid="study-123",
    series_uid="series-456",
    upload_data_api_url="http://production-api.example.com"
)
assert params.upload_data_api_url == "http://production-api.example.com"
```

#### Scenario: Environment variable fallback

**Given** a task dispatcher does not provide `upload_data_api_url` parameter
**And** the environment variable `UPLOAD_DATA_API_URL` is set to "http://env-api.example.com"
**When** the dispatcher creates task parameters without `upload_data_api_url`
**Then** the schema automatically populates `upload_data_api_url` from the environment variable
**And** validates the environment-provided URL

**Example**:
```python
# Environment: UPLOAD_DATA_API_URL=http://env-api.example.com
params = Dicom2NiiSeriesParams(
    study_uid="study-123",
    series_uid="series-456"
    # upload_data_api_url not provided
)
assert params.upload_data_api_url == "http://env-api.example.com"
```

#### Scenario: Invalid URL format rejection

**Given** a task dispatcher provides an invalid URL format
**When** the dispatcher attempts to create task parameters with `upload_data_api_url="not-a-url"`
**Then** the schema raises a `ValueError` with message containing "must be valid HTTP URL"
**And** the task parameters are not created

**Example**:
```python
with pytest.raises(ValueError, match="must be valid HTTP URL"):
    Dicom2NiiSeriesParams(
        study_uid="study-123",
        series_uid="series-456",
        upload_data_api_url="ftp://invalid-protocol.com"
    )
```

#### Scenario: Missing URL configuration error

**Given** a task dispatcher does not provide `upload_data_api_url` parameter
**And** the environment variable `UPLOAD_DATA_API_URL` is not set
**When** the dispatcher attempts to create task parameters
**Then** the schema raises a `ValueError` with message containing "upload_data_api_url must be provided"
**And** the task parameters are not created

**Example**:
```python
# Environment: UPLOAD_DATA_API_URL not set
with pytest.raises(ValueError, match="upload_data_api_url must be provided"):
    Dicom2NiiSeriesParams(
        study_uid="study-123",
        series_uid="series-456"
    )
```

### Requirement: Task Function Parameter Usage

Task functions MUST use the `upload_data_api_url` parameter from task_params instead of reading environment variables directly.

#### Scenario: Task uses parameter for API calls

**Given** a task function receives task_params with `upload_data_api_url="http://test-api.example.com"`
**When** the task function needs to make an API callback
**Then** the function uses `task_params.upload_data_api_url` as the base URL
**And** does NOT call `os.getenv("UPLOAD_DATA_API_URL")`
**And** constructs the full API endpoint by concatenating the base URL with the path

**Example**:
```python
@Booster(...)
def dicom_2_nii_series(func_params: Dict[str, any]):
    task_params = Dicom2NiiSeriesParams.model_validate(func_params)

    # Use parameter, not environment variable
    UPLOAD_DATA_API_URL = task_params.upload_data_api_url  # ✅ Correct
    # UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")  # ❌ Wrong

    # Make API call
    call_post_httpx.push({
        'url': f"{UPLOAD_DATA_API_URL}{sync_urls.SYNC_PROT_OPE_NO}",
        'data': event_data
    })
```

#### Scenario: Multiple tasks route to different APIs

**Given** a Production backend dispatches task A with `upload_data_api_url="http://prod-api.com"`
**And** a Testing backend dispatches task B with `upload_data_api_url="http://test-api.com"`
**And** both tasks are processed by the same worker pool
**When** the worker executes task A
**Then** task A makes callbacks to "http://prod-api.com"
**When** the worker executes task B
**Then** task B makes callbacks to "http://test-api.com"
**And** the worker's environment variables do NOT affect routing

**Validation**: Integration test confirms correct API routing in dual-deployment scenario

### Requirement: Backend Service Dispatcher Integration

Backend services that dispatch tasks MUST provide `upload_data_api_url` parameter explicitly using centralized configuration.

#### Scenario: Backend service dispatches task with API URL

**Given** a backend service needs to dispatch a `dicom_2_nii_series` task
**And** the service has access to `get_upload_data_api_url()` helper function
**When** the service constructs task parameters
**Then** the service calls `get_upload_data_api_url()` to get the current environment's API URL
**And** adds `upload_data_api_url` to the task parameter dictionary
**And** dispatches the task with the complete parameters

**Example**:
```python
from backend.app.config.api_urls import get_upload_data_api_url

# In service method
task_dict = task_params.get_str_dict()
task_dict['upload_data_api_url'] = get_upload_data_api_url()
dicom_2_nii_series.push(task_dict)
```

#### Scenario: Configuration helper provides correct URL per environment

**Given** the Production environment has `UPLOAD_DATA_API_URL=http://prod-api.com`
**And** the Testing environment has `UPLOAD_DATA_API_URL=http://test-api.com`
**When** Production backend calls `get_upload_data_api_url()`
**Then** the function returns "http://prod-api.com"
**When** Testing backend calls `get_upload_data_api_url()`
**Then** the function returns "http://test-api.com"

**Validation**: Unit tests verify helper function returns correct URLs for each environment

## MODIFIED Requirements

None - This is a new capability being added.

## REMOVED Requirements

None - Existing environment variable mechanism remains as fallback for backward compatibility.

## Dependencies

### Internal Dependencies

- **Capability**: `funboost-task-queue` - Task parameter serialization and queue dispatch
- **Capability**: `pydantic-schema-validation` - Parameter validation and model_validator support
- **Capability**: `backend-api-configuration` - Environment variable loading and management

### External Dependencies

- **Funboost**: Task parameter JSON serialization via `BaseJsonAbleModel`
- **Pydantic**: Schema validation via `@model_validator` decorator
- **Python-dotenv**: Environment variable loading via `load_dotenv()`

## Migration Path

### Phase 1: Backward Compatible Addition (Recommended)

1. Add `upload_data_api_url` as optional parameter with environment fallback
2. Update task functions to use parameter instead of direct env access
3. Update backend services to pass parameter explicitly
4. Monitor and validate in production
5. **Optional**: Remove environment fallback after successful migration

### Phase 2: Strict Parameter Enforcement (Future)

If environment fallback is removed:
1. Make `upload_data_api_url` required (remove `Optional`)
2. Remove `@model_validator` environment fallback logic
3. Update all callers to provide parameter explicitly
4. Remove environment variable dependency entirely

**Recommendation**: Stay in Phase 1 indefinitely for flexibility and backward compatibility.

## Testing Requirements

### Unit Test Coverage

- ✅ Schema validation with explicit parameter
- ✅ Schema validation with environment fallback
- ✅ Invalid URL format rejection
- ✅ Missing URL configuration error
- ✅ Configuration helper URL retrieval

### Integration Test Coverage

- ✅ Production environment API routing
- ✅ Testing environment API routing
- ✅ Dual deployment routing correctness
- ✅ Backward compatibility with environment variables

### Acceptance Criteria

- All scenarios in this spec have corresponding automated tests
- Test coverage for parameter logic ≥ 90%
- Zero task routing failures in staging deployment
- Migration guide validated by manual walkthrough

## Implementation Notes

### Validation Logic

```python
@model_validator(mode='after')
def validate_upload_url(self):
    """Fallback to environment variable if not provided."""
    if self.upload_data_api_url is None:
        import os
        self.upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        if self.upload_data_api_url is None:
            raise ValueError(
                "upload_data_api_url must be provided or UPLOAD_DATA_API_URL must be set"
            )

    # Validate URL format
    if not self.upload_data_api_url.startswith(('http://', 'https://')):
        raise ValueError(
            f"upload_data_api_url must be valid HTTP URL: {self.upload_data_api_url}"
        )

    return self
```

### Security Considerations

- URLs are not sensitive credentials (internal service addresses)
- No additional security risks introduced
- Parameter visibility in logs is acceptable for debugging
- Consider URL whitelisting for production environments

### Performance Considerations

- URL validation adds negligible overhead (~1μs per task)
- No impact on task execution performance
- Queue message size increases by ~50 bytes (URL string)

## Related Capabilities

- **`dual-deployment-infrastructure`**: Enables Production/Testing separation while sharing workers
- **`distributed-frequency-control`**: Maintains shared queue requirement for QPS management
- **`environment-aware-configuration`**: Provides foundation for per-environment settings

## Revision History

- **1.0.0 (2024-12-24)**: Initial specification for parameter-based API URL configuration
