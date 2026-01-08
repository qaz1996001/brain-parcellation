# Design: Parameterize UPLOAD_DATA_API_URL

**Change ID**: `parameterize-upload-api-url`
**Design Version**: 1.0
**Last Updated**: 2024-12-24

## Architecture Overview

This change converts `UPLOAD_DATA_API_URL` from environment-based configuration to task-level parameter, enabling dynamic API endpoint routing based on task origin rather than worker environment.

### Current Architecture (Before)

```
┌─────────────────────┐
│  Backend Service    │
│  (Production)       │
└──────────┬──────────┘
           │ .push(task_params)
           ▼
┌─────────────────────┐
│   RabbitMQ Queue    │
│  (Shared Instance)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Worker Process     │──────▶│  os.getenv()        │
│  (ENV=production)   │      │  UPLOAD_DATA_API_URL│
└──────────┬──────────┘      └─────────────────────┘
           │
           ▼
    Production API ✅ (Correct)


┌─────────────────────┐
│  Backend Service    │
│  (Testing)          │
└──────────┬──────────┘
           │ .push(task_params)
           ▼
┌─────────────────────┐
│   RabbitMQ Queue    │
│  (Shared Instance)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Worker Process     │──────▶│  os.getenv()        │
│  (ENV=production)   │      │  UPLOAD_DATA_API_URL│
└──────────┬──────────┘      └─────────────────────┘
           │
           ▼
    Production API ❌ (Wrong! Should call Testing API)
```

**Problem**: Worker environment determines API endpoint, not task origin.

### Target Architecture (After)

```
┌─────────────────────┐
│  Backend Service    │
│  (Production)       │
│  API: prod-url      │
└──────────┬──────────┘
           │ .push({...params, upload_data_api_url: "prod-url"})
           ▼
┌─────────────────────┐
│   RabbitMQ Queue    │
│  (Shared Instance)  │
│  [task with URL]    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Worker Process     │──────▶│  task_params.       │
│  (Any Environment)  │      │  upload_data_api_url│
└──────────┬──────────┘      └─────────────────────┘
           │
           ▼
    Production API ✅ (Correct)


┌─────────────────────┐
│  Backend Service    │
│  (Testing)          │
│  API: test-url      │
└──────────┬──────────┘
           │ .push({...params, upload_data_api_url: "test-url"})
           ▼
┌─────────────────────┐
│   RabbitMQ Queue    │
│  (Shared Instance)  │
│  [task with URL]    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Worker Process     │──────▶│  task_params.       │
│  (Any Environment)  │      │  upload_data_api_url│
└──────────┬──────────┘      └─────────────────────┘
           │
           ▼
    Testing API ✅ (Correct)
```

**Solution**: Task parameter determines API endpoint, worker-agnostic.

## Implementation Strategy

### Phase 1: Schema Extension (Backward Compatible)

Add `upload_data_api_url` parameter to existing schemas with optional type and environment fallback.

#### 1.1 Update `Dicom2NiiSeriesParams`

**File**: `code_ai/task/schema/intput_params.py`

```python
class Dicom2NiiSeriesParams(Dicom2NiiParams):
    study_uid           : Optional[str]
    series_uid          : Optional[str]
    upload_data_api_url : Optional[str] = None  # NEW: Optional for backward compatibility

    @model_validator(mode='after')
    def validate_upload_url(self):
        """Fallback to environment variable if not provided."""
        if self.upload_data_api_url is None:
            import os
            self.upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
            if self.upload_data_api_url is None:
                raise ValueError("upload_data_api_url must be provided or UPLOAD_DATA_API_URL must be set")
        # Validate URL format
        if not self.upload_data_api_url.startswith(('http://', 'https://')):
            raise ValueError(f"upload_data_api_url must be valid HTTP URL: {self.upload_data_api_url}")
        return self
```

#### 1.2 Update `Dicom2NiiParams`

```python
class Dicom2NiiParams(BaseJsonAbleModel):
    sub_dir             : Optional[Path]
    output_dicom_path   : Optional[Path]
    output_nifti_path   : Optional[Path]
    upload_data_api_url : Optional[str] = None  # NEW: Optional for backward compatibility

    @model_validator(mode='after')
    def validate_upload_url(self):
        """Fallback to environment variable if not provided."""
        if self.upload_data_api_url is None:
            import os
            self.upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
            if self.upload_data_api_url is None:
                raise ValueError("upload_data_api_url must be provided or UPLOAD_DATA_API_URL must be set")
        if not self.upload_data_api_url.startswith(('http://', 'https://')):
            raise ValueError(f"upload_data_api_url must be valid HTTP URL: {self.upload_data_api_url}")
        return self
```

### Phase 2: Task Function Updates

Update task functions to use parameter instead of environment variable.

#### 2.1 Update `dicom_2_nii_series`

**File**: `code_ai/task/task_dicom2nii.py`

**Before** (Line 253):
```python
UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
```

**After**:
```python
# Get from task parameters (already validated in schema)
UPLOAD_DATA_API_URL = task_params.upload_data_api_url
```

#### 2.2 Update `process_dir`

**File**: `code_ai/task/task_dicom2nii.py`

**Before** (Line 388):
```python
UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
```

**After**:
```python
# Get from task parameters (already validated in schema)
UPLOAD_DATA_API_URL = task_params.upload_data_api_url
```

### Phase 3: Backend Service Updates

Update all callers to pass `upload_data_api_url` parameter.

#### 3.1 Central Configuration Helper

Create helper function in backend to provide consistent URL resolution:

**File**: `backend/app/config/api_urls.py` (NEW)

```python
import os
from typing import Optional

def get_upload_data_api_url(env_override: Optional[str] = None) -> str:
    """
    Get UPLOAD_DATA_API_URL for task dispatch.

    Args:
        env_override: Optional environment override ('production', 'testing')

    Returns:
        The API URL to use for uploading data

    Raises:
        ValueError: If URL cannot be determined
    """
    # Priority: explicit override > current ENV > environment variable
    if env_override:
        # Load from specific environment file
        from code_ai import load_dotenv
        load_dotenv(f'.env.{env_override}')

    url = os.getenv("UPLOAD_DATA_API_URL")
    if not url:
        raise ValueError("UPLOAD_DATA_API_URL not configured")

    return url
```

#### 3.2 Update `backend/app/sync/service.py`

**Current** (Line 369):
```python
dicom_2_nii_series.push(task_params.get_str_dict())
```

**Updated**:
```python
from backend.app.config.api_urls import get_upload_data_api_url

# Add upload_data_api_url to task params
task_dict = task_params.get_str_dict()
task_dict['upload_data_api_url'] = get_upload_data_api_url()
dicom_2_nii_series.push(task_dict)
```

**Current** (Line 479):
```python
task = dicom_to_nii.push(task_params.get_str_dict())
```

**Updated**:
```python
task_dict = task_params.get_str_dict()
task_dict['upload_data_api_url'] = get_upload_data_api_url()
task = dicom_to_nii.push(task_dict)
```

#### 3.3 Update `backend/app/listen/service.py`

Same pattern as sync/service.py:
- Line 362: `dicom_2_nii_series.push()`
- Line 418: `dicom_to_nii.push()`

#### 3.4 Update `backend/app/study/service.py`

Same pattern as sync/service.py:
- Line 361: `dicom_2_nii_series.push()`
- Line 417: `dicom_to_nii.push()`

#### 3.5 Update CLI Scripts (Optional)

**File**: `code_ai/pipeline/dicom_to_nii.py`, `code_ai/pipeline/raw_diom_to_nii_inference.py`

Add command-line argument for URL or rely on environment variable fallback.

### Phase 4: Testing Strategy

#### 4.1 Unit Tests

**File**: `tests/test_task_dicom2nii_parameterization.py` (NEW)

```python
import pytest
import os
from code_ai.task.schema.intput_params import Dicom2NiiSeriesParams, Dicom2NiiParams

def test_upload_url_from_parameter():
    """Test that upload_data_api_url parameter is used when provided."""
    params = Dicom2NiiSeriesParams(
        study_uid="test-study",
        series_uid="test-series",
        upload_data_api_url="http://test-api.com"
    )
    assert params.upload_data_api_url == "http://test-api.com"

def test_upload_url_fallback_to_env(monkeypatch):
    """Test fallback to environment variable when parameter not provided."""
    monkeypatch.setenv("UPLOAD_DATA_API_URL", "http://env-api.com")
    params = Dicom2NiiSeriesParams(
        study_uid="test-study",
        series_uid="test-series"
    )
    assert params.upload_data_api_url == "http://env-api.com"

def test_upload_url_validation_invalid():
    """Test that invalid URLs are rejected."""
    with pytest.raises(ValueError, match="must be valid HTTP URL"):
        Dicom2NiiSeriesParams(
            study_uid="test-study",
            series_uid="test-series",
            upload_data_api_url="not-a-url"
        )

def test_upload_url_missing_error(monkeypatch):
    """Test that missing URL raises error."""
    monkeypatch.delenv("UPLOAD_DATA_API_URL", raising=False)
    with pytest.raises(ValueError, match="upload_data_api_url must be provided"):
        Dicom2NiiSeriesParams(
            study_uid="test-study",
            series_uid="test-series"
        )
```

#### 4.2 Integration Tests

**File**: `tests/integration/test_dual_deployment_routing.py` (NEW)

Test that Production and Testing backends correctly route to respective APIs.

### Phase 5: Migration and Rollback

#### Migration Plan

1. **Week 1**: Deploy schema changes with backward compatibility
2. **Week 2**: Update backend services to pass parameter explicitly
3. **Week 3**: Monitor production for any issues
4. **Week 4**: Optional - remove environment variable fallback

#### Rollback Plan

If issues arise:
1. Revert backend service changes (remove explicit parameter passing)
2. Schema fallback will automatically use environment variables
3. No data migration required (parameter-based approach is backward compatible)

## Trade-offs and Alternatives

### Alternative 1: Keep Environment-Based (Rejected)

**Pros**:
- No code changes required
- Simple configuration

**Cons**:
- Cannot support dual deployment with shared workers
- Inflexible for testing and multi-tenant scenarios

### Alternative 2: Per-Queue Configuration (Rejected)

Create separate queues for Production and Testing tasks.

**Pros**:
- Clear separation of concerns
- Environment variables remain usable

**Cons**:
- Distributed frequency control breaks (workers need same queue)
- More complex queue management
- Violates Solution B architecture requirement

### Alternative 3: Task Parameter with Strict Validation (Selected)

**Pros**:
- Flexible and explicit
- Supports dual deployment
- Maintains distributed frequency control
- Testable and traceable

**Cons**:
- Requires updating all callers
- Slightly more complex than environment variables

## Security Considerations

### URL Validation

- Validate URL format at schema level
- Reject non-HTTP(S) protocols
- Optional: Whitelist allowed API domains

### Logging and Auditing

- Log API URL used for each task (for debugging)
- Do not log sensitive authentication tokens
- Consider redacting URLs in debug logs if needed

### Backward Compatibility

- Environment variable fallback maintains security posture
- No new attack surface introduced
- Task parameters already logged in current implementation

## Performance Impact

### Negligible Impact

- Parameter validation adds ~1μs per task dispatch
- No impact on worker processing time
- No additional network calls
- Queue message size increases by ~50 bytes per task (URL string)

### Monitoring

- Monitor task dispatch latency
- Track URL validation failures
- Alert on missing URL configuration

## Documentation Updates

### Required Documentation

1. **Migration Guide**: `docs/MIGRATION_UPLOAD_API_URL.md`
   - Step-by-step migration instructions
   - Before/after code examples
   - Troubleshooting common issues

2. **API Reference**: Update task parameter documentation
   - `Dicom2NiiSeriesParams.upload_data_api_url`
   - `Dicom2NiiParams.upload_data_api_url`

3. **Architecture Decision Record**: `docs/ADR-UPLOAD-API-URL-PARAMETERIZATION.md`
   - Context and motivation
   - Decision rationale
   - Consequences and trade-offs

## Success Metrics

### Functional Metrics

- ✅ Zero task failures due to wrong API endpoint after deployment
- ✅ All unit tests passing
- ✅ Integration tests verify correct routing

### Operational Metrics

- ✅ No increase in task dispatch latency (< 1ms delta)
- ✅ Zero production incidents related to URL misconfiguration
- ✅ 100% of callers updated within 2 weeks

### Quality Metrics

- ✅ Code coverage for new parameter logic > 90%
- ✅ All edge cases (missing URL, invalid URL) tested
- ✅ Documentation completeness score > 95%
