# Series Conversion Flow Specification

## Overview

完整的 Series Level 轉換流程支援兩種模式：直接模式和轉換模式。

## ADDED Requirements

### Requirement: API 支援轉換模式請求

The Series Inference API MUST support conversion mode requests where the caller specifies `needs_conversion=true` along with `raw_dicom_series_paths`.

#### Scenario: Caller explicitly requests conversion mode

**Given** a POST request to `/api/v1/inference/series` with:
```json
{
  "study_uid": "1.2.3.4.5",
  "series_uids": ["series-a", "series-b"],
  "model_name": "synthseg",
  "model_version": "1.0",
  "needs_conversion": true,
  "raw_dicom_series_paths": ["/raw/series-a", "/raw/series-b"]
}
```
**When** the request is processed
**Then** the task is queued with `needs_conversion=true` and `raw_dicom_series_paths`
**And** the response status is `queued`

---

### Requirement: 自動偵測 Series 狀態

The inference service MUST automatically detect Series conversion status from DCOP events when `needs_conversion` is not explicitly specified.

#### Scenario: Series has CONVERSION_COMPLETE event

**Given** a Series with `SERIES_CONVERSION_COMPLETE` event in DCOP
**When** the Series is included in an inference request
**Then** the service uses direct mode with `nifti_series_paths` extracted from the event
**And** `needs_conversion` is NOT set in task parameters

#### Scenario: Series has only TRANSFER_COMPLETE event

**Given** a Series with `SERIES_TRANSFER_COMPLETE` event but no `SERIES_CONVERSION_COMPLETE`
**When** the Series is included in an inference request
**Then** the service uses conversion mode with `needs_conversion=true`
**And** `raw_dicom_series_paths` is extracted from the transfer event

#### Scenario: Series has no transfer event

**Given** a Series with no DCOP events
**When** the Series is included in an inference request
**Then** the Series is rejected with reason "Series not transferred"
**And** the rejection is included in the response

---

### Requirement: 混合狀態批次處理

The inference service MUST handle batch requests where Series have different conversion states.

#### Scenario: Batch with mixed conversion states

**Given** a batch request with:
- `series-a`: has `SERIES_CONVERSION_COMPLETE`
- `series-b`: has only `SERIES_TRANSFER_COMPLETE`
- `series-c`: has no DCOP events

**When** the request is processed
**Then** the response includes:
- `series-a`: accepted (direct mode)
- `series-b`: accepted (conversion mode)
- `series-c`: rejected with reason

**And** the task is queued with both direct and conversion mode Series

---

### Requirement: 路徑配置支援

The task path configuration MUST include `path_rename_dicom` and `path_rename_nifti` for conversion mode.

#### Scenario: Conversion mode receives path configuration

**Given** environment variables `PATH_RENAME_DICOM` and `PATH_RENAME_NIFTI` are set
**When** a conversion mode task is queued
**Then** the task parameters include:
- `path_rename_dicom`: value from configuration
- `path_rename_nifti`: value from configuration

#### Scenario: Missing conversion path configuration

**Given** `PATH_RENAME_DICOM` or `PATH_RENAME_NIFTI` is not set
**When** a conversion mode task is queued
**Then** a `ValueError` is raised with clear error message
**And** the request fails with 500 status

---

## MODIFIED Requirements

### Requirement: 擴充 validate_series_ready 返回值

The `validate_series_ready` method MUST return additional information about conversion mode requirements.

#### Scenario: Return value includes mode information

**Given** a list of series UIDs to validate
**When** `validate_series_ready` is called
**Then** the return value includes:
- `accepted`: list of accepted series UIDs
- `rejected`: list of rejected series with reasons
- `nifti_paths`: list of NIfTI paths (for direct mode series)
- `conversion_required`: list of series requiring conversion
- `raw_dicom_paths`: list of raw DICOM paths (for conversion mode series)

---

### Requirement: 擴充任務參數構建

The `queue_series_inference` method MUST build task parameters based on Series conversion state.

#### Scenario: Task params for direct mode only

**Given** all accepted Series have `SERIES_CONVERSION_COMPLETE`
**When** task parameters are built
**Then** `func_params` includes `nifti_series_paths`
**And** `func_params` does NOT include `needs_conversion`

#### Scenario: Task params for conversion mode

**Given** some accepted Series require conversion
**When** task parameters are built
**Then** `func_params` includes:
- `needs_conversion`: true
- `raw_dicom_series_paths`: paths for conversion mode series
- `nifti_series_paths`: paths for direct mode series
- `path_rename_dicom`: from configuration
- `path_rename_nifti`: from configuration

---

## Related Capabilities

- **series-inference**: Series Level inference API (existing)
- **task-pipeline**: Worker task execution (existing, not modified)
- **dcop-events**: Event tracking system (existing, not modified)
