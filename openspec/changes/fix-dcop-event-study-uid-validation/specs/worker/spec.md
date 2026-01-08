# Worker Capability Specification

## MODIFIED Requirements

### Requirement: Series Conversion with study_uid

The `_batch_convert_series_to_nifti()` function SHALL accept a `study_uid` parameter and use it when creating DCOP conversion events.

**Rationale**: DCOP events require a valid `study_uid` for event traceability in the Backend. The Pydantic `DCOPEventRequest` model does not accept `None` for `study_uid`.

**Acceptance Criteria**:
- Function signature includes `study_uid: str` parameter
- DCOPEventRequest constructor uses the provided `study_uid` parameter
- No Pydantic validation errors occur when sending conversion events

#### Scenario: DWI0 conversion sends event with study_uid

**Given**:
- A DWI series with UID "308454c5-d2ff7ec1-74ce99a4-4281ba1b-a3a1d20b"
- Target label "DWI0"
- Study UID "e9364c14-51867d5e-4ecffab6-36054e99-ad1ff077"
- Study ID "14914694_20220905_MR_21109050071"

**When**:
- `_batch_convert_series_to_nifti()` is called with:
  - `series_uids=["308454c5-..."]`
  - `target_labels=["DWI0"]`
  - `study_uid="e9364c14-..."`
  - `study_id="14914694_..."`

**Then**:
- DCOP conversion event is created with:
  - `study_uid="e9364c14-..."`
  - `series_uid="308454c5-..."`
  - `study_id="14914694_..."`
  - `tool_id="NIFTI_TOOL"`
  - `ope_no="SERIES_CONVERSION_COMPLETE"`
- Event sending succeeds without Pydantic validation errors
- Log contains "Conversion complete for DWI0"
- Log does NOT contain "Failed to send conversion event"

#### Scenario: Multiple series conversions preserve study_uid

**Given**:
- Three series: ADC, DWI0, DWI1000
- All belong to study_uid "e9364c14-..."

**When**:
- `_batch_convert_series_to_nifti()` converts all three series

**Then**:
- Three DCOP events are sent
- All events have `study_uid="e9364c14-..."`
- No events have `study_uid=None`
- All events pass Pydantic validation

---

### Requirement: Series Pipeline Passes study_uid

The `_task_series_pipeline_inference()` function SHALL extract `study_uid` from `func_params` and pass it to `_batch_convert_series_to_nifti()`.

**Rationale**: The study_uid must be propagated through the call chain from the task entry point to the DCOP event creation point.

**Acceptance Criteria**:
- `study_uid` is extracted from `func_params.get("study_uid")`
- `study_uid` is passed as an argument to `_batch_convert_series_to_nifti()`
- Type checking passes without errors

#### Scenario: Series inference propagates study_uid

**Given**:
- `func_params` contains:
  ```python
  {
    "study_uid": "e9364c14-...",
    "study_id": "14914694_...",
    "series_uids": ["308454c5-...", "86364c14-..."],
    "model_id": "infarct_v1"
  }
  ```

**When**:
- `_task_series_pipeline_inference(func_params)` is executed

**Then**:
- `study_uid` variable is assigned "e9364c14-..."
- `_batch_convert_series_to_nifti()` is called with `study_uid="e9364c14-..."`
- No AttributeError or KeyError occurs

---

### Requirement: DCOP Event Validation Passes

All DCOP conversion events created by the worker SHALL pass Pydantic validation without errors.

**Rationale**: Pydantic model `DCOPEventRequest` requires `study_uid: str`, not `Optional[str]`. Validation failures prevent event tracking in Backend.

**Acceptance Criteria**:
- `DCOPEventRequest` constructor receives `study_uid` as a string
- No `ValidationError` exceptions are raised
- Event successfully serializes to JSON

#### Scenario: DCOP event model validation succeeds

**Given**:
- Valid study_uid string "e9364c14-..."

**When**:
- `DCOPEventRequest` is constructed with:
  ```python
  DCOPEventRequest(
      study_uid="e9364c14-...",
      series_uid="308454c5-...",
      study_id="14914694_...",
      ope_no="SERIES_CONVERSION_COMPLETE",
      tool_id="NIFTI_TOOL",
      result_data={...}
  )
  ```

**Then**:
- No `ValidationError` is raised
- Event object is created successfully
- `event.study_uid == "e9364c14-..."`

#### Scenario: DCOP event with None study_uid fails validation

**Given**:
- study_uid is None

**When**:
- `DCOPEventRequest` is constructed with `study_uid=None`

**Then**:
- `ValidationError` is raised
- Error message contains "Input should be a valid string"
- Error indicates field "study_uid"

---

## REMOVED Requirements

None. This change only modifies existing requirements, does not remove any.

---

## ADDED Requirements

None. This change enhances existing conversion event requirements with proper study_uid handling.

---

## Implementation Notes

### Code Changes

**File**: `code_ai/task/task_pipeline.py`

**Change 1**: Function signature (Line 467-474)
```python
def _batch_convert_series_to_nifti(
    raw_dicom_paths: List[str],
    series_uids: List[str],
    target_labels: List[str],
    output_dicom_base: str,
    output_nifti_base: str,
    study_id: str,
    study_uid: str,  # ADDED
    upload_data_api_url: Optional[str] = None,
) -> tuple:
```

**Change 2**: Call site (Line 990-998)
```python
dicom_series_paths, nifti_paths = _batch_convert_series_to_nifti(
    raw_dicom_paths=raw_dicom_paths,
    series_uids=series_uids,
    target_labels=target_labels,
    output_dicom_base=path_rename_dicom,
    output_nifti_base=path_rename_nifti,
    study_id=study_id,
    study_uid=study_uid,  # ADDED
    upload_data_api_url=upload_data_api_url,
)
```

**Change 3**: DCOPEventRequest (Line 540-556)
```python
dcop_event = DCOPEventRequest(
    study_uid=study_uid,  # MODIFIED: was None
    series_uid=series_uid,
    study_id=study_id,
    ope_no=status,
    tool_id="NIFTI_TOOL",
    result_data={...},
)
```

### Test Coverage

- Unit test: `test_batch_convert_series_includes_study_uid()`
- Unit test: `test_dcop_event_has_valid_study_uid()`
- Unit test: `test_dcop_event_pydantic_validation_passes()`
- Integration test: Series Level inference end-to-end
- Regression test: Verify no validation errors in logs

### Risk Assessment

- **Impact**: Low (single call site)
- **Complexity**: Low (parameter passing)
- **Reversibility**: High (fully reversible)
- **Testing**: High (multiple test levels)

### Traceability

| Requirement | Implementation | Test |
|------------|----------------|------|
| REQ-WORKER-001 | `_batch_convert_series_to_nifti()` signature | `test_batch_convert_series_includes_study_uid()` |
| REQ-WORKER-002 | `_task_series_pipeline_inference()` call | `test_series_pipeline_passes_study_uid()` |
| REQ-WORKER-003 | `DCOPEventRequest` constructor | `test_dcop_event_pydantic_validation_passes()` |

---

**Specification Version**: 1.0
**Author**: Claude Sonnet 4.5
**Date**: 2026-01-07
**Status**: Pending Review
