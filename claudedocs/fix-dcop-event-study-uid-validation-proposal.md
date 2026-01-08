# OpenSpec Proposal: Fix DCOP Event study_uid Validation Error

**Created**: 2026-01-07
**Status**: Ready for Review
**Change ID**: `fix-dcop-event-study-uid-validation`

---

## Executive Summary

Created comprehensive OpenSpec proposal following Donald Knuth's Literate Programming philosophy to fix a Pydantic validation error in Series Level inference. The error occurs when Worker attempts to send DCOP conversion events with `study_uid=None`, causing event tracking failure.

## Problem Statement

### Error Manifestation

```
Failed to send conversion event for DWI0:
1 validation error for DCOPEventRequest
study_uid
  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
```

### Root Cause

In `code_ai/task/task_pipeline.py:540`, the `_batch_convert_series_to_nifti()` function passes `study_uid=None` to DCOPEventRequest, but:

1. **Pydantic Model Requirement**: `DCOPEventRequest` requires `study_uid: str` (not Optional[str])
2. **Broken Call Chain**: Parent function `_task_series_pipeline_inference()` has `study_uid` in `func_params` but doesn't pass it
3. **False Assumption**: Comment "Series Level 可能沒有 study_uid" is incorrect - Series ALWAYS belong to a Study

### Impact

- ❌ DCOP conversion events fail to send (3/3 events in test case)
- ❌ Backend cannot trace conversion events to Study records
- ❌ Event tracking chain incomplete
- ✅ Inference still succeeds (events are optional for pipeline execution)

---

## Solution Design

### Knuth Principles Applied

**Principle 2 - Literate Programming**:
> "Let us concentrate rather on explaining to human beings what we want a computer to do."

- Document structure: Proposal → Design → Tasks → Spec
- Mathematical invariants defined first, then implementation
- Code reads like a mathematical proof

**Principle 3 - Mathematical Rigor**:
> "每一個變數、每一個邊界條件都要精確定義。不容許模糊——如果你無法精確描述,你就不理解。"

- Three invariants defined:
  1. Event Integrity: `∀ event: DCOPEventRequest, event.study_uid ≠ None`
  2. Traceability: `event.study_uid` must be traceable to database
  3. Parameter Chain: `study_uid` must propagate through call chain

**Principle 6 - Error Recording**:
> "Record every mistake... such information is invaluable."

- Created error log entry #001 with full context
- Extracted learning points for future prevention

### Core Solution

**Fix the parameter passing chain**:

```
Before (broken):
  func_params["study_uid"] → ❌ not passed → _batch_convert_series_to_nifti()
    → study_uid=None → ❌ validation fails

After (fixed):
  func_params["study_uid"] → ✅ passed → _batch_convert_series_to_nifti(study_uid)
    → study_uid=study_uid → ✅ validation succeeds
```

---

## Implementation Changes

### Change 1: Function Signature

**File**: `code_ai/task/task_pipeline.py:467-474`

```python
def _batch_convert_series_to_nifti(
    raw_dicom_paths: List[str],
    series_uids: List[str],
    target_labels: List[str],
    output_dicom_base: str,
    output_nifti_base: str,
    study_id: str,
    study_uid: str,  # ✅ ADDED
    upload_data_api_url: Optional[str] = None,
) -> tuple:
```

### Change 2: Call Site

**File**: `code_ai/task/task_pipeline.py:990-998`

```python
dicom_series_paths, nifti_paths = _batch_convert_series_to_nifti(
    raw_dicom_paths=raw_dicom_paths,
    series_uids=series_uids,
    target_labels=target_labels,
    output_dicom_base=path_rename_dicom,
    output_nifti_base=path_rename_nifti,
    study_id=study_id,
    study_uid=study_uid,  # ✅ ADDED
    upload_data_api_url=upload_data_api_url,
)
```

### Change 3: Event Construction

**File**: `code_ai/task/task_pipeline.py:540-556`

```python
dcop_event = DCOPEventRequest(
    study_uid=study_uid,  # ✅ MODIFIED: was None
    series_uid=series_uid,
    study_id=study_id,
    ope_no=status,
    tool_id="NIFTI_TOOL",
    result_data={...},
)
```

---

## OpenSpec Structure

### Files Created

```
openspec/changes/fix-dcop-event-study-uid-validation/
├── proposal.md        # Knuth-style mathematical analysis
├── design.md          # Architectural decisions and impact
├── tasks.md           # Ordered implementation steps
└── specs/
    └── worker/
        └── spec.md    # 3 MODIFIED requirements with scenarios
```

### Validation Status

```bash
$ openspec validate fix-dcop-event-study-uid-validation
Change 'fix-dcop-event-study-uid-validation' is valid ✅
```

---

## Requirements Summary

### REQ-1: Series Conversion with study_uid

The `_batch_convert_series_to_nifti()` function SHALL accept a `study_uid` parameter and use it when creating DCOP conversion events.

**Scenarios**:
- DWI0 conversion sends event with study_uid
- Multiple series conversions preserve study_uid

### REQ-2: Series Pipeline Passes study_uid

The `_task_series_pipeline_inference()` function SHALL extract `study_uid` from `func_params` and pass it to `_batch_convert_series_to_nifti()`.

**Scenarios**:
- Series inference propagates study_uid

### REQ-3: DCOP Event Validation Passes

All DCOP conversion events created by the worker SHALL pass Pydantic validation without errors.

**Scenarios**:
- DCOP event model validation succeeds
- DCOP event with None study_uid fails validation (negative test)

---

## Task Execution Order

```
Task 1: 修改函數簽名添加 study_uid 參數
  ↓
Task 2: 更新調用點傳遞 study_uid
  ↓
Task 3: 使用 study_uid 參數替代 None
  ↓
Task 4: 運行 Code Quality Checks (ty, ruff)
  ↓
Task 5: 端到端驗證
  ↓
Task 6: 添加回歸測試 (可選)
```

---

## Testing Strategy

### Unit Tests

```python
def test_batch_convert_series_includes_study_uid():
    """Verify function signature accepts study_uid parameter"""

def test_dcop_event_has_valid_study_uid():
    """Verify DCOPEventRequest contains valid study_uid"""

def test_dcop_event_pydantic_validation_passes():
    """Verify no ValidationError is raised"""
```

### Integration Test

```bash
# Trigger Series Level inference
curl -X POST http://localhost:8000/api/v1/inference/series \
  -d '{"study_uid": "...", "series_uids": [...], "model_id": "infarct_v1"}'

# Verify no validation errors in logs
tail -f /path/logs/*.log | grep "Failed to send conversion event"
# Should have NO output ✅
```

### End-to-End Test

1. Full Study sync → Series conversion → Inference flow
2. Verify Backend database Study status updates
3. Confirm all DCOP events successfully recorded

---

## Risk Assessment

| Aspect | Risk Level | Mitigation |
|--------|-----------|------------|
| **Impact** | Low | Single call site, isolated change |
| **Complexity** | Low | Simple parameter passing |
| **Reversibility** | High | Fully reversible, no DB migrations |
| **Testing** | Low | Multiple test levels planned |
| **Performance** | None | O(1) parameter passing, ~8 bytes |

---

## Success Criteria

- ✅ DCOP event sending success rate: 0% → 100%
- ✅ Pydantic validation errors: 3/3 → 0/3
- ✅ Backend event traceability: 0% → 100%
- ✅ All code quality checks pass (ty, ruff)
- ✅ Mathematical invariants verified

---

## Rollback Plan

**If issues arise**:
1. Revert function signature (remove `study_uid` parameter)
2. Revert call site (remove parameter passing)
3. Revert DCOPEventRequest (back to `study_uid=None`)
4. Re-run quality checks to confirm stability

**Rollback Risk**: Very low (no external dependencies, no data migrations)

---

## Knuth Philosophy Integration

### Error Log Entry #001

**Error Type**: Parameter chain breakage + Pydantic validation failure

**Discovery**: 2026-01-07 21:05:34 during Series Level inference execution

**Root Cause**:
1. Refactoring assumption "Series Level 可能沒有 study_uid" was incorrect
2. Parent function had `study_uid` available but didn't pass it
3. No end-to-end test caught the validation failure

**Learning Points**:
- ✅ Refactoring must maintain invariant integrity
- ✅ Assumptions (like "Series has no study_uid") need verification
- ✅ End-to-end tests can catch call chain breakage early

### Code Aesthetics (Knuth's Standards)

1. **Readability** ✅: Clear parameter names, explicit dependencies
2. **Mathematical Rigor** ✅: Three invariants defined and provable
3. **Maintainability** ✅: Clear extension points, no implicit assumptions
4. **Testability** ✅: Pure function design, automatable invariant tests

---

## Documentation

### Primary Documents

- **proposal.md**: 367 lines, Knuth-style mathematical analysis with Chinese commentary
- **design.md**: Architecture decisions, data flow diagrams, implementation details
- **tasks.md**: 6 ordered tasks with validation criteria
- **specs/worker/spec.md**: 3 MODIFIED requirements with 5 scenarios

### Related Documents

- **claudedocs/knuth-analysis-series-uids-refactoring.md**: Previous UID/Label separation refactoring (context)
- **CLAUDE.md**: Project-level documentation (update recommended for "Common Pitfalls" section)

---

## Next Steps

1. **Review**: Technical review of proposal by team
2. **Approval**: Get sign-off on changes
3. **Implementation**: Execute Task 1-6 in order
4. **Validation**: Run all tests and verify success criteria
5. **Documentation**: Update CLAUDE.md with new pitfall entry

---

## Timeline Estimate

- **Code Changes** (Task 1-3): ~30 minutes
- **Quality Checks** (Task 4): ~10 minutes
- **E2E Verification** (Task 5): ~20 minutes
- **Regression Tests** (Task 6, optional): ~1 hour
- **Total**: ~2 hours including testing

---

## Conclusion

This OpenSpec proposal demonstrates Literate Programming principles:

> "The best programs are written so that computing machines can perform them quickly and so that human beings can understand them clearly."
> — Donald Knuth

By applying Knuth's philosophy:
- **Art**: Elegant code structure with clear naming
- **Science**: Mathematical invariants, provable correctness
- **Literature**: Reads like a paper, explains "why" before "how"

Six months from now, any maintainer (including ourselves) will **immediately understand** the intent and correctness of this code.

---

**Proposal Author**: Claude Sonnet 4.5
**Philosophy Guide**: Donald Knuth - Literate Programming
**Status**: ✅ Valid OpenSpec change, ready for review
