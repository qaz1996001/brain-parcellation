# Design: Fix target_labels Parameter Duplication

## Architectural Context

### Current System Architecture

```
Backend (AI Model Level)
    ↓ passes target_labels=["DWI0", "DWI1000"]
_batch_convert_series_to_nifti
    ↓ for each (series_uid, target_label)
    ↓ calls with target_label="DWI1000"
_convert_single_series_to_nifti
    ↓ IGNORES target_label
    ↓ derives series_name from path
    ↓ gets series_name="DWI0" (WRONG!)
Output: DWI0.nii.gz (expected: DWI1000.nii.gz)
```

### Design Problem: Impedance Mismatch

The current design has an **abstraction level impedance mismatch**:

| Component | Abstraction Level | Data Source | Authority |
|-----------|-------------------|-------------|-----------|
| Backend | AI Model Level | Pydantic schemas, DB queries | Defines what series to create |
| `_batch_convert` | AI Model Level | Receives `target_labels` | Orchestrates conversions |
| `_convert_single` | **Filesystem Level** | Derives from `rename_dicom_path.name` | **Ignores caller intent** |

**The Bug**: Callee operates at a lower abstraction level than the caller.

## Design Principles Applied

### Knuth Principle 1: Mathematical Precision

**Current State - Undefined Invariant**:
```python
# Implicit assumption (never verified):
# ∀i: rename_dicom_path.name == target_labels[i]
#
# Reality from logs:
# target_labels[1] = "DWI1000"
# rename_dicom_path.name = "DWI0"
# Invariant VIOLATED ❌
```

**Design Fix - Explicit Authority**:
```python
# New invariant (explicitly enforced):
# series_name := target_label (parameter is authoritative)
# output_file := f"{series_name}.nii.gz"
#
# Guarantee:
# If target_label="DWI1000" → output="DWI1000.nii.gz" ✅
```

### Knuth Principle 2: Literate Programming

**Current Code - Literary Deception**:
```python
def _convert_single_series_to_nifti(
    # ... parameters ...
    # target_label is IMPLIED but not in signature
):
    # Reader expects: use target_label
    # Code does: derive from path
    # Result: CONFUSION
```

**Design Fix - Honest Contract**:
```python
def _convert_single_series_to_nifti(
    raw_dicom_path: str,
    output_dicom_base: str,
    output_nifti_base: str,
    series_uid: str,
    study_id: str,
    target_label: str,  # ← EXPLICIT: "I use this to name output"
) -> tuple:
    """
    Convert single series: raw_dicom → rename_dicom → nifti.

    Args:
        target_label: AI model's expected series name (e.g., "DWI1000").
                      This is the AUTHORITATIVE source for output naming.
    """
    # Reader expectation: ✅ matches implementation
```

### Knuth Principle 3: Data Flow Integrity

**Current Design - Ambiguous Flow**:
```
Backend calculates target_label="DWI1000"
    ↓ (pass but ignore)
Worker derives series_name="DWI0" from path
    ↓ (use derived value)
Output: DWI0.nii.gz ❌

Question: Which is authoritative? (UNDEFINED)
```

**New Design - Explicit Flow**:
```
Backend calculates target_label="DWI1000"
    ↓ (pass as parameter)
Worker uses target_label directly
    ↓ series_name := target_label
Output: DWI1000.nii.gz ✅

Authority: Caller (Backend) defines intent, Callee (Worker) honors it
```

## Design Decisions

### Decision 1: Parameter Authority

**Choice**: Make `target_label` the **single source of truth** for series naming.

**Rationale**:
1. **Caller Intent**: Backend knows what the AI model needs (DWI0 vs DWI1000)
2. **Abstraction Level**: Worker should operate at caller's level (AI model), not filesystem level
3. **DRY Principle**: One source of truth eliminates divergence risk

**Alternative Rejected**: Derive from path
- ❌ Loses caller's intent
- ❌ Assumes path structure matches AI model expectation
- ❌ Brittle (breaks if rename logic changes)

### Decision 2: Study ID Handling

**Current Implementation**:
```python
study_folder = rename_dicom_path.parent  # Derived from path
series_name = rename_dicom_path.name     # Derived from path

nifti_study_path = output_nifti_base_path / study_folder.name
nifti_series_path = nifti_study_path / series_name
```

**Issue**: Derives `study_folder.name` from filesystem, but we have `study_id` parameter.

**Design Fix**: Use `study_id` parameter directly
```python
series_name = target_label  # ✅ Use parameter

nifti_study_path = output_nifti_base_path / study_id  # ✅ Use parameter
nifti_series_path = nifti_study_path / series_name
nifti_file_path = pathlib.Path(f"{nifti_series_path}.nii.gz")
```

**Rationale**:
- Consistent with parameter authority principle
- `study_id` is already passed, should be used
- Eliminates another derivation point

### Decision 3: Path Validation Strategy

**Question**: Should we validate `rename_dicom_path.name == target_label`?

**Options**:
1. **No Validation** (trust caller)
   - ✅ Simpler
   - ❌ Silent failures if mismatch

2. **Assertion** (fail fast)
   - ✅ Catches bugs early
   - ❌ Will fail in production (current logs show mismatch)

3. **Warning Log** (best effort)
   - ✅ Visibility without crashing
   - ❌ Doesn't prevent bad output

**Decision**: Start with **Option 1** (no validation), but add **logging** for observability.

**Rationale**:
- Option 2 would fail immediately (paths currently don't match)
- Option 3 adds complexity without fixing root cause
- Option 1 + logging provides observability while we transition

**Implementation**:
```python
# Log both for comparison (during transition)
logger.info(
    f"Series conversion: target={target_label}, "
    f"path_derived={rename_dicom_path.name}"
)

# Use target_label regardless
series_name = target_label
```

### Decision 4: Backward Compatibility

**Question**: Will this break existing behavior?

**Analysis**:
- These are **private functions** (`_convert_single_series_to_nifti`)
- No external API contracts
- Only caller is `_batch_convert_series_to_nifti` (same file)

**Decision**: No backward compatibility needed.

**Risk Mitigation**:
- Add unit tests before refactoring
- Validate with integration tests (DWI series conversion)
- Monitor logs after deployment

## Implementation Plan

### Phase 1: Add target_label Parameter

**File**: `code_ai/task/task_pipeline.py`

**Change 1**: Update function signature
```python
def _convert_single_series_to_nifti(
    raw_dicom_path: str,
    output_dicom_base: str,
    output_nifti_base: str,
    series_uid: str,
    study_id: str,
    target_label: str,  # ← NEW PARAMETER
) -> tuple:
```

**Change 2**: Update caller
```python
def _batch_convert_series_to_nifti(...):
    for i, (raw_path, series_uid, target_label) in enumerate(...):
        dicom_path, nifti_path = _convert_single_series_to_nifti(
            raw_dicom_path=raw_path,
            output_dicom_base=output_dicom_base,
            output_nifti_base=output_nifti_base,
            series_uid=series_uid,
            study_id=study_id,
            target_label=target_label,  # ← PASS PARAMETER
        )
```

### Phase 2: Use target_label for Naming

**Change 3**: Replace derivation with parameter usage
```python
# OLD CODE (line 434-435):
study_folder = rename_dicom_path.parent
series_name = rename_dicom_path.name

# NEW CODE:
series_name = target_label  # Authoritative source
```

**Change 4**: Use study_id parameter
```python
# OLD CODE (line 438):
nifti_study_path = output_nifti_base_path / study_folder.name

# NEW CODE:
nifti_study_path = output_nifti_base_path / study_id
```

### Phase 3: Add Observability

**Change 5**: Log for validation
```python
logger.info(
    f"DICOM renamed: {raw_dicom_path} → {rename_dicom_path}, "
    f"using target_label={target_label}"
)
```

## Testing Strategy

### Unit Tests

**Test 1**: Parameter is used
```python
def test_convert_single_series_uses_target_label():
    result_dicom, result_nifti = _convert_single_series_to_nifti(
        raw_dicom_path="/path/raw/series_uid",
        output_dicom_base="/path/rename_dicom",
        output_nifti_base="/path/rename_nifti",
        series_uid="abc-123",
        study_id="study_001",
        target_label="DWI1000",
    )

    assert "DWI1000.nii.gz" in result_nifti
    assert "DWI1000" in result_dicom
```

**Test 2**: DWI series expansion
```python
def test_batch_convert_dwi_expansion():
    dicom_paths, nifti_paths = _batch_convert_series_to_nifti(
        raw_dicom_paths=["/raw/dwi", "/raw/dwi"],
        series_uids=["uid-dwi", "uid-dwi"],
        target_labels=["DWI0", "DWI1000"],  # Backend expansion
        output_dicom_base="/rename_dicom",
        output_nifti_base="/rename_nifti",
        study_id="study_001",
        study_uid="study_uid_001",
    )

    assert "DWI0.nii.gz" in nifti_paths[0]
    assert "DWI1000.nii.gz" in nifti_paths[1]
```

### Integration Tests

**Test 3**: End-to-end DWI conversion
```python
def test_dwi_series_conversion_integration():
    # Given: Raw DICOM DWI series (single series from MRI)
    # When: Backend expands to DWI0 + DWI1000 targets
    # Then: Worker produces both DWI0.nii.gz and DWI1000.nii.gz
```

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Output path mismatch | Medium | High | Add logging, validate with tests |
| Breaking existing pipelines | Low | High | Comprehensive integration tests |
| Performance regression | Low | Low | No algorithmic changes |
| Data loss | Low | Critical | Existing files untouched (only affects new conversions) |

## Success Metrics

1. **Code Quality**:
   - ✅ Function signature matches implementation
   - ✅ No unused parameters
   - ✅ Single source of truth for series naming

2. **Correctness**:
   - ✅ Logs show: `target_label="DWI1000"` → output `"DWI1000.nii.gz"`
   - ✅ No mismatches between caller intent and callee behavior

3. **Test Coverage**:
   - ✅ Unit tests for parameter usage
   - ✅ Integration tests for DWI expansion

4. **Observability**:
   - ✅ Logs clearly show which `target_label` is being processed
   - ✅ No silent data corruption

## Related Changes

- `refactor-to-pure-functions`: Pure function pattern for task parameters
- `parameterize-task-pipeline-paths`: Dual deployment GPU architecture
- `add-series-inference-api`: Series-level processing patterns

## References

- Donald Knuth Philosophy Guide (§2 Literate Programming, §3 Mathematical Precision, §5 Abstraction Levels)
- CLAUDE.md - Backend DWI Expansion Pattern
- `code_ai/task/task_pipeline.py` - Current implementation
