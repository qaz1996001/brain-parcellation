# Proposal: Fix target_labels Parameter Duplication

## Problem Statement

The function `_batch_convert_series_to_nifti` passes `target_labels` parameter to `_convert_single_series_to_nifti`, but the callee function does not use this parameter. Instead, `_convert_single_series_to_nifti` internally re-derives the series name from the DICOM rename path (`rename_dicom_path.name`), which is fundamentally redundant and violates core software engineering principles.

**Evidence from logs**:
```
2026-01-07 21:11:27 - _batch_convert_series_to_nifti - INFO - Converting series 2/3: DWI1000 (UID: 308454c5-d2ff7ec1-74ce99a4-4281ba1b-a3a1d20b)
2026-01-07 21:11:27 - _convert_single_series_to_nifti - INFO - DICOM renamed: /mnt/e/pipeline/test/raw_dicom/.../308454c5... → .../10089413_20210201_MR_21002010079/DWI0
```

**The Issue**:
- `_batch_convert_series_to_nifti` logs "Converting series 2/3: **DWI1000**" (using `target_label`)
- `_convert_single_series_to_nifti` renames to "**.../DWI0**" (re-deriving from path)
- The mismatch reveals that `target_label` is **passed but ignored**, violating data flow integrity

## Philosophical Analysis: Donald Knuth Principles Violated

### 1. **Literate Programming Violation** (文學式程式設計)

> "Let us concentrate rather on explaining to human beings what we want a computer to do."
> — Knuth, *Literate Programming*

**Current State**: The code tells two conflicting stories:
- **Narrative 1** (in `_batch_convert_series_to_nifti`): "I'm passing `target_label` to tell you what series name to use"
- **Narrative 2** (in `_convert_single_series_to_nifti`): "I ignore your label and derive my own from the path"

**Knuth's Critique**: A human reader cannot understand the intended data flow. The function signature **lies** by accepting a parameter it doesn't use. This is literary deception.

**What Knuth Would Say**:
```
If target_label is important enough to pass, it is important enough to USE.
If it is not used, it should not be in the signature.
A function's parameters are its CONTRACT with the reader.
```

### 2. **Mathematical Precision Violation** (數學嚴謹性)

> "Every variable, every boundary condition must be precisely defined."
> — Knuth Philosophy Guide, §3.3

**Current State**: The function has an **undefined invariant**:
- Is `series_name = rename_dicom_path.name` guaranteed to equal `target_label`?
- If yes → parameter is redundant (DRY violation)
- If no → which one is authoritative? (ambiguity)

**Knuth's Invariant Analysis**:
```python
# Declared invariant (line 485-486):
assert len(series_uids) == len(target_labels) == len(raw_dicom_paths)

# Missing invariant (should be declared but isn't):
# ∀i: rename_dicom_path.name == target_labels[i] after rename operation
```

**The Bug**: The code assumes an invariant without **proving** or **testing** it. From the log evidence:
- Caller expects: `target_label = "DWI1000"`
- Function produces: `series_name = "DWI0"` (from path)
- **Invariant violated!**

### 3. **Data Flow Integrity Violation** (數據流完整性)

> "The psychological profiling of a programmer is mostly the ability to shift levels of abstraction."
> — Knuth, on abstraction levels (§5.1)

**Current State**: Confusion of abstraction levels:

| Abstraction Level | Data Source | Authority |
|-------------------|-------------|-----------|
| **High-level intent** | `target_labels[i]` from caller | Backend expansion logic |
| **Low-level implementation** | `rename_dicom_path.name` | DICOM file metadata |

**The Problem**: The function **receives high-level intent but obeys low-level implementation**, creating a semantic mismatch.

**Knuth's Perspective**:
```
The caller (batch function) operates at the AI-model abstraction level:
  "I want DWI1000.nii.gz for the AI model"

The callee (single function) operates at the filesystem abstraction level:
  "I found DWI0 in the path, so I'll use that"

This is a LEVEL MISMATCH. The function must honor the caller's level.
```

### 4. **"Don't Repeat Yourself" Violation** (DRY Principle)

**Current State**: Two sources of truth for the same information:
1. `target_label` parameter (explicit intent)
2. `rename_dicom_path.name` (implicit derivation)

**Knuth's Analysis**:
```
Information redundancy without validation creates opportunity for divergence.
When two sources claim the same truth, ONE must be authoritative.
The authoritative source should be the EXPLICIT one, not the DERIVED one.
```

**Evidence of Divergence** (from logs):
- Source 1 (`target_label`): "DWI1000"
- Source 2 (`path.name`): "DWI0"
- **Result**: Silent data corruption (wrong series name used)

### 5. **"Beware of Bugs" - Testing Philosophy**

> "Beware of bugs in the above code; I have only proved it correct, not tried it."
> — Knuth (§3.2)

**Current State**: The code has neither:
1. **Proof** that `rename_dicom_path.name == target_label`
2. **Tests** that validate this invariant
3. **Assertions** that fail fast when invariant breaks

**What's Missing**:
```python
# Should exist but doesn't:
def _convert_single_series_to_nifti(
    raw_dicom_path: str,
    output_dicom_base: str,
    output_nifti_base: str,
    series_uid: str,
    study_id: str,
    target_label: str,  # ← Actually use this parameter!
) -> tuple:
    # ... rename logic ...

    # Knuth's advice: VERIFY your assumptions
    assert rename_dicom_path.name == target_label, (
        f"Invariant violation: path produced {rename_dicom_path.name}, "
        f"but caller expected {target_label}"
    )
```

## Root Cause Analysis

The issue stems from **Backend DWI Expansion Pattern** (documented in CLAUDE.md):

1. **Backend** expands `DWI` → `["DWI0", "DWI1000"]` (AI model level)
2. **Backend** passes `target_labels = ["DWI0", "DWI1000"]` to Worker
3. **Worker batch function** receives and logs `target_label = "DWI1000"`
4. **Worker single function** ignores `target_label`, derives from path → gets `"DWI0"`

**The Mismatch**: Backend's expansion logic is not aligned with Worker's path derivation logic.

## Proposed Solution

### Option A: Use target_label as Authoritative Source (Recommended)

**Principle**: Honor the caller's explicit intent over implicit derivation.

**Change**:
```python
def _convert_single_series_to_nifti(
    raw_dicom_path: str,
    output_dicom_base: str,
    output_nifti_base: str,
    series_uid: str,
    study_id: str,
    target_label: str,  # ← Add this parameter
) -> tuple:
    # ... DICOM rename logic ...

    # Step 2: Use target_label directly (don't derive from path)
    series_name = target_label  # ✅ Explicit, authoritative

    # Calculate NIFTI output paths
    nifti_study_path = output_nifti_base_path / study_id
    nifti_series_path = nifti_study_path / series_name
    nifti_file_path = pathlib.Path(f"{nifti_series_path}.nii.gz")

    logger.info(f"Running dcm2niix for: {rename_dicom_path} → {nifti_file_path}")
```

**Advantages**:
- ✅ Eliminates redundancy (single source of truth)
- ✅ Honors caller's intent (AI model level abstraction)
- ✅ Makes data flow explicit and testable
- ✅ Aligns with Backend expansion pattern

**Risks**:
- ⚠️ Requires validation that `rename_dicom_path` structure matches expectation
- ⚠️ May need study_id parameter instead of deriving from path

### Option B: Remove target_label and Document Derivation

**Principle**: If not using the parameter, don't accept it.

**Change**: Remove `target_label` from `_batch_convert_series_to_nifti` call, document that series name is derived from DICOM metadata.

**Disadvantages**:
- ❌ Loses caller's explicit intent
- ❌ Doesn't solve the DWI0/DWI1000 mismatch issue
- ❌ Violates Backend expansion pattern design

### Option C: Validate Invariant with Assertion

**Principle**: If two sources must agree, verify they do.

**Change**: Keep both, but assert they match:
```python
derived_name = rename_dicom_path.name
assert derived_name == target_label, (
    f"Path-derived name '{derived_name}' doesn't match "
    f"caller's target '{target_label}'"
)
```

**Issues**:
- ❌ Still has redundancy
- ❌ Will fail in production (as logs show they differ)
- ✅ But fails fast and makes the bug visible

## Recommendation

**Adopt Option A**: Use `target_label` as the authoritative source.

**Rationale** (Knuth Principles):
1. **Literate Programming**: Makes intent explicit through parameter usage
2. **Mathematical Precision**: Single source of truth = no invariant to prove
3. **Abstraction Levels**: Function operates at caller's abstraction level (AI model)
4. **Correctness**: Aligns with Backend's DWI expansion design pattern

## Impact Assessment

**Files Affected**:
- `code_ai/task/task_pipeline.py`:
  - `_convert_single_series_to_nifti()` signature and implementation
  - `_batch_convert_series_to_nifti()` call site

**Testing Required**:
- Unit tests for `_convert_single_series_to_nifti` with explicit `target_label`
- Integration tests for DWI series conversion (DWI → DWI0 + DWI1000)
- Contract tests validating `target_label` matches output file names

**Backward Compatibility**:
- No external API changes (these are private functions)
- Internal refactoring only

## Success Criteria

1. **Code Clarity**: Function signature accurately represents parameter usage
2. **Data Flow Integrity**: `target_label` is the single source of truth for series naming
3. **Log Consistency**: Batch and single function logs show matching series names
4. **Test Coverage**: Explicit tests for `target_label` parameter behavior
5. **Knuth Compliance**: Code passes all philosophical checks from §2, §3, §5

## References

- `.claude/philosophy_guide/donald_knuth_philosophy_guide.md` (§2, §3, §5)
- `CLAUDE.md` - Backend DWI Expansion Pattern
- `code_ai/task/task_pipeline.py:467-527` - Current implementation
- Production logs (2026-01-07 21:11:27) - Evidence of mismatch

---

**Change ID**: `fix-target-labels-parameter-duplication`
**Type**: Code Quality / Bug Fix
**Severity**: Medium (silent data corruption potential)
**Philosophy**: Donald Knuth - Literate Programming, Mathematical Precision, Data Flow Integrity
