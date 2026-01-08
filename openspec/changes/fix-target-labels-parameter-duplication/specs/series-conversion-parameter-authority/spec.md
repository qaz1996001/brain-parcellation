# Spec: Series Conversion Parameter Authority

## ADDED Requirements

### Requirement: Target Label Parameter Usage

The `_convert_single_series_to_nifti` function MUST use the `target_label` parameter as the authoritative source for output series naming, not derive series name from filesystem paths.

**Requirement ID**: SCPA-001
**Type**: Functional
**Priority**: High
**Rationale**: Eliminates data flow ambiguity and honors caller's intent (Knuth: Literate Programming principle)

#### Scenario: Single Series Conversion with Explicit Target Label

**Given**:
- Raw DICOM path: `/path/raw/raw_dicom/study_uid/series_uid`
- Target label: `"DWI1000"` (from Backend expansion)
- Study ID: `"10089413_20210201_MR_21002010079"`
- Output base paths configured

**When**:
- `_convert_single_series_to_nifti()` is called with `target_label="DWI1000"`

**Then**:
- Output NIfTI path MUST be: `/path/rename_nifti/{study_id}/DWI1000.nii.gz`
- Output DICOM path MUST be: `/path/rename_dicom/{study_id}/DWI1000/`
- Function MUST NOT derive series name from `rename_dicom_path.name`
- Log MUST show: `"using target_label=DWI1000"`

#### Scenario: DWI Series Expansion (DWI0 + DWI1000)

**Given**:
- Single raw DICOM DWI series (from MRI machine)
- Backend expansion: `target_labels = ["DWI0", "DWI1000"]`
- Same `series_uid` for both targets (Backend DWI Expansion Pattern)

**When**:
- `_batch_convert_series_to_nifti()` is called with:
  ```python
  raw_dicom_paths=["/raw/dwi", "/raw/dwi"]
  series_uids=["uid-dwi", "uid-dwi"]
  target_labels=["DWI0", "DWI1000"]
  ```

**Then**:
- First call to `_convert_single_series_to_nifti`:
  - MUST use `target_label="DWI0"`
  - MUST produce: `/path/rename_nifti/{study_id}/DWI0.nii.gz`
- Second call to `_convert_single_series_to_nifti`:
  - MUST use `target_label="DWI1000"`
  - MUST produce: `/path/rename_nifti/{study_id}/DWI1000.nii.gz`
- Both outputs MUST exist with correct names
- NO silent name mismatch (e.g., expecting DWI1000 but getting DWI0)

---

### Requirement: Study ID Parameter Usage

The `_convert_single_series_to_nifti` function MUST use the `study_id` parameter directly for path construction, not derive study folder name from `rename_dicom_path.parent.name`.

**Requirement ID**: SCPA-002
**Type**: Functional
**Priority**: Medium
**Rationale**: Consistency with parameter authority principle; eliminates filesystem derivation

#### Scenario: Study ID in Output Path Construction

**Given**:
- Study ID parameter: `"10089413_20210201_MR_21002010079"`
- Rename DICOM path: `/path/rename_dicom/10089413_20210201_MR_21002010079/DWI0/`

**When**:
- Constructing NIfTI output path

**Then**:
- MUST use: `nifti_study_path = output_nifti_base_path / study_id`
- MUST NOT use: `nifti_study_path = output_nifti_base_path / rename_dicom_path.parent.name`
- Output path MUST be: `/path/rename_nifti/{study_id}/{series_name}.nii.gz`

---

### Requirement: Function Signature Honesty

Function signatures MUST accurately represent parameter usage. If a parameter is required for correct behavior, it MUST be in the signature and MUST be used.

**Requirement ID**: SCPA-003
**Type**: Code Quality
**Priority**: High
**Rationale**: Knuth's Literate Programming - function signature is a contract with the reader

#### Scenario: target_label Parameter in Function Signature

**Given**:
- Function `_convert_single_series_to_nifti` needs series name for output

**When**:
- Reviewing function signature

**Then**:
- Signature MUST include:
  ```python
  def _convert_single_series_to_nifti(
      raw_dicom_path: str,
      output_dicom_base: str,
      output_nifti_base: str,
      series_uid: str,
      study_id: str,
      target_label: str,  # ← MUST be present
  ) -> tuple:
  ```
- Docstring MUST document `target_label` as authoritative source
- Implementation MUST use `target_label` parameter (not derive from path)

---

### Requirement: Data Flow Observability

Conversion functions MUST log sufficient information to verify that `target_label` parameter is being used correctly.

**Requirement ID**: SCPA-004
**Type**: Observability
**Priority**: Medium
**Rationale**: Enable debugging and validation of parameter usage

#### Scenario: Logging Target Label Usage

**Given**:
- Function processing series with `target_label="DWI1000"`
- DICOM rename path derived from files

**When**:
- Function executes conversion

**Then**:
- Log MUST include:
  - `target_label` value being used
  - Output file path being created
  - Series UID being processed
- Log format example:
  ```
  INFO - _convert_single_series_to_nifti - Series conversion:
         uid=308454c5..., target=DWI1000, output=/path/.../DWI1000.nii.gz
  ```
- Log SHOULD NOT show mismatches between `target_label` and output filename

---

## MODIFIED Requirements

### Requirement: Series Conversion Data Flow (Modified)

Series conversion data flow MUST honor caller's explicit intent through parameters, not implicit filesystem structure.

**Previous**: Function derives series name from DICOM rename path structure
**New**: Function uses explicit `target_label` parameter for series naming

**Requirement ID**: SCPA-MOD-001
**Type**: Functional
**Priority**: High

#### Scenario: Data Flow from Backend to Worker

**Given**:
- Backend calculates AI model requirements: `target_labels=["DWI0", "DWI1000"]`
- Worker receives task with `target_labels` parameter

**When**:
- Worker executes `_batch_convert_series_to_nifti` → `_convert_single_series_to_nifti`

**Then**:
- Data flow MUST be:
  ```
  Backend (AI Model Level)
      ↓ calculates target_labels
  _batch_convert_series_to_nifti
      ↓ passes target_label parameter
  _convert_single_series_to_nifti
      ↓ uses target_label (NOT derives from path)
  Output: filename matches target_label ✅
  ```
- Data flow MUST NOT be:
  ```
  Backend → passes target_label → Worker ignores → derives from path ❌
  ```

---

## REMOVED Requirements

### Requirement: Series Name Derivation from Path (Removed)

**Requirement ID**: SCPA-REM-001
**Previous Behavior**: Derive `series_name` from `rename_dicom_path.name`
**Reason for Removal**: Violates parameter authority principle; causes silent data corruption

**Description**: The practice of deriving series name from filesystem path structure is REMOVED in favor of explicit parameter usage.

#### Scenario: NO Derivation from Rename Path (Anti-pattern)

**Given**:
- Function has `target_label` parameter available
- Rename DICOM path exists with name "DWI0"

**When**:
- Constructing output paths

**Then**:
- MUST NOT use: `series_name = rename_dicom_path.name`
- MUST use: `series_name = target_label`
- Even if `rename_dicom_path.name != target_label`, parameter is authoritative

---

## Cross-References

### Related Capabilities

- **Configuration Centralization** (`refactor-to-pure-functions`):
  - Both enforce "single source of truth" principle
  - Pure function parameters are authoritative sources

- **Series-Level Processing** (`add-series-inference-api`):
  - Backend DWI Expansion Pattern depends on parameter authority
  - Series validation logic assumes correct target naming

- **Path Parameterization** (`parameterize-task-pipeline-paths`):
  - Path construction uses explicit parameters
  - No environment variable derivation

### Dependencies

- **Upstream**: Backend `validate_series_ready()` must pass correct `target_labels`
- **Downstream**: DICOM-SEG creation expects filenames to match AI model expectations

---

## Validation Criteria

### Code Review Checklist

- [ ] `target_label` parameter added to `_convert_single_series_to_nifti` signature
- [ ] Function implementation uses `target_label` (not `rename_dicom_path.name`)
- [ ] Function implementation uses `study_id` parameter (not derived from path)
- [ ] Docstring documents `target_label` as authoritative source
- [ ] Caller (`_batch_convert_series_to_nifti`) passes `target_label` parameter
- [ ] Logging includes `target_label` value
- [ ] No derivation of series name from filesystem paths

### Test Coverage Checklist

- [ ] Unit test: Single series conversion with explicit `target_label`
- [ ] Unit test: DWI expansion (DWI0 + DWI1000) produces correct outputs
- [ ] Unit test: Output filename matches `target_label` parameter
- [ ] Integration test: End-to-end DWI conversion with real DICOM files
- [ ] Type checker passes: `uvx ty check code_ai/task/task_pipeline.py`
- [ ] Linter passes: `uvx ruff check code_ai/task/ --fix`

### Knuth Philosophy Compliance

- [ ] **Literate Programming**: Function signature is honest contract
- [ ] **Mathematical Precision**: Single source of truth (no ambiguity)
- [ ] **Data Flow Integrity**: Abstraction levels aligned (caller intent honored)
- [ ] **DRY Principle**: No redundant sources of truth for series naming
- [ ] **Correctness**: Tests verify parameter usage, not just behavior

---

## References

- Donald Knuth Philosophy Guide: §2 (Literate Programming), §3 (Mathematical Precision), §5 (Abstraction Levels)
- CLAUDE.md: Backend DWI Expansion Pattern
- `code_ai/task/task_pipeline.py`: Implementation file
- Production logs (2026-01-07 21:11:27): Evidence of current mismatch
