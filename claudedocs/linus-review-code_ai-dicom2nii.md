# Code_AI DICOM2NII Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `code_ai/dicom2nii/`
- **Files Reviewed**: `main.py`, `convert/*.py`
- **Total Lines**: ~5000+

---

## Critical Issues

### 1. CRITICAL: `dicom_rename_mr.py` - 2039 Lines
**File**: `code_ai/dicom2nii/convert/dicom_rename_mr.py`

This file is massively too large:
- Contains 10+ processing strategy classes
- Each strategy has duplicated `type_process` pattern
- Should be split into multiple files

### 2. CRITICAL: `dicom_rename.py` - 999 Lines
**File**: `code_ai/dicom2nii/convert/dicom_rename.py`

Contains:
- Duplicate class definitions (DwiProcessingStrategy defined TWICE)
- T1/T2 strategies with nearly identical code

### 3. Deep Nesting in Processing Functions
Multiple functions have 4+ levels of nesting:
- `ADCProcessingStrategy.process()` - 82 lines
- `nii_to_dicom()` - 49 lines
- `T1ProcessingStrategy.type_process()` - 165 lines

---

## Magic Numbers Inventory

### File Size Thresholds (KB)
| File | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| convert_nifti_postprocess.py | `100` | ADC threshold | `postprocess.adc_kb` |
| convert_nifti_postprocess.py | `800` | SWAN threshold | `postprocess.swan_kb` |
| convert_nifti_postprocess.py | `800` | T1 threshold | `postprocess.t1_kb` |
| convert_nifti_postprocess.py | `800` | T2 threshold | `postprocess.t2_kb` |
| convert_nifti_postprocess.py | `1024` | Default (1MB) | `postprocess.default_kb` |

### TR/TE Thresholds (MRI Parameters)
| File | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| dicom_rename_mr.py | `800` | T1 FLAIR TR min | `mr.t1.flair_tr_min` |
| dicom_rename_mr.py | `3000` | T1 FLAIR TR max | `mr.t1.flair_tr_max` |
| dicom_rename_mr.py | `30` | T1 FLAIR TE max | `mr.t1.flair_te_max` |
| dicom_rename_mr.py | `80` | T2 FLAIR TE min | `mr.t2.flair_te_min` |
| dicom_rename_mr.py | `5990` | T2 FLAIR TR min | `mr.t2.flair_tr_min` |
| dicom_rename_mr.py | `10000` | T2 FLAIR TR max | `mr.t2.flair_tr_max` |

### Worker Thread Counts
| File | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| main.py | `4` | Default workers | `workers.default` |
| main.py | `2` | Max DICOM workers | `workers.max_dicom` |
| convert_nifti.py | `8` | Max NIfTI workers | `workers.max_nifti` |

### B-Values (DWI)
| Value | Purpose | TOML Key |
|-------|---------|----------|
| `0` | DWI b=0 | `dwi.b_value_0` |
| `1000` | DWI b=1000 | `dwi.b_value_1000` |

---

## Magic Strings Inventory

### Regex Patterns
| Pattern | Purpose | TOML Key |
|---------|---------|----------|
| `.*(DWI\|AUTODIFF).*` | DWI detection | `patterns.dwi` |
| `.*(?<!e)(ADC\|Apparent Diffusion Coefficient).*` | ADC detection | `patterns.adc` |
| `.*(eADC).*` | eADC detection | `patterns.eadc` |
| `.*(?<!e)(SWAN).*` | SWAN detection | `patterns.swan` |
| `.+(TOF)(((?!Neck).)*)$` | TOF Brain | `patterns.tof_brain` |
| `.*(TOF).*((Neck+).*)$` | TOF Neck | `patterns.tof_neck` |

### DICOM Image Types
| String | Purpose | TOML Key |
|--------|---------|----------|
| `ORIGINAL` | Image type check | `image_types.original` |
| `DERIVED` | Image type check | `image_types.derived` |
| `REFORMATTED` | Image type check | `image_types.reformatted` |
| `MIN IP` | Minimum intensity | `image_types.min_ip` |

### File Extensions
| String | Purpose | TOML Key |
|--------|---------|----------|
| `.nii.gz` | NIfTI compressed | `extensions.nifti_gz` |
| `.json` | JSON sidecar | `extensions.json` |
| `.dcm` | DICOM files | `extensions.dicom` |
| `.meta` | Metadata folder | `folders.meta` |

---

## DICOM Tags Used

Should be centralized as constants:

| Tag | Purpose |
|-----|---------|
| `(0x08, 0x60)` | Modality |
| `(0x08, 0x08)` | Image Type |
| `(0x08, 0x103E)` | Series Description |
| `(0x18, 0x80)` | Repetition Time |
| `(0x18, 0x81)` | Echo Time |
| `(0x20, 0x37)` | Image Orientation Patient |
| `(0x43, 0x1039)` | GE Private: B-Values |

---

## DRY Violations

1. **T1/T2 Strategies**: Nearly identical code in both `dicom_rename.py` and `dicom_rename_mr.py`
2. **Worker capping**: `min(N, max(1, args.work))` pattern repeated 6+ times
3. **File extension checking**: Same pattern for `.nii.gz`, `.json`, `.dcm` repeated

---

## Dead Code to Remove

1. **convert.py lines 25-57**: Entire main functionality commented out
2. **dicom_rename.py**: Duplicate `DwiProcessingStrategy` class
3. **convert_nifti.py line 26**: Commented CSV export

---

## Typos Found

| File | Context |
|------|---------|
| dicom_rename_mr.py:1084 | `mr_acquisition_typemr_acquisition_type` |

---

## Mutable Class Attributes (Anti-pattern)

```python
class SomeStrategy:
    series_group_fn_list = []  # Shared across ALL instances!
```

This causes potential state leakage between instances.

---

## Suggested TOML Configuration

```toml
[dicom2nii.workers]
default = 4
max_dicom = 4
max_nifti = 8
max_nifti_to_dicom = 2

[dicom2nii.dcm2niix]
compression = "y"
output_format = "{filename}"

[dicom2nii.files]
nifti_extension = ".nii.gz"
json_extension = ".json"
dicom_extension = ".dcm"
meta_folder = ".meta"

[dicom2nii.postprocess.file_sizes_kb]
default = 1024
adc = 100
dwi = 100
swan = 800
t1 = 800
t2 = 800

[dicom2nii.dwi]
b_value_0 = 0
b_value_1000 = 1000

[dicom2nii.mr.thresholds.t1]
flair_tr_min = 800
flair_tr_max = 3000
flair_te_max = 30
plain_tr_max = 800

[dicom2nii.mr.thresholds.t2]
flair_te_min = 80
flair_tr_min = 5990
flair_tr_max = 10000

[dicom2nii.mr.image_types]
original = "ORIGINAL"
derived = "DERIVED"
reformatted = "REFORMATTED"
min_ip = "MIN IP"

[dicom2nii.patterns]
dwi = ".*(DWI|AUTODIFF).*"
adc = ".*(?<!e)(ADC|Apparent Diffusion Coefficient).*"
eadc = ".*(eADC).*"
swan = ".*(?<!e)(SWAN).*"
tof_brain = ".+(TOF)(((?!Neck).)*)$"
tof_neck = ".*(TOF).*((Neck+).*)$"
t1 = ".*(T1|AX|COR|SAG).*"
t2 = ".*(T2).*"
flair = "(FLAIR)"
bravo = ".*(BRAVO|FSPGR).*"
contrast = "(\\+C|C\\+)"

[dicom2nii.dicom.tags]
modality = "0008,0060"
image_type = "0008,0008"
series_description = "0008,103E"
repetition_time = "0018,0080"
echo_time = "0018,0081"
image_orientation = "0020,0037"

[dicom2nii.dicom.private_tags]
ge_b_values = "0043,1039"
```

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | FAIL |
| Files < 500 lines | FAIL (2039 lines!) |
| Nesting < 3 levels | FAIL |
| No magic numbers | FAIL |
| No duplicate code | FAIL |
| No mutable class attrs | FAIL |
| DRY principle | FAIL |
