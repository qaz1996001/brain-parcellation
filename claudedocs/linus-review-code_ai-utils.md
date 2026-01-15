# Code_AI Utils Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `code_ai/` (root level utils files)
- **Files Reviewed**: `utils_parcellation.py`, `utils_synthseg.py`, `utils_synthsegOnnx.py`
- **Total Lines**: ~4000+

---

## Critical Issues

### 1. CRITICAL: Hardcoded User Path
**File**: `code_ai/utils_synthsegOnnx.py`
**Line**: 527

```python
cmd = f"export FSLOUTPUTTYPE=NIFTI_GZ && /home/seanho/fsl/bin/flirt ..."
```

This will FAIL on any machine that is not seanho's. Must be environment variable.

### 2. Typo: `intput_size` (missing 'n')
**Files**: `utils_synthseg.py`, `utils_synthsegOnnx.py`
**Impact**: Confusing variable name throughout both files

```python
self.intput_size = 192  # Should be "input_size"
```

### 3. Massive Code Duplication
`utils_synthseg.py` and `utils_synthsegOnnx.py` are ~80% identical:
- Same classes: `VolumeProcessor`, `SegmentationStrategy`, etc.
- Same methods: `prepare_output_files`, `preprocess`, etc.
- Same magic numbers

**Recommendation**: Extract shared code to base module.

### 4. Sentinel Value Anti-Pattern
**File**: `utils_parcellation.py`
Multiple occurrences:

```python
distance = 999999  # Should use np.inf or float('inf')
```

Found at lines: 505, 563, 968, 1515, 2010, 2199

---

## Data Masquerading as Code

### Label Mapping Dictionaries (~1000+ lines total)
**File**: `utils_parcellation.py`

The file contains massive nested dictionaries for label mappings:
- `WhiteMatterParcellation` class: 300+ lines
- `CMBProcess` class: 100+ lines
- `DWIProcess` class: 500 lines
- `WMHProcess` class: 100+ lines

**Linus**: "Good programmers worry about data structures."

These should be in external JSON/YAML files.

---

## Function Length Violations

### utils_parcellation.py
| Function | Lines | Status |
|----------|-------|--------|
| `generate_wmparc` | 84 | FAIL |
| `ec_ic_parcellation` | 48 | FAIL |
| `hemi_revise` | 49 | FAIL |
| `cc_parcellation` | 59 | FAIL |
| `white_matter_parcellation` | 37 | FAIL |
| `re_white_matter_parcellation` | 41 | FAIL |

### utils_synthseg.py
| Function | Lines | Status |
|----------|-------|--------|
| `prepare_output_files` | 114 | FAIL |
| `run` | 84 | FAIL |
| `run_segmentations33` | 65 | FAIL |
| `build_unet_model` | 59 | FAIL |
| `preprocess` | 48 | FAIL |

---

## Magic Numbers Inventory

### utils_parcellation.py
| Value | Purpose | Count | TOML Key |
|-------|---------|-------|----------|
| `999999` | Sentinel distance | 6 | Use `np.inf` |
| `5` | Depth number | 5+ | `parcellation.depth_number` |
| `8` | Decimal places | 4 | `parcellation.decimal_places` |
| `128` | Min pad | 2 | `parcellation.min_pad` |
| `0.97` | X-axis boundary | 1 | `parcellation.x_boundary` |
| `16` | Brain stem label | 3 | `parcellation.labels.brain_stem` |
| `3`, `42` | Cortex labels | 4 | `parcellation.labels.cortex` |

### utils_synthseg.py
| Value | Purpose | Count | TOML Key |
|-------|---------|-------|----------|
| `192` | Input/crop size | 4 | `synthseg.input_size` |
| `128` | Min pad | 3 | `synthseg.min_pad` |
| `19` | Neutral labels | 1 | `synthseg.n_neutral_labels` |
| `0.25` | Posterior threshold | 3 | `synthseg.threshold` |
| `5` | UNet levels | 2 | `synthseg.unet.levels` |
| `24` | UNet features | 2 | `synthseg.unet.features` |

---

## Magic Strings Inventory

| File | String | Purpose | TOML Key |
|------|--------|---------|----------|
| utils_synthseg.py | `'synthseg_2.0.h5'` | Model file | `models.synthseg` |
| utils_synthseg.py | `'elu'` | Activation | `unet.activation` |
| utils_synthsegOnnx.py | `/home/seanho/fsl/bin/flirt` | FSL path | `fsl.flirt_path` |
| utils_parcellation.py | `'_david.nii.gz'` | Output suffix | `output.david_suffix` |
| utils_parcellation.py | `'_CMB.nii.gz'` | Output suffix | `output.cmb_suffix` |

---

## Deep Nesting Violations

**File**: `utils_parcellation.py`

```python
# Lines 618-667: hemi_revise method
while True:
    for z in range(...):
        if condition:
            for x in range(...):
                if another_condition:
                    while nested_while:  # 6 levels!
                        ...
```

---

## Bare `except:` Anti-Pattern

**Files**: `utils_synthseg.py`, `utils_synthsegOnnx.py`

```python
try:
    from tensorflow.keras.models import Model
except:  # Catches ALL exceptions, even KeyboardInterrupt
    ...
```

**Fix**: Use specific exception types:
```python
except ImportError:
    ...
```

---

## Side Effects at Import Time

**File**: `utils_parcellation.py`
**Lines**: 109-112

```python
# Configures GPU at module import time
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
```

This should be in a function, not at module level.

---

## Suggested TOML Configuration

```toml
[parcellation]
depth_number = 5
depth_number_range = [4, 10]
decimal_places = 8
min_pad = 128
inner_size = 2
outer_size = 5

[parcellation.thresholds]
x_axis_boundary = 0.97
ventricle_percentile_low = 20
ventricle_percentile_high = 80
prerequisite = 0.5

[parcellation.labels]
brain_stem = 16
brain_stem_output = 301
cortex_left = 3
cortex_right = 42

[parcellation.output_suffixes]
david = "_david.nii.gz"
cmb = "_CMB.nii.gz"
dwi = "_DWI.nii.gz"
wmh = "_WMH.nii.gz"

[parcellation.label_files]
# External JSON files for large mappings
white_matter = "config/labels/white_matter.json"
cmb = "config/labels/cmb.json"
dwi = "config/labels/dwi.json"
wmh = "config/labels/wmh.json"

[synthseg]
input_size = 192  # Fixed typo from "intput_size"
crop = 192
min_pad = 128
n_neutral_labels = 19

[synthseg.normalization]
ct_clip_min = 0
ct_clip_max = 80
rescale_min = 0.0
rescale_max = 1.0
percentile_min = 0.5
percentile_max = 99.5

[synthseg.thresholds]
mask_probability = 0.25
mask_probability_fast = 0.2
parcellation_mask = 0.1

[synthseg.unet]
levels = 5
conv_per_level = 2
conv_size = 3
features = 24
feat_mult = 2
activation = "elu"
batch_norm = -1

[synthseg.fsl]
# CRITICAL: Make configurable!
flirt_path = "${FSL_DIR}/bin/flirt"
output_type = "NIFTI_GZ"
dof = 6
cost = "corratio"
interp = "nearestneighbour"
```

---

## Files to Refactor

### High Priority
1. **Extract shared code** between `utils_synthseg.py` and `utils_synthsegOnnx.py`
2. **Externalize label mappings** from `utils_parcellation.py` to JSON files
3. **Fix hardcoded FSL path** in `utils_synthsegOnnx.py`
4. **Fix typo** `intput_size` → `input_size`

### Medium Priority
5. Replace `999999` with `np.inf`
6. Move GPU configuration to function
7. Fix bare `except:` clauses

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | FAIL |
| Nesting < 3 levels | FAIL |
| No magic numbers | FAIL |
| No hardcoded paths | FAIL |
| No typos | FAIL |
| DRY principle | FAIL |
| Data in config files | FAIL |
