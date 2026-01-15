# Code_AI SynthSeg Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `code_ai/SynthSeg/`
- **Files Reviewed**: `predict.py`, `evaluate.py`, `__init__.py`
- **Total Lines**: ~1000

---

## Critical Issues

### 1. Monolithic `predict()` Function - 217 Lines
**File**: `code_ai/SynthSeg/predict.py`
**Lines**: 36-253

This function does everything:
- Path preparation
- Model building
- Preprocessing
- Inference
- Postprocessing
- Evaluation
- File saving

**Must decompose into 5+ functions**.

### 2. Nested Helper Functions Anti-Pattern
**File**: `code_ai/SynthSeg/predict.py`
**Lines**: 255-378 (`prepare_output_files`)

Contains 3 nearly identical nested helper functions:
- `text_helper()` - 15 lines
- `helper_dir()` - 30 lines
- `helper_im()` - 35 lines

These share logic and violate DRY. Should be unified.

---

## Function Length Violations

| Function | Lines | File |
|----------|-------|------|
| `predict` | 217 | predict.py |
| `prepare_output_files` | 123 | predict.py |
| `evaluation` | 157 | evaluate.py |
| `surface_distances` | 87 | evaluate.py |
| `build_model` | 65 | predict.py |
| `postprocess` | 57 | predict.py |
| `preprocess` | 48 | predict.py |

---

## Magic Numbers Inventory

### predict.py
| Line | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| 47 | `1.` | Target resolution mm | `synthseg.target_resolution` |
| 51 | `0.5` | Gaussian sigma | `synthseg.sigma_smoothing` |
| 53 | `5` | UNet levels | `synthseg.unet.levels` |
| 54 | `2` | Conv per level | `synthseg.unet.conv_per_level` |
| 55 | `3` | Conv kernel size | `synthseg.unet.conv_size` |
| 56 | `24` | UNet features | `synthseg.unet.features` |
| 57 | `2` | Feature multiplier | `synthseg.unet.feat_mult` |
| 177 | `10` | Verbose threshold | `synthseg.verbose_threshold` |
| 389 | `0.05` | Resolution tolerance | `synthseg.resolution_tolerance` |
| 409 | `0., 1.` | Rescale range | `synthseg.rescale_min/max` |
| 409 | `0.5, 99.5` | Percentile range | `synthseg.percentile_min/max` |
| 492 | `0.5` | LR average weight | `synthseg.lr_average_weight` |
| 509 | `0.25` | Posterior threshold | `synthseg.posterior_threshold` |
| 553 | `3` | Volume precision | `synthseg.volume_precision` |

### evaluate.py
| Line | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| 43-44 | `0.1` | Histogram bin offset | `evaluate.histogram_offset` |
| 49 | `1e-5` | Dice epsilon | `evaluate.dice_epsilon` |
| 80 | `100` | Hausdorff percentile | `evaluate.hausdorff_percentile` |
| 223 | `10` | Default crop margin | `evaluate.crop_margin` |
| 300 | `10` | Progress interval | `evaluate.progress_interval` |

---

## Magic Strings Inventory

| File | String | Purpose | TOML Key |
|------|--------|---------|----------|
| predict.py | `'elu'` | Activation | `synthseg.unet.activation` |
| predict.py | `'synthseg_robust_2.0.h5'` | Model file | `synthseg.models.robust` |
| predict.py | `'synthseg_2.0.h5'` | Model file | `synthseg.models.standard` |
| predict.py | `'synthseg_parc_2.0.h5'` | Model file | `synthseg.models.parcellation` |
| predict.py | `'.nii.gz'` | Extension | `synthseg.extensions.nifti` |
| predict.py | `'int32'`, `'float32'` | Dtypes | `synthseg.dtypes.*` |
| predict.py | `'sobel'` | Gradient op | `synthseg.gradient_operator` |
| evaluate.py | `'two-sided'` | Hypothesis | `evaluate.default_hypothesis` |

---

## File Extension Checking Anti-Pattern

```python
# Current (bad - repeated)
if path[-4:] != '.txt':
if path[-7:] == '.nii.gz':
if path[-4:] == '.nii':

# Should be
SUPPORTED_EXTENSIONS = {'.nii.gz', '.nii', '.mgz', '.npz', '.txt'}
ext = os.path.splitext(path)[1]
if ext not in SUPPORTED_EXTENSIONS:
    ...
```

---

## Suggested TOML Configuration

```toml
[synthseg]
target_resolution_mm = 1.0
resolution_tolerance = 0.05
sigma_smoothing = 0.5
posterior_threshold = 0.25
lr_average_weight = 0.5
volume_precision = 3
verbose_threshold = 10

[synthseg.normalization]
rescale_min = 0.0
rescale_max = 1.0
percentile_min = 0.5
percentile_max = 99.5

[synthseg.unet]
levels = 5
conv_per_level = 2
conv_size = 3
features = 24
feat_mult = 2
activation = "elu"
batch_norm = -1

[synthseg.models]
standard = "synthseg_2.0.h5"
robust = "synthseg_robust_2.0.h5"
parcellation = "synthseg_parc_2.0.h5"

[synthseg.labels]
segmentation = "synthseg_segmentation_labels_2.0.npy"
denoiser = "synthseg_denoiser_labels_2.0.npy"
parcellation = "synthseg_parcellation_labels.npy"
names = "synthseg_parcellation_names.npy"
topology = "synthseg_topological_classes_2.0.npy"

[synthseg.dtypes]
segmentation = "int32"
posteriors = "float32"

[synthseg.extensions]
nifti_gz = ".nii.gz"
nifti = ".nii"
mgz = ".mgz"
npz = ".npz"
txt = ".txt"
csv = ".csv"

[synthseg.output_suffixes]
segmentation = "synthseg"
posteriors = "posteriors"
resampled = "resampled"

[evaluate]
histogram_bin_offset = 0.1
dice_epsilon = 1e-5
default_hausdorff_percentile = 100
default_crop_margin = 10
progress_update_interval = 10
default_hypothesis = "two-sided"
```

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | FAIL |
| Nesting < 3 levels | FAIL |
| No magic numbers | FAIL |
| No nested helper functions | FAIL |
| DRY principle | FAIL |
| Extension checking pattern | FAIL |
