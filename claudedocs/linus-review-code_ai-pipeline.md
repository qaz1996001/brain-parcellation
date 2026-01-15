# Code_AI Pipeline Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `code_ai/pipeline/`
- **Files Reviewed**: `main.py`, `cmb.py`, `pipeline_cmb_tensorflow.py`, `dicomseg/*`, `upload/*`
- **Total Lines**: ~5000+

---

## Critical Issues

### 1. CRITICAL: `main.py` - 316 Line Function
**File**: `code_ai/pipeline/main.py`
**Function**: `main()`
**Lines**: 386-702

This is a "god function" that violates every Linus principle:
- Handles file I/O, model loading, inference, postprocessing
- 5+ levels of nesting
- Complex branching based on `args.template`

### 2. CRITICAL: `CMBServiceTF` Class - 455 Lines
**File**: `code_ai/pipeline/cmb.py`

Monolithic class with multiple 50+ line methods:
- `sliding_window_inference` (lines 269-307)
- `object_analysis` (lines 341-392)

### 3. Code Duplication Across Pipeline Files

Four near-identical `pipeline_synthseg_*.py` files:
- `pipeline_synthseg_tensorflow.py`
- `pipeline_synthseg_wmh_tensorflow.py`
- `pipeline_synthseg_dwi_tensorflow.py`
- `pipeline_synthseg5class_tensorflow.py`

All share the same structure with minimal differences.

---

## Magic Numbers Inventory

### CMB Model Parameters
| File | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| cmb.py | `(64, 64, 64)` | Patch size | `cmb.model.patch_size` |
| cmb.py | `0.125` | Gaussian sigma | `cmb.model.sigma` |
| cmb.py | `0.5` | Overlap ratio | `cmb.model.overlap` |
| cmb.py | `0.084` | Min probability | `cmb.thresholds.min_probability` |
| cmb.py | `0.357` | FP reduction | `cmb.thresholds.fp_reduction` |
| cmb.py | `0.5175` | Uncertainty | `cmb.thresholds.uncertainty` |
| cmb.py | `0.6` | Intensity mean | `cmb.thresholds.intensity_mean` |

### GPU Settings
| File | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| pipeline_cmb_tensorflow.py | `0.6` | GPU memory threshold | `gpu.memory_threshold` |
| pipeline_synthseg*.py | `0.6` | GPU memory threshold | `gpu.memory_threshold` |

### Upload Parameters
| File | Value | Purpose | TOML Key |
|------|-------|---------|----------|
| upload_dicom_seg.py | `128` | Concurrent limit | `upload.concurrent_limit` |
| upload_dicom_seg.py | `300` | Orthanc timeout | `upload.orthanc_timeout` |
| upload_dicom_seg.py | `500` | Batch size | `upload.batch_size` |
| inference_complete.py | `30.0` | HTTP timeout | `upload.http_timeout` |

---

## Magic Strings Inventory

| File | String | Purpose | TOML Key |
|------|--------|---------|----------|
| cmb.py | `"gaussian"` | Blend mode | `cmb.model.blend_mode` |
| cmb.py | Model path strings | Model directories | `cmb.models.*` |
| pipeline_cmb.py | `"Pred_CMB.nii.gz"` | Output filename | `cmb.output.nifti` |
| pipeline_cmb.py | `"Pred_CMB.json"` | Output filename | `cmb.output.json` |
| pipeline_cmb.py | `"cmb_model"` | Model name | `cmb.model_name` |
| dicomseg/build/cmb.py | `"48c0cfa2-..."` | Hardcoded UUID | `cmb.model_id` |
| main.py | `5` | Default depth | `main.depth_number` |

---

## Hardcoded UUIDs

**File**: `code_ai/pipeline/dicomseg/build/cmb.py`
```python
'48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6'  # CMB model ID
```

Should be in configuration.

---

## Function Length Violations

| Function | Lines | File |
|----------|-------|------|
| `main()` | 316 | main.py |
| `pipeline_cmb` | 152 | pipeline_cmb_tensorflow.py |
| `CMBServiceTF` class | 455 | cmb.py |
| `sliding_window_inference` | 38 | cmb.py |
| `object_analysis` | 51 | cmb.py |
| `create_dicom_seg_file` | 73 | dicomseg/utils/base.py |

---

## Dead Code to Remove

1. **upload_dicom_seg.py lines 103-151**: Commented out code block
2. **pipeline/__init__.py lines 57-130**: Old PipelineConfig class commented out
3. **Duplicate file**: `upload_dicom_seg.py` and `orthanc_dicom.py` are identical

---

## Typos Found

| File | Wrong | Correct |
|------|-------|---------|
| dicomseg/utils/base.py | `reslut_list` | `result_list` |

---

## Suggested TOML Configuration

```toml
[pipeline.cmb]
# Model parameters
patch_size = [64, 64, 64]
blend_mode = "gaussian"
sigma = 0.125
overlap = 0.5
n_class = 1

# Thresholds
min_probability = 0.084
fp_reduction = 0.357
uncertainty = 0.5175
label_threshold = 0.05
intensity_mean = 0.6
spacing_upsampling = 0.6

# Watershed
ball_radius = 4
zoom_factors = [1, 1, 3]
crop_size = 26

# Output
output_nii = "Pred_CMB.nii.gz"
output_json = "Pred_CMB.json"
model_name = "cmb_model"
model_id = "48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6"

[pipeline.upload]
concurrent_limit = 128
orthanc_timeout_seconds = 300
batch_size = 500
http_timeout_seconds = 30.0

[pipeline.gpu]
memory_threshold = 0.6

[pipeline.main]
depth_number = 5
cmb_file = "CMB"
dwi_file = "DWI"
wmh_file = "WMH_PVS"

[pipeline.dicomseg]
content_creator = "Reader1"
series_number = "300"
algorithm_type = "MANUAL"
```

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | FAIL |
| Nesting < 3 levels | FAIL |
| No magic numbers | FAIL |
| No hardcoded UUIDs | FAIL |
| No code duplication | FAIL |
| DRY principle | FAIL |
