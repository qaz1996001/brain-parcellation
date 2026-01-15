# Backend Inference Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `backend/app/inference/`
- **Files Reviewed**: `service.py`, `routers.py`, `schemas.py`, `deps.py`
- **Total Lines**: ~1417 (service.py)

---

## Critical Issues

### 1. CRITICAL: `queue_series_inference` - 296 lines
**File**: `backend/app/inference/service.py`

This is a "god function" that handles:
- Series validation
- Path construction
- DWI expansion
- Task dispatch
- Error handling
- Event creation

**Recommendation**: Split into:
- `_validate_series_for_inference()`
- `_construct_inference_paths()`
- `_expand_dwi_series()`
- `_dispatch_inference_task()`
- `_create_completion_event()`

### 2. Bug: Incorrect String Comparison
**File**: `backend/app/inference/service.py`
**Line**: ~450

```python
# Wrong
if rename_result is not '':

# Correct
if rename_result != '':
```

### 3. Hardcoded DWI Expansion Targets
**File**: `backend/app/inference/service.py`

```python
# Line ~200
["DWI0", "DWI1000"]  # Should be configurable
```

---

## Magic Numbers Inventory

| File | Line | Value | Purpose | TOML Key |
|------|------|-------|---------|----------|
| service.py | ~100 | `180` | HTTP timeout | `inference.http_timeout` |
| service.py | ~120 | `21600` | Redis TTL (6 hours) | `inference.redis_ttl_seconds` |
| service.py | ~300 | `600` | Subprocess timeout | `inference.subprocess_timeout` |

---

## Magic Strings Inventory

| File | Line | String | Purpose | TOML Key |
|------|------|--------|---------|----------|
| service.py | ~50 | `"INFERENCE_TOOL"` | Tool ID | `inference.tool_id` |
| service.py | ~200 | `["DWI0", "DWI1000"]` | DWI targets | `inference.dwi_targets` |
| schemas.py | ~30 | `"100.020"` | Status code | `status_codes.pending` |
| schemas.py | ~35 | `"200.150"` | Status code | `status_codes.complete` |

---

## Function Length Analysis

| Function | Lines | Status |
|----------|-------|--------|
| `queue_series_inference` | 296 | FAIL |
| `validate_series_ready` | 85 | FAIL |
| `_extract_raw_dicom_path` | 45 | FAIL |
| `_build_inference_params` | 30 | FAIL |

---

## DRY Violations

1. **HTTP timeout**: `timeout=180` appears 5+ times
2. **Redis TTL**: `ex=21600` appears 3+ times
3. **Path construction**: Similar path building logic repeated

---

## Suggested TOML Configuration

```toml
[inference]
tool_id = "INFERENCE_TOOL"
series_tool_id = "SERIES_INFERENCE_TOOL"

[inference.timeouts]
http_seconds = 180
subprocess_seconds = 600
redis_ttl_seconds = 21600

[inference.dwi]
# DWI series expansion configuration
targets = ["DWI0", "DWI1000"]
source_series = "DWI"

[inference.status_codes]
pending = "100.020"
running = "200.100"
complete = "200.150"
failed = "300.055"
```

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | FAIL |
| Nesting < 3 levels | FAIL |
| No magic numbers | FAIL |
| No string comparison bugs | FAIL |
| DRY principle | FAIL |
