# Backend Sync Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `backend/app/sync/`
- **Files Reviewed**: `service.py`, `routers.py`, `schemas.py`
- **Total Lines**: ~1172 (service.py)

---

## Critical Issues

### 1. SQL Typo: "stydy" instead of "study"
**File**: `backend/app/sync/service.py`
**Impact**: May cause SQL errors or incorrect data retrieval

```python
# Found in SQL query
"SELECT * FROM stydy_table..."  # Should be "study_table"
```

### 2. Multiple Functions Exceeding 50+ Lines
| Function | Lines |
|----------|-------|
| `sync_study_from_platform` | 85 |
| `process_series_sync` | 72 |
| `handle_sync_callback` | 68 |

---

## Magic Numbers Inventory

| File | Line | Value | Purpose | TOML Key |
|------|------|-------|---------|----------|
| service.py | ~50 | `21600` | Redis cache expiry (6h) | `sync.redis_ttl` |
| service.py | ~80 | `180` | HTTP timeout | `sync.http_timeout` |
| service.py | ~120 | `50` | Default page size | `sync.default_page_size` |
| service.py | ~125 | `20` | Batch size | `sync.batch_size` |

---

## Magic Strings Inventory

| File | Line | String | Purpose | TOML Key |
|------|------|--------|---------|----------|
| service.py | ~30 | `"DICOM_TOOL"` | Tool identifier | `sync.tools.dicom` |
| service.py | ~35 | `"NIFTI_TOOL"` | Tool identifier | `sync.tools.nifti` |
| service.py | ~40 | `"INFERENCE_TOOL"` | Tool identifier | `sync.tools.inference` |
| service.py | SQL | `"stydy"` | TYPO - should be "study" | N/A (fix bug) |

---

## Tool ID Duplication

The same tool IDs appear in multiple modules:
- `backend/app/sync/service.py`: `"DICOM_TOOL"`, `"NIFTI_TOOL"`
- `backend/app/inference/service.py`: `"INFERENCE_TOOL"`
- `code_ai/task/task_dicom2nii.py`: `"DICOM_TOOL"`, `"NIFTI_TOOL"`

**Recommendation**: Centralize tool IDs in TOML or constants module.

---

## Suggested TOML Configuration

```toml
[sync]
# HTTP settings
http_timeout_seconds = 180

# Redis cache
redis_ttl_seconds = 21600  # 6 hours

# Pagination
default_page_size = 50
batch_size = 20

[sync.tools]
# Tool identifiers (shared across modules)
dicom = "DICOM_TOOL"
nifti = "NIFTI_TOOL"
inference = "INFERENCE_TOOL"
```

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | FAIL |
| Nesting < 3 levels | BORDERLINE |
| No magic numbers | FAIL |
| No typos in SQL | FAIL |
| DRY principle | FAIL |
