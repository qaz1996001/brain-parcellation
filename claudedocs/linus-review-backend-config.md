# Backend Config Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `backend/app/config/`
- **Files Reviewed**: `loader.py`, `models.py`, `deps.py`, `task_paths.py`, `__init__.py`

---

## Critical Issues

### 1. CRITICAL: `deps.py` - Monster Function (513 lines)
**File**: `backend/app/config/deps.py`
**Function**: `_create_filter_aggregate_function_fastapi`

This function violates every Linus principle:
- **Line count**: 513 lines (guideline: <24 lines)
- **Nesting**: 6+ levels deep
- **Single responsibility**: Does filtering, aggregation, validation, and SQL generation

**Recommendation**: Split into:
- `_build_filter_clause()`
- `_build_aggregate_clause()`
- `_validate_filter_params()`
- `_generate_sql_query()`

### 2. CRITICAL: Hardcoded Database Credentials
**File**: `backend/app/config/loader.py`

```python
# Line ~150
"postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom"
```

**Risk**: Security vulnerability - credentials in source code
**Recommendation**: Must be environment variable only

---

## Magic Numbers Inventory

| File | Line | Value | Purpose | TOML Key |
|------|------|-------|---------|----------|
| loader.py | ~80 | `8000` | Default API port | `api.port` |
| loader.py | ~85 | `8042` | Orthanc port | `orthanc.port` |
| loader.py | ~90 | `15433` | PostgreSQL port | `database.port` |
| loader.py | ~100 | `180` | HTTP timeout seconds | `http.timeout_seconds` |

---

## Magic Strings Inventory

| File | Line | String | Purpose | TOML Key |
|------|------|--------|---------|----------|
| loader.py | ~50 | `"AI_APP_PORT"` | Env var name | N/A (keep as env) |
| loader.py | ~55 | `"AI_APP_TITLE"` | Env var name | N/A |
| loader.py | ~60 | `"UPLOAD_DATA_API_URL"` | Env var name | N/A |
| loader.py | ~150 | `"postgresql+asyncpg://..."` | DB connection | `database.url` |

---

## Function Length Analysis

| Function | Lines | Status |
|----------|-------|--------|
| `_create_filter_aggregate_function_fastapi` | 513 | FAIL |
| `load_backend_config_from_env` | ~80 | FAIL |
| `_get_env_or_default` | 25 | BORDERLINE |

---

## Suggested TOML Configuration

```toml
[backend.api]
port = 8000
title = "AI Backend"
version = "1.0.0"

[backend.database]
host = "127.0.0.1"
port = 15433
name = "dicom"
# URL constructed from above, never hardcoded with credentials

[backend.orthanc]
host = "127.0.0.1"
port = 8042
timeout_seconds = 300

[backend.http]
timeout_seconds = 180
retry_count = 3
```

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | FAIL |
| Nesting < 3 levels | FAIL |
| No magic numbers | FAIL |
| No hardcoded credentials | FAIL |
| Clear variable names | PASS |
