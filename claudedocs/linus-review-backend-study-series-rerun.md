# Backend Study/Series/Rerun Modules - Linus Torvalds Philosophy Review

## Modules Overview
- **Locations**: `backend/app/study/`, `backend/app/series/`, `backend/app/rerun/`
- **Files Reviewed**: `service.py`, `routers.py`, `schemas.py` for each module

---

## Study Module Issues

### Magic Numbers
| File | Line | Value | Purpose | TOML Key |
|------|------|-------|---------|----------|
| service.py | ~50 | `180` | HTTP timeout (repeated 5x) | `study.http_timeout` |
| service.py | ~80 | `21600` | Redis cache expiry | `study.redis_ttl` |

### DRY Violation
HTTP timeout `180` is hardcoded 5 times in the same file.

---

## Series Module Issues

### 1. CRITICAL: Data Masquerading as Code
**File**: `backend/app/series/schemas.py`
**Lines**: ~140+

```python
# 140+ lines of hardcoded series sort dictionaries
SERIES_SORT_ORDER = {
    "T1_BRAVO": 1,
    "T2_FLAIR": 2,
    "DWI": 3,
    # ... 100+ more entries
}
```

**Problem**: This is data, not code. Should be in JSON/TOML file.

### 2. Anti-Pattern: Identity Lambda
**File**: `backend/app/series/routers.py`

```python
map(lambda x: x, iterable)  # Does nothing
```

**Fix**: Remove the useless lambda, just use `list(iterable)`

---

## Rerun Module Issues

### 1. Empty Files (Dead Code)
**Files**: `backend/app/rerun/model.py`, `backend/app/rerun/schemas.py`

These files are empty and should be removed.

### 2. Variable Typo
**File**: `backend/app/rerun/service.py`

```python
flage = True  # Should be "flag"
```

### 3. Magic Numbers
| File | Line | Value | Purpose | TOML Key |
|------|------|-------|---------|----------|
| service.py | ~30 | `10` | Default limit | `rerun.default_limit` |
| service.py | ~35 | `0` | Default offset | `rerun.default_offset` |

---

## Combined Magic Numbers Summary

| Module | Value | Count | TOML Key |
|--------|-------|-------|----------|
| study | `180` | 5 | `http.timeout_seconds` |
| study | `21600` | 2 | `redis.ttl_seconds` |
| series | sort dicts | 140+ lines | External JSON file |
| rerun | `10`, `0` | 2 | `pagination.default_*` |

---

## Suggested TOML Configuration

```toml
[study]
http_timeout_seconds = 180
redis_ttl_seconds = 21600

[series]
# Reference to external file for large data
sort_order_file = "config/series_sort_order.json"

[rerun]
default_limit = 10
default_offset = 0

[pagination]
# Shared pagination defaults
default_limit = 50
default_offset = 0
max_limit = 1000
```

---

## Files to Delete

1. `backend/app/rerun/model.py` - Empty file
2. `backend/app/rerun/schemas.py` - Empty file

---

## Compliance Summary

| Module | Functions < 24 | Nesting < 3 | No Magic | DRY |
|--------|----------------|-------------|----------|-----|
| study | FAIL | PASS | FAIL | FAIL |
| series | PASS | PASS | FAIL | PASS |
| rerun | PASS | PASS | FAIL | PASS |
