# Code_AI Config Module - Linus Torvalds Philosophy Review

## Module Overview
- **Location**: `code_ai/config/`
- **Files Reviewed**: `__init__.py`, `models.py`, `loader.py`
- **Total Lines**: ~320

---

## Assessment: GOOD Overall

This module follows Linus principles well compared to other modules.

---

## Positive Observations

### Clean Data Structures
**File**: `code_ai/config/models.py`

```python
@dataclass(frozen=True)
class ModelConfig:
    path_synthseg: Path
    gpu_n: int = 0

@dataclass(frozen=True)
class PathConfig:
    path_code: Path
    path_process: Path
    path_json: Path
    path_log: Path
```

Frozen dataclasses = immutable = good.

### Single Source of Truth
**File**: `code_ai/config/loader.py`

The docstring explicitly states:
> "This is the SINGLE SOURCE OF TRUTH for environment configuration."

---

## Minor Issues

### 1. Duplicated Default Values
**Problem**: Same default values defined in both `models.py` and `loader.py`

```python
# models.py
gpu_n: int = 0

# loader.py
DEFAULT_CONFIG = CodeAIConfig(
    model=ModelConfig(gpu_n=0)  # Duplicated!
)
```

**Recommendation**: Define defaults in ONE place only.

### 2. Magic /tmp Paths
**File**: `code_ai/config/loader.py`

```python
DEFAULT_CONFIG = CodeAIConfig(
    model=ModelConfig(path_synthseg=Path("/tmp/models/synthseg")),
    paths=PathConfig(
        path_process=Path("/tmp/process"),
        path_json=Path("/tmp/json"),
        path_log=Path("/tmp/logs")
    )
)
```

**Recommendation**: Move to TOML configuration.

### 3. Environment Variable Names as Inline Strings
```python
_get_env_or_default("PATH_SYNTHSEG", None, fail_safe)
_get_env_or_default("GPU_N", None, fail_safe)
_get_env_or_default("PATH_CODE", None, fail_safe)
```

**Recommendation**: Create constants class:
```python
class EnvVars:
    PATH_SYNTHSEG = "PATH_SYNTHSEG"
    GPU_N = "GPU_N"
    PATH_CODE = "PATH_CODE"
```

---

## Magic Numbers

| File | Line | Value | Purpose | TOML Key |
|------|------|-------|---------|----------|
| models.py | 31 | `0` | Default GPU | `model.default_gpu` |
| models.py | 67 | `"3"` | TF log level | `tensorflow.log_level` |
| models.py | 68 | `"0"` | Mixed precision | `tensorflow.mixed_precision` |

---

## Function Length Analysis

| Function | Lines | Status |
|----------|-------|--------|
| `load_code_ai_config_from_env` | ~80 | FAIL (but acceptable) |
| `_get_env_or_default` | ~25 | BORDERLINE |
| `_convert_to_path` | ~20 | PASS |
| `_convert_to_int` | ~20 | PASS |

**Note**: The main loader function is long but repetitive in a consistent pattern. Could be improved by breaking into sub-loaders.

---

## Suggested TOML Configuration

```toml
[code_ai.model]
default_gpu = 0
synthseg_path = "/tmp/models/synthseg"

[code_ai.paths]
code = "."
process = "/tmp/process"
json = "/tmp/json"
log = "/tmp/logs"

[code_ai.tensorflow]
cpp_min_log_level = "3"      # 0=all, 1=info, 2=warnings, 3=errors
auto_mixed_precision = "0"   # 0=off, 1=on

[code_ai.environment_mapping]
# Document env var to config key mapping
synthseg_path = "PATH_SYNTHSEG"
gpu_n = "GPU_N"
code_path = "PATH_CODE"
process_path = "PATH_PROCESS"
json_path = "PATH_JSON"
log_path = "PATH_LOG"
tf_log_level = "TF_CPP_MIN_LOG_LEVEL"
tf_mixed_precision = "TF_ENABLE_AUTO_MIXED_PRECISION"
```

---

## Compliance Summary

| Criterion | Status |
|-----------|--------|
| Functions < 24 lines | BORDERLINE |
| Nesting < 3 levels | PASS |
| No magic numbers | FAIL (minor) |
| Immutable data structures | PASS |
| Single source of truth | PASS |
| Clear documentation | PASS |
