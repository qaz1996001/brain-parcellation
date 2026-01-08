# Infarct 模型验证逻辑说明

## 问题背景

Infarct 模型虽然是 `batch_inputs=True`，但**并非总是**需要传递多个 DICOM 目录。

### Study Level 的实现逻辑

Study Level 代码中有**验证和 fallback 机制**：

```python
INFARCT_TARGET_SERIES = ("ADC", "DWI0", "DWI1000")

def _resolve_infarct_dicom_inputs(task, ...):
    dicom_dirs: List[str] = []
    missing_reason: Optional[str] = None

    # 验证所有 target series
    for series in INFARCT_TARGET_SERIES:
        matched_name = _match_series_basename(candidate_names, series)
        if not matched_name:
            missing_reason = f"missing series {series}"
            break
        dicom_path = _build_dicom_series_path(...)
        if not dicom_path or not dicom_path.exists():
            missing_reason = f"dicom path not found for {matched_name}"
            break
        dicom_dirs.append(str(dicom_path))

    if missing_reason:
        logger.warning(
            "Falling back to single DICOM directory for infarct study %s: %s",
            study_id,
            missing_reason,
        )
        return default_dir, None  # ← Fallback 到单个！

    return default_dir, dicom_dirs  # ← 验证通过，返回多个
```

**关键点**:
1. ✅ 必须包含 **ADC, DWI0, DWI1000** 三个 series
2. ✅ 所有 DICOM 路径必须存在
3. ❌ **如果验证失败 → fallback 到单个 DICOM 目录**

---

## Series Level 实现

### 新增代码

**文件**: `code_ai/task/task_pipeline.py`

#### 1. 定义 Target Series 常量

```python
# Line 67
INFARCT_TARGET_SERIES = ("ADC", "DWI0", "DWI1000")
```

#### 2. 添加验证函数

```python
# Line 631-688
def _validate_infarct_target_series(
    nifti_paths: List[str], dicom_paths: List[str]
) -> bool:
    """
    驗證 Infarct 模型是否包含所有必需的 target series。

    與 Study Level 的 _resolve_infarct_dicom_inputs 邏輯一致：
    - 必須包含 ADC, DWI0, DWI1000 三個 series
    - 所有 DICOM 路徑必須非空

    如果驗證失敗，應該 fallback 到單個 DICOM 目錄。
    """
    if len(nifti_paths) != len(dicom_paths):
        return False

    # 提取 series 名稱
    series_names = []
    for path in nifti_paths:
        basename = os.path.basename(path)
        if basename.endswith(".nii.gz"):
            name = basename[:-7]
        elif basename.endswith(".nii"):
            name = basename[:-4]
        else:
            name = basename
        series_names.append(name)

    # 檢查是否包含所有 target series
    for target in INFARCT_TARGET_SERIES:
        found = False
        for name in series_names:
            # 模糊匹配（與 Study Level 一致）
            if target == name or target.lower() in name.lower():
                found = True
                break
        if not found:
            logger.warning(
                f"Infarct validation failed: missing target series '{target}'"
            )
            return False

    # 檢查所有 DICOM 路徑都非空
    for i, dicom_path in enumerate(dicom_paths):
        if not dicom_path or dicom_path == "":
            logger.warning(
                f"Infarct validation failed: empty DICOM path at index {i}"
            )
            return False

    return True
```

#### 3. Batch Mode 中使用验证

```python
# Line 1076-1120
is_infarct = _is_infarct_model(model_id)

# 決定使用多個 DICOM 目錄還是單個
use_multiple_dicom_dirs = False
if is_infarct:
    # Infarct: 驗證是否包含所有必需的 target series
    if _validate_infarct_target_series(valid_nifti_paths, valid_dicom_dirs):
        use_multiple_dicom_dirs = True
        logger.info(
            f"Infarct validation passed: using {len(valid_dicom_dirs)} DICOM directories"
        )
    else:
        logger.warning(
            "Infarct validation failed: falling back to single DICOM directory"
        )

if use_multiple_dicom_dirs:
    # Infarct (驗證通過): 傳入所有對應的 DICOM 目錄
    inference_cmd = _build_series_inference_cmd(
        ...,
        dicom_dirs=valid_dicom_dirs,  # 多個
    )
else:
    # CMB 或 Infarct (驗證失敗): 只傳入第一個有效的 DICOM 目錄
    dicom_dir = next((path for path in valid_dicom_dirs if path), None)
    inference_cmd = _build_series_inference_cmd(
        ...,
        dicom_dir=dicom_dir,  # 單個
    )
```

---

## 验证场景

### Scenario 1: 验证通过（标准 Infarct）

**输入**:
```python
series_uids = ["series_ADC", "series_DWI0", "series_DWI1000"]
nifti_paths = [
    "/path/ADC.nii.gz",
    "/path/DWI0.nii.gz",
    "/path/DWI1000.nii.gz"
]
dicom_paths = [
    "/rename_dicom/.../ADC",
    "/rename_dicom/.../DWI0",
    "/rename_dicom/.../DWI1000"
]
```

**验证结果**: ✅ **PASS**
- 包含所有 3 个 target series
- 所有 DICOM 路径非空

**生成的命令**:
```bash
bash pipeline_infarct.sh \
  study_id \
  ADC.nii.gz DWI0.nii.gz DWI1000.nii.gz \      # 3 NIfTI
  /rename_dicom/.../ADC \                      # 3 DICOM dirs ✅
  /rename_dicom/.../DWI0 \
  /rename_dicom/.../DWI1000 \
  output
```

**日志**:
```
[INFO] Infarct validation passed: using 3 DICOM directories
```

---

### Scenario 2: 缺少 target series (Fallback)

**输入**:
```python
series_uids = ["series_ADC", "series_DWI0"]  # ← 缺少 DWI1000
nifti_paths = [
    "/path/ADC.nii.gz",
    "/path/DWI0.nii.gz"
]
dicom_paths = [
    "/rename_dicom/.../ADC",
    "/rename_dicom/.../DWI0"
]
```

**验证结果**: ❌ **FAIL**
- 缺少 DWI1000 → missing target series

**生成的命令** (Fallback):
```bash
bash pipeline_infarct.sh \
  study_id \
  ADC.nii.gz DWI0.nii.gz \                     # 2 NIfTI
  /rename_dicom/.../ADC \                      # 单个 DICOM dir (第一个) ✅
  output
```

**日志**:
```
[WARNING] Infarct validation failed: missing target series 'DWI1000'
[WARNING] Infarct validation failed: falling back to single DICOM directory
```

---

### Scenario 3: DICOM 路径为空 (Fallback)

**输入**:
```python
series_uids = ["series_ADC", "series_DWI0", "series_DWI1000"]
nifti_paths = [
    "/path/ADC.nii.gz",
    "/path/DWI0.nii.gz",
    "/path/DWI1000.nii.gz"
]
dicom_paths = [
    "/rename_dicom/.../ADC",
    "",                        # ← 空路径
    "/rename_dicom/.../DWI1000"
]
```

**验证结果**: ❌ **FAIL**
- DWI0 的 DICOM 路径为空

**生成的命令** (Fallback):
```bash
bash pipeline_infarct.sh \
  study_id \
  ADC.nii.gz DWI0.nii.gz DWI1000.nii.gz \      # 3 NIfTI
  /rename_dicom/.../ADC \                      # 单个 DICOM dir (第一个有效) ✅
  output
```

**日志**:
```
[WARNING] Infarct validation failed: empty DICOM path at index 1
[WARNING] Infarct validation failed: falling back to single DICOM directory
```

---

### Scenario 4: 非标准命名 (模糊匹配通过)

**输入**:
```python
nifti_paths = [
    "/path/ADC_processed.nii.gz",      # ← 包含 "ADC"
    "/path/DWI0_resampled.nii.gz",     # ← 包含 "DWI0"
    "/path/DWI1000_corrected.nii.gz"   # ← 包含 "DWI1000"
]
dicom_paths = [
    "/rename_dicom/.../ADC_processed",
    "/rename_dicom/.../DWI0_resampled",
    "/rename_dicom/.../DWI1000_corrected"
]
```

**验证结果**: ✅ **PASS**
- 模糊匹配: `target.lower() in name.lower()`
- "adc" in "adc_processed" → 匹配成功

**生成的命令**:
```bash
bash pipeline_infarct.sh \
  study_id \
  ADC_processed.nii.gz DWI0_resampled.nii.gz DWI1000_corrected.nii.gz \
  /rename_dicom/.../ADC_processed \
  /rename_dicom/.../DWI0_resampled \
  /rename_dicom/.../DWI1000_corrected \
  output
```

---

## 与 CMB 的对比

| 模型 | 验证逻辑 | Fallback 机制 | 生成命令 |
|------|---------|--------------|---------|
| **Infarct** | ✅ 验证 target series | ✅ 失败 → 单个 DICOM | 验证通过: 多个 DICOM<br>验证失败: 单个 DICOM |
| **CMB** | ❌ 无验证 | ❌ 无需 fallback | 总是单个 DICOM |

**CMB 为什么不需要验证？**
- CMB 只生成一个合并的 DICOM-SEG 输出
- 只需要第一个有效的 DICOM 目录（通常是 SWAN）
- 无需验证特定的 series 组合

---

## 实现对比：Study Level vs Series Level

### Study Level

```python
def _resolve_infarct_dicom_inputs(task, ...):
    # 从 task.input_path_list 提取 candidate names
    candidate_names = [_extract_basename_from_path(path) for path in task.input_path_list]

    # 验证并构建 dicom_dirs
    for series in INFARCT_TARGET_SERIES:
        matched_name = _match_series_basename(candidate_names, series)
        if not matched_name:
            return default_dir, None  # Fallback
        dicom_path = _build_dicom_series_path(...)
        if not dicom_path.exists():
            return default_dir, None  # Fallback
        dicom_dirs.append(str(dicom_path))

    return default_dir, dicom_dirs  # 验证通过
```

### Series Level

```python
def _validate_infarct_target_series(nifti_paths, dicom_paths):
    # 从 nifti_paths 提取 series names
    series_names = [extract_basename_without_extension(path) for path in nifti_paths]

    # 验证所有 target series
    for target in INFARCT_TARGET_SERIES:
        found = any(target == name or target.lower() in name.lower() for name in series_names)
        if not found:
            return False  # 验证失败

    # 验证所有 DICOM 路径非空
    if any(not path for path in dicom_paths):
        return False

    return True  # 验证通过

# 使用
if is_infarct and _validate_infarct_target_series(...):
    # 传递多个
else:
    # Fallback 到单个
```

**差异**:
1. **Study Level**: 边验证边构建 dicom_dirs，失败立即返回
2. **Series Level**: 先验证，后决定使用多个还是单个
3. **逻辑一致性**: 两者验证条件完全相同

---

## 日志示例

### 成功场景

```
[INFO] Batch inputs model detected: <infarct_uuid>, processing 3 series as single inference
[INFO] Infarct validation passed: using 3 DICOM directories
[INFO] Executing batch inference: cd /mnt/e/pipeline/test/chuan/code && bash .../pipeline_infarct.sh ...
```

### Fallback 场景

```
[INFO] Batch inputs model detected: <infarct_uuid>, processing 2 series as single inference
[WARNING] Infarct validation failed: missing target series 'DWI1000'
[WARNING] Infarct validation failed: falling back to single DICOM directory
[INFO] Executing batch inference: cd /mnt/e/pipeline/test/chuan/code && bash .../pipeline_infarct.sh ...
```

---

## 向后兼容性

### 对现有 Infarct 推论的影响

**场景 1**: 标准 3-series Infarct 推论
- **Before**: 传递 3 个 DICOM 目录（假设没有验证）
- **After**: 验证通过 → 传递 3 个 DICOM 目录 ✅
- **影响**: 无变化

**场景 2**: 非标准 Infarct 推论（缺少某个 series）
- **Before**: 可能传递不完整的 DICOM 列表（错误行为）
- **After**: 验证失败 → fallback 到单个 DICOM 目录 ✅
- **影响**: 修正错误行为，与 Study Level 一致

### 对其他模型的影响

- **CMB**: 无影响（不触发 Infarct 验证）
- **Aneurysm**: 无影响（非 batch_inputs 模型）
- **WMH**: 无影响（非 batch_inputs 模型）

---

## 总结

### 关键改进

1. ✅ **添加 INFARCT_TARGET_SERIES 验证** - 确保包含必需的 series
2. ✅ **实现 Fallback 机制** - 验证失败时使用单个 DICOM 目录
3. ✅ **与 Study Level 逻辑一致** - 相同的验证条件和 fallback 行为
4. ✅ **不影响 CMB 和其他模型** - 验证仅针对 Infarct

### 验证决策流程

```
Batch Mode
    ↓
是否 Infarct 模型？
    ├─ NO → 使用单个 DICOM 目录 (CMB 等)
    └─ YES → 验证 target series
              ├─ 验证通过 → 使用多个 DICOM 目录
              └─ 验证失败 → Fallback 到单个 DICOM 目录
```

### 与用户提供代码的一致性

| 方面 | Study Level | Series Level |
|------|------------|-------------|
| Target series 定义 | `("ADC", "DWI0", "DWI1000")` | `("ADC", "DWI0", "DWI1000")` ✅ |
| 验证逻辑 | 检查所有 series 存在 + 路径有效 | 检查所有 series 存在 + 路径非空 ✅ |
| Fallback 行为 | 失败 → 返回单个 | 失败 → 使用单个 ✅ |
| 日志级别 | `logger.warning` | `logger.warning` ✅ |

**结论**: Series Level 实现完全遵循 Study Level 的设计哲学和行为模式 ✅
