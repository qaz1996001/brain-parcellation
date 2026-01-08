# Batch Inputs 模型对比：Infarct vs CMB

## 核心区别

虽然 **Infarct** 和 **CMB** 都是 `batch_inputs=True` 的模型，但它们对 DICOM 目录的需求**完全不同**。

| 特性 | Infarct | CMB |
|------|---------|-----|
| **batch_inputs** | ✅ True | ✅ True |
| **输入数量** | 3 个 series (ADC, DWI0, DWI1000) | 2 个 series (SWAN, T1BRAVO) |
| **DICOM 目录需求** | **每个 NIfTI 对应一个** | **只需要单个** |
| **命令参数** | `$2 $3 $4 $5 $6 $7` | `$2 $3 $4` |
| **参数含义** | 3 NIfTI + 3 DICOM + output | 2 NIfTI + 1 DICOM + output |

---

## Infarct 模型

### 配置
```python
InferenceEnum.Infarct: PipelineConfig(
    "pipeline_infarct.sh",
    "Infarct",
    batch_inputs=True  # ← 多输入批量模式
)
```

### 需求：每个 NIfTI 对应一个 DICOM 目录

**原因**: Infarct 模型需要为每个输入 series 生成独立的 DICOM-SEG 输出。

**Study Level 实现** (用户提供的代码):
```python
def _resolve_infarct_dicom_inputs(task, nifti_study_path, dicom_study_path, study_id):
    INFARCT_TARGET_SERIES = ("ADC", "DWI0", "DWI1000")

    dicom_dirs = []
    for series in INFARCT_TARGET_SERIES:
        matched_name = _match_series_basename(candidate_names, series)
        dicom_path = _build_dicom_series_path(dicom_study_path, nifti_study_path, matched_name)
        if dicom_path.exists():
            dicom_dirs.append(str(dicom_path))

    return default_dir, dicom_dirs  # 返回多个 DICOM 目录
```

### Series Level 实现

**Backend 提取**:
```python
# backend/app/inference/service.py
rename_dicom_paths = [
    "/mnt/e/rename_dicom/.../ADC",
    "/mnt/e/rename_dicom/.../DWI0",
    "/mnt/e/rename_dicom/.../DWI1000"
]
```

**Worker 处理**:
```python
# code_ai/task/task_pipeline.py
is_infarct = _is_infarct_model(model_id)

if is_infarct:
    # 传递所有对应的 DICOM 目录
    inference_cmd = _build_series_inference_cmd(
        nifti_paths=valid_nifti_paths,
        dicom_dirs=valid_dicom_dirs,  # ← 3 个 DICOM 目录
        ...
    )
```

### 生成的命令

```bash
cd /mnt/e/pipeline/test/chuan/code && \
bash /mnt/e/pipeline/test/chuan/code/pipeline_infarct.sh \
10089413_20210201_MR_21002010079 \                                    # $1: study_id
/mnt/e/.../rename_nifti/.../ADC.nii.gz \                              # $2: NIfTI 1
/mnt/e/.../rename_nifti/.../DWI0.nii.gz \                             # $3: NIfTI 2
/mnt/e/.../rename_nifti/.../DWI1000.nii.gz \                          # $4: NIfTI 3
/mnt/e/.../rename_dicom/.../ADC \                                     # $5: DICOM 1 (对应 $2)
/mnt/e/.../rename_dicom/.../DWI0 \                                    # $6: DICOM 2 (对应 $3)
/mnt/e/.../rename_dicom/.../DWI1000 \                                 # $7: DICOM 3 (对应 $4)
/mnt/e/.../rename_nifti/10089413_20210201_MR_21002010079              # $8: Output folder
```

**关键**: NIfTI 和 DICOM 参数**一一对应**：
- `$2` (ADC.nii.gz) ↔ `$5` (ADC DICOM dir)
- `$3` (DWI0.nii.gz) ↔ `$6` (DWI0 DICOM dir)
- `$4` (DWI1000.nii.gz) ↔ `$7` (DWI1000 DICOM dir)

---

## CMB 模型

### 配置
```python
InferenceEnum.CMB: PipelineConfig(
    "pipeline_cmb_tensorflow.py",
    "CMB",
    batch_inputs=True  # ← 多输入批量模式
)
```

### 需求：只需要单个 DICOM 目录

**原因**: CMB 模型只生成一个合并的 DICOM-SEG 输出，不需要每个输入 series 的独立 DICOM。

**Study Level 实现** (用户提供的代码):
```python
def _resolve_dicom_inputs(key, task, ...):
    if key == InferenceEnum.Infarct:
        return _resolve_infarct_dicom_inputs(...)  # 多个 DICOM

    # CMB 和其他模型
    basename = _extract_basename_from_path(task.input_path_list[0])
    dicom_path = _build_dicom_series_path(...)
    return (str(dicom_path), None)  # ← 只返回单个！
    #                        ^^^^
    #                        input_dicom_dirs=None
```

### Series Level 实现

**Worker 处理**:
```python
# code_ai/task/task_pipeline.py
is_infarct = _is_infarct_model(model_id)

if is_infarct:
    # Infarct 逻辑...
else:
    # CMB: 只传入第一个有效的 DICOM 目录
    dicom_dir = None
    for path in valid_dicom_dirs:
        if path:
            dicom_dir = path  # ← 只取第一个
            break

    inference_cmd = _build_series_inference_cmd(
        nifti_paths=valid_nifti_paths,
        dicom_dir=dicom_dir,  # ← 单个 DICOM 目录
        ...
    )
```

### 生成的命令

```bash
export PYTHONPATH=/mnt/d/00_Chen/Task04_git_test && \
/opt/miniconda3/envs/tf_2_14/bin/python3 code_ai/pipeline/pipeline_cmb_tensorflow.py \
--ID <study_id> \                                                     # study_id
--Inputs \
  /mnt/e/.../SWAN.nii.gz \                                            # NIfTI 1
  /mnt/e/.../T1BRAVO.nii.gz \                                         # NIfTI 2
--Output_folder /mnt/e/.../rename_nifti \                             # Output folder
--InputsDicomDir /mnt/e/.../rename_dicom/.../SWAN                     # 单个 DICOM 目录
```

**关键**: 只传递**第一个有效的 DICOM 目录**（通常是 SWAN 的 DICOM）。

---

## 为什么 CMB 不需要多个 DICOM 目录？

### DICOM-SEG 生成逻辑差异

**Infarct**:
- 为 ADC 生成 `Pred_Infarct_ADC.dcm`
- 为 DWI0 生成 `Pred_Infarct_DWI0.dcm`
- 为 DWI1000 生成 `Pred_Infarct_DWI1000.dcm`
- **需要**: 每个输入 series 的完整 DICOM metadata (IOP, IPP, UIDs)

**CMB**:
- 只生成一个合并的 `Pred_CMB.dcm`（基于 SWAN 的 metadata）
- 检测结果标注在 SWAN 图像的 3D 空间中
- **需要**: 只需要 SWAN（或第一个输入）的 DICOM metadata

### 输出文件对比

**Infarct** (`generate_output_files`):
```python
files = [
    "Pred_Infarct.nii.gz",          # 合并结果
    "Pred_Infarct_ADCth.nii.gz",    # ADC 特定
    "Pred_Infarct_synthseg.nii.gz", # SynthSeg
    "Pred_Infarct.json"             # JSON 结果
]
```
→ 需要多个 series 的 DICOM 用于生成不同输出

**CMB** (`generate_output_files`):
```python
files = [
    "synthseg_SWAN_original_CMB_from_synthseg_T1BRAVO_original_CMB.nii.gz",
    "Pred_CMB.nii.gz",              # 单一合并结果
    "Pred_CMB.json"                 # JSON 结果
]
```
→ 只需要 SWAN 的 DICOM metadata

---

## 代码实现对比

### `_is_infarct_model()` 检查

```python
def _is_infarct_model(model_id: str) -> bool:
    """
    檢查模型是否為 Infarct 模型。

    Infarct 模型需要為每個 NIfTI 輸入傳遞對應的 DICOM 目錄。
    其他 batch_inputs 模型（如 CMB）只需要單個 DICOM 目錄。
    """
    try:
        inference_enum = _resolve_model_id_to_inference_enum(model_id)
        return inference_enum == InferenceEnum.Infarct
    except (ValueError, KeyError):
        return False
```

### Batch Mode 分支逻辑

```python
if is_batch_model:
    # 收集所有 DICOM 目录
    valid_dicom_dirs = []
    for series_uid, nifti_path, dicom_path in zip(...):
        valid_dicom_dirs.append(dicom_path if dicom_path else "")

    # 根据模型类型决定传递方式
    is_infarct = _is_infarct_model(model_id)

    if is_infarct:
        # Infarct: 传递所有 DICOM 目录
        inference_cmd = _build_series_inference_cmd(
            ...,
            dicom_dirs=valid_dicom_dirs  # ← 多个
        )
    else:
        # CMB: 只传递第一个有效 DICOM 目录
        dicom_dir = next((path for path in valid_dicom_dirs if path), None)
        inference_cmd = _build_series_inference_cmd(
            ...,
            dicom_dir=dicom_dir  # ← 单个
        )
```

### `PipelineConfig.generate_cmd()` 调用

**Infarct** (via `_build_chuan_command`):
```python
cmd_str = pipeline.generate_cmd(
    study_id=study_id,
    task=task,
    input_dicom_dir=None,           # ← 不使用单个
    input_dicom_dirs=dicom_dirs,    # ← 使用多个
    path_root=path_root
)

# _collect_dicom_args() 返回:
# ["/path/ADC", "/path/DWI0", "/path/DWI1000"]

# _build_chuan_command() 生成:
# bash script.sh study_id nifti1 nifti2 nifti3 dicom1 dicom2 dicom3 output
```

**CMB** (via `_build_python_command`):
```python
cmd_str = pipeline.generate_cmd(
    study_id=study_id,
    task=task,
    input_dicom_dir=dicom_dir,      # ← 使用单个
    input_dicom_dirs=None,          # ← 不使用多个
    path_root=path_root
)

# _collect_dicom_args() 返回:
# ["/path/SWAN"]

# _build_python_command() 生成:
# python script.py --ID study_id --Inputs nifti1 nifti2 --InputsDicomDir dicom1 --Output_folder output
```

---

## 向后兼容性

### Study Level (不受影响)

Study Level 使用 `build_inference_cmd()` (在 `code_ai/utils/inference/__init__.py`)，其逻辑保持不变：

```python
def build_inference_cmd(nifti_study_path, dicom_study_path, config_path):
    for key, value in analysis.model_dump().items():
        # Infarct: 调用 _resolve_infarct_dicom_inputs() → 返回多个
        # CMB: 调用通用逻辑 → 返回单个
        input_dicom_dir, input_dicom_dirs = _resolve_dicom_inputs(...)

        cmd_str = pipeline.generate_cmd(
            study_id,
            task,
            input_dicom_dir=input_dicom_dir,
            input_dicom_dirs=input_dicom_dirs
        )
```

### Series Level (新实现)

Series Level 在 Batch Mode 中添加了模型检查：

```python
if is_batch_model:
    is_infarct = _is_infarct_model(model_id)

    if is_infarct:
        # 新逻辑: 传递多个 DICOM 目录
    else:
        # 保持 CMB 原有行为: 传递单个 DICOM 目录
```

---

## 测试场景

### Scenario 1: Infarct 推论

**输入**:
```json
{
  "series_uids": ["series_ADC", "series_DWI0", "series_DWI1000"],
  "model_id": "<infarct_uuid>"
}
```

**期望命令**:
```bash
bash pipeline_infarct.sh \
  study_id \
  ADC.nii.gz DWI0.nii.gz DWI1000.nii.gz \      # 3 NIfTI
  /rename_dicom/.../ADC \                      # 3 DICOM dirs
  /rename_dicom/.../DWI0 \
  /rename_dicom/.../DWI1000 \
  output_folder
```

**验证**:
- ✅ 3 个 NIfTI 输入
- ✅ 3 个对应的 DICOM 目录
- ✅ 位置一一对应

### Scenario 2: CMB 推论

**输入**:
```json
{
  "series_uids": ["series_SWAN", "series_T1BRAVO"],
  "model_id": "<cmb_uuid>"
}
```

**期望命令**:
```bash
python pipeline_cmb_tensorflow.py \
  --ID study_id \
  --Inputs SWAN.nii.gz T1BRAVO.nii.gz \        # 2 NIfTI
  --InputsDicomDir /rename_dicom/.../SWAN \    # 单个 DICOM dir
  --Output_folder output_folder
```

**验证**:
- ✅ 2 个 NIfTI 输入
- ✅ 只有 1 个 DICOM 目录（第一个有效路径）
- ✅ 与 Study Level 行为一致

---

## 总结

| 方面 | Infarct | CMB |
|------|---------|-----|
| **batch_inputs** | ✅ True | ✅ True |
| **脚本类型** | Bash (chuan) | Python |
| **DICOM 需求** | 每个 NIfTI 对应一个 | 只需要第一个 |
| **参数传递** | `input_dicom_dirs=[...]` | `input_dicom_dir="..."` |
| **命令生成器** | `_build_chuan_command` | `_build_python_command` |
| **DICOM-SEG 输出** | 每个 series 独立 | 单一合并输出 |
| **Series Level 检查** | `_is_infarct_model()` | 默认分支 |

**关键设计原则**:
1. **不修改 CMB 和 Aneurysm 的行为** - 通过 `is_infarct` 检查分支
2. **Infarct 特殊处理** - 只对 Infarct 传递多个 DICOM 目录
3. **向后兼容** - Study Level 和 Series Level 保持一致的行为
4. **遵循 Linus 哲学** - 数据结构驱动行为，消除特殊情况
