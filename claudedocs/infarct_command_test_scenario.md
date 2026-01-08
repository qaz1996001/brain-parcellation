# Infarct 命令生成测试场景

## 测试场景：3 Series Infarct 检测

### 输入数据

**API Request**:
```json
POST /api/inference/series/queue
{
  "series_uids": [
    "1.2.3.4.5.1",  // ADC
    "1.2.3.4.5.2",  // DWI0
    "1.2.3.4.5.3"   // DWI1000
  ],
  "model_id": "<infarct_model_uuid>"
}
```

### Backend 处理

**validate_series_ready() 提取路径**:
```python
# 从 SERIES_CONVERSION_COMPLETE 事件提取
nifti_paths = [
    "/mnt/e/pipeline/test/sean/rename_nifti/10089413_20210201_MR_21002010079/ADC.nii.gz",
    "/mnt/e/pipeline/test/sean/rename_nifti/10089413_20210201_MR_21002010079/DWI0.nii.gz",
    "/mnt/e/pipeline/test/sean/rename_nifti/10089413_20210201_MR_21002010079/DWI1000.nii.gz"
]

rename_dicom_paths = [
    "/mnt/e/pipeline/test/sean/rename_dicom/10089413_20210201_MR_21002010079/ADC",
    "/mnt/e/pipeline/test/sean/rename_dicom/10089413_20210201_MR_21002010079/DWI0",
    "/mnt/e/pipeline/test/sean/rename_dicom/10089413_20210201_MR_21002010079/DWI1000"
]
```

**func_params**:
```python
{
    "series_uids": ["1.2.3.4.5.1", "1.2.3.4.5.2", "1.2.3.4.5.3"],
    "model_id": "<infarct_model_uuid>",
    "study_uid": "1.2.3.4.5",
    "study_id": "10089413_20210201_MR_21002010079",
    "inference_id": "inf_abc123",
    "nifti_series_paths": [...],       # Direct Mode 提供
    "rename_dicom_paths": [...],       # NEW! 对应每个 series
    "path_params": {
        "path_root": "/mnt/e/pipeline/test/sean",
        "path_rename_nifti": "/mnt/e/pipeline/test/sean/rename_nifti",
        "path_json": "/mnt/e/pipeline/test/sean/path_json",
        ...
    }
}
```

### Worker 处理 (Batch Mode)

**Step 1: 检测为 batch_inputs 模型**:
```python
is_batch_model = _is_batch_inputs_model(model_id)  # True for Infarct
```

**Step 2: 收集有效路径**:
```python
valid_nifti_paths = []
valid_dicom_dirs = []

for series_uid, nifti_path, dicom_path in zip(
    series_uids, nifti_paths, dicom_series_paths
):
    if nifti_path and os.path.exists(nifti_path):
        valid_nifti_paths.append(nifti_path)
        valid_dicom_dirs.append(dicom_path if dicom_path else "")

# 结果:
valid_nifti_paths = [
    "/mnt/e/.../ADC.nii.gz",
    "/mnt/e/.../DWI0.nii.gz",
    "/mnt/e/.../DWI1000.nii.gz"
]

valid_dicom_dirs = [
    "/mnt/e/.../rename_dicom/.../ADC",
    "/mnt/e/.../rename_dicom/.../DWI0",
    "/mnt/e/.../rename_dicom/.../DWI1000"
]
```

**Step 3: 调用 _build_series_inference_cmd**:
```python
inference_cmd = _build_series_inference_cmd(
    nifti_paths=valid_nifti_paths,      # 3 个路径
    model_id=model_id,
    output_dir="/mnt/e/.../path_json/inf_abc123",
    dicom_dirs=valid_dicom_dirs,        # 3 个 DICOM 目录
    study_id="10089413_20210201_MR_21002010079",
    path_root="/mnt/e/pipeline/test/sean"
)
```

### 命令生成流程

**Step 1: PipelineConfig 查找**:
```python
pipeline_config = pipelines[InferenceEnum.Infarct]
# PipelineConfig("pipeline_infarct.sh", "Infarct", batch_inputs=True)
```

**Step 2: 路径排序** (batch_inputs=True):
```python
# check_study_mapping_inference 确保顺序与 config.yaml 一致
sorted_nifti_paths = [ADC.nii.gz, DWI0.nii.gz, DWI1000.nii.gz]
```

**Step 3: 生成 Task 对象**:
```python
task = Task(
    intput_path_list=sorted_nifti_paths,
    output_path="/mnt/e/.../rename_nifti/10089413_20210201_MR_21002010079",
    output_path_list=[...]
)
```

**Step 4: 调用 generate_cmd**:
```python
cmd_str = pipeline_config.generate_cmd(
    study_id="10089413_20210201_MR_21002010079",
    task=task,
    input_dicom_dir=None,               # 不使用单个
    input_dicom_dirs=valid_dicom_dirs,  # 使用多个
    path_root="/mnt/e/pipeline/test/sean"
)
```

**Step 5: _collect_dicom_args**:
```python
def _collect_dicom_args(input_dicom_dir, input_dicom_dirs):
    if input_dicom_dirs:
        return [dicom_dir for dicom_dir in input_dicom_dirs if dicom_dir]
    if input_dicom_dir:
        return [input_dicom_dir]
    return []

# 执行:
dicom_args = [
    "/mnt/e/.../rename_dicom/.../ADC",
    "/mnt/e/.../rename_dicom/.../DWI0",
    "/mnt/e/.../rename_dicom/.../DWI1000"
]
```

**Step 6: _resolve_command_builder**:
```python
# "Infarct" in chuan_root_data_key? YES
builder = self._build_chuan_command
```

**Step 7: _build_chuan_command**:
```python
# 计算 chuan_code 路径
PATH_ROOT = pathlib.Path("/mnt/e/pipeline/test/sean")
chuan_root = PATH_ROOT.parent / "chuan"  # /mnt/e/pipeline/test/chuan
chuan_code = chuan_root / "code"          # /mnt/e/pipeline/test/chuan/code

# 构建命令
command_parts = [
    f"cd {chuan_code}",                         # cd /mnt/e/pipeline/test/chuan/code
    "&&",
    f"bash {chuan_code}/pipeline_infarct.sh",   # bash .../pipeline_infarct.sh
    "10089413_20210201_MR_21002010079",         # $1: study_id
    "/mnt/e/.../ADC.nii.gz",                    # $2: NIfTI 1
    "/mnt/e/.../DWI0.nii.gz",                   # $3: NIfTI 2
    "/mnt/e/.../DWI1000.nii.gz",                # $4: NIfTI 3
    "/mnt/e/.../rename_dicom/.../ADC",          # $5: DICOM 1
    "/mnt/e/.../rename_dicom/.../DWI0",         # $6: DICOM 2
    "/mnt/e/.../rename_dicom/.../DWI1000",      # $7: DICOM 3
    "/mnt/e/.../rename_nifti/10089413_..."      # $8: Output folder
]

return " ".join(command_parts)
```

### 期望输出命令

```bash
cd /mnt/e/pipeline/test/chuan/code && \
bash /mnt/e/pipeline/test/chuan/code/pipeline_infarct.sh \
10089413_20210201_MR_21002010079 \
/mnt/e/pipeline/test/sean/rename_nifti/10089413_20210201_MR_21002010079/ADC.nii.gz \
/mnt/e/pipeline/test/sean/rename_nifti/10089413_20210201_MR_21002010079/DWI0.nii.gz \
/mnt/e/pipeline/test/sean/rename_nifti/10089413_20210201_MR_21002010079/DWI1000.nii.gz \
/mnt/e/pipeline/test/sean/rename_dicom/10089413_20210201_MR_21002010079/ADC \
/mnt/e/pipeline/test/sean/rename_dicom/10089413_20210201_MR_21002010079/DWI0 \
/mnt/e/pipeline/test/sean/rename_dicom/10089413_20210201_MR_21002010079/DWI1000 \
/mnt/e/pipeline/test/sean/rename_nifti/10089413_20210201_MR_21002010079
```

### 参数对应关系

| 参数位置 | 值 | 说明 |
|---------|---|------|
| $1 | 10089413_20210201_MR_21002010079 | study_id |
| $2 | .../ADC.nii.gz | NIfTI Input 1 |
| $3 | .../DWI0.nii.gz | NIfTI Input 2 |
| $4 | .../DWI1000.nii.gz | NIfTI Input 3 |
| $5 | .../rename_dicom/.../ADC | DICOM Dir 1 (对应 $2) |
| $6 | .../rename_dicom/.../DWI0 | DICOM Dir 2 (对应 $3) |
| $7 | .../rename_dicom/.../DWI1000 | DICOM Dir 3 (对应 $4) |
| $8 | .../rename_nifti/... | Output folder |

### 关键改进点

#### Before (错误实现)
```python
# 只传递第一个 DICOM 路径
dicom_dir = dicom_series_paths[0]

inference_cmd = _build_series_inference_cmd(
    nifti_paths=valid_nifti_paths,  # 3 个
    dicom_dir=dicom_dir,            # 只有 1 个！
    ...
)

# 生成的命令（错误）:
bash pipeline_infarct.sh study_id \
  ADC.nii.gz DWI0.nii.gz DWI1000.nii.gz \
  /rename_dicom/.../ADC \              # 只有 1 个 DICOM 路径！
  /rename_nifti/...
```

#### After (正确实现)
```python
# 传递所有 DICOM 路径
valid_dicom_dirs = [path1, path2, path3]

inference_cmd = _build_series_inference_cmd(
    nifti_paths=valid_nifti_paths,   # 3 个
    dicom_dirs=valid_dicom_dirs,     # 3 个对应！
    ...
)

# 生成的命令（正确）:
bash pipeline_infarct.sh study_id \
  ADC.nii.gz DWI0.nii.gz DWI1000.nii.gz \
  /rename_dicom/.../ADC \
  /rename_dicom/.../DWI0 \
  /rename_dicom/.../DWI1000 \         # 3 个 DICOM 路径！
  /rename_nifti/...
```

### 边缘情况处理

#### 情况 1: 某个 series 缺少 rename_dicom_path

**场景**: series_B 的 `_extract_rename_dicom_path()` 失败，返回空字符串

```python
rename_dicom_paths = [
    "/mnt/e/.../ADC",
    "",                    # ← 提取失败
    "/mnt/e/.../DWI1000"
]

# Worker 收集:
valid_dicom_dirs = [
    "/mnt/e/.../ADC",
    "",                    # ← 保留空字符串
    "/mnt/e/.../DWI1000"
]

# _collect_dicom_args 过滤空字符串:
dicom_args = [
    "/mnt/e/.../ADC",
    "/mnt/e/.../DWI1000"   # ← 空字符串被移除！
]

# 生成的命令:
bash pipeline_infarct.sh study_id \
  ADC.nii.gz DWI0.nii.gz DWI1000.nii.gz \  # 3 个 NIfTI
  /rename_dicom/.../ADC \                   # 只有 2 个 DICOM！
  /rename_dicom/.../DWI1000 \
  /rename_nifti/...

# ❌ 问题: 位置不对应！DWI0 没有对应的 DICOM 路径
```

**解决方案**:
1. **选项 A**: 修改 `_collect_dicom_args`，不过滤空字符串（保持位置对应）
2. **选项 B**: Backend 确保所有 series 都有 rename_dicom_path，否则拒绝推论
3. **选项 C**: Worker 在收集时，过滤掉没有 dicom_path 的 series（同时移除对应的 NIfTI）

**推荐**: **选项 B** - Backend 验证所有 series 都有 rename_dicom_path

#### 情况 2: Mixed Mode (部分 Direct + 部分 Convert)

**场景**: series_A Direct (已有 NIfTI), series_B 需要转换

```python
# Backend 提取:
nifti_series_paths = ["/path/ADC.nii.gz"]  # Direct
raw_dicom_series_paths = ["/path/raw_dicom/DWI0"]  # Convert
rename_dicom_paths = ["/path/rename_dicom/ADC"]

# Worker Mixed Mode:
# 1. 转换 series_B
dicom_paths_converted = _batch_convert_series_to_nifti([raw_path])
# Returns: (["/path/DWI0.nii.gz"], ["/path/rename_dicom/DWI0"])

# 2. 合并路径
nifti_paths = nifti_series_paths + nifti_paths_converted
# ["/path/ADC.nii.gz", "/path/DWI0.nii.gz"]

rename_dicom_paths_direct = func_params.get("rename_dicom_paths", [])
dicom_series_paths = rename_dicom_paths_direct + dicom_paths_converted[1]
# ["/path/rename_dicom/ADC", "/path/rename_dicom/DWI0"]

# ✅ 正确: 两个 NIfTI 对应两个 DICOM 目录
```

### 测试验证步骤

1. **提交 Infarct 推论请求**:
   ```bash
   curl -X POST "http://localhost:8000/api/inference/series/queue" \
     -H "Content-Type: application/json" \
     -d '{
       "series_uids": ["1.2.3.4.5.1", "1.2.3.4.5.2", "1.2.3.4.5.3"],
       "model_id": "<infarct_model_uuid>"
     }'
   ```

2. **检查 Worker 日志**:
   ```
   [INFO] Batch inputs model detected: <infarct_model_uuid>, processing 3 series
   [INFO] Executing batch inference: cd /mnt/e/pipeline/test/chuan/code && bash ...
   ```

3. **验证生成的命令**:
   - ✅ 3 个 NIfTI 输入路径
   - ✅ 3 个对应的 rename_dicom 路径
   - ✅ 路径顺序正确 (ADC, DWI0, DWI1000)
   - ✅ chuan_code 路径正确

4. **检查 DCOP Event**:
   ```json
   {
     "ope_no": "SERIES_INFERENCE_COMPLETE",
     "params_data": {
       "series_uids": ["1.2.3.4.5.1", "1.2.3.4.5.2", "1.2.3.4.5.3"],
       "batch_inference": true,
       "inference_id": "inf_abc123"
     },
     "result_data": {
       "inference_cmd": [{
         "cmd_str": "cd /mnt/e/pipeline/test/chuan/code && bash ..."
       }]
     }
   }
   ```

### 成功标准

- ✅ 命令包含 3 个 NIfTI 路径
- ✅ 命令包含 3 个 rename_dicom 路径
- ✅ 路径顺序对应 (NIfTI[i] ↔ DICOM[i])
- ✅ 使用 rename_dicom（非 raw_dicom）
- ✅ chuan_code 路径正确解析
- ✅ Output folder 使用 study_id (中间处理阶段)
