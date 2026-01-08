# Infarct 模型命令生成完整流程

## 模型特性

**Infarct 模型配置** (`code_ai/pipeline/__init__.py:242-244`):
```python
InferenceEnum.Infarct: PipelineConfig(
    "pipeline_infarct.sh",  # Chuan 项目的 bash 脚本
    "Infarct",              # data_key
    batch_inputs=True       # 多输入批量模式（类似 CMB）
)
```

**关键特性**:
- ✅ `batch_inputs=True`: 支持多个 series 作为单次推论的输入
- ✅ `chuan_root_data_key`: 使用 Chuan 项目的 bash 脚本（非 Python 脚本）
- ✅ Series Level 处理: 支持指定 series_uids 进行推论

---

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: API Request (Backend)                                      │
├─────────────────────────────────────────────────────────────────────┤
│ POST /api/inference/series/queue                                   │
│ {                                                                   │
│   "series_uids": ["series_A", "series_B"],  ← 多个 series           │
│   "model_id": "<infarct_model_uuid>"                                │
│ }                                                                   │
│                                                                     │
│ backend/app/inference/service.py:InferenceService                  │
│   └─ queue_series_inference()                                      │
│       ├─ validate_series_ready()  ← 验证 series 状态               │
│       │   ├─ Direct Mode: Extract from SERIES_CONVERSION_COMPLETE  │
│       │   │   ├─ nifti_paths: 从 result_data 提取                  │
│       │   │   └─ rename_dicom_paths: 从 params_data 提取           │
│       │   │       (_extract_rename_dicom_path)                     │
│       │   └─ Convert Mode: Extract from SERIES_TRANSFER_COMPLETE   │
│       │       └─ raw_dicom_paths: 从 result_data 提取              │
│       │                                                             │
│       └─ Build func_params:                                         │
│           {                                                         │
│             "series_uids": ["series_A", "series_B"],                │
│             "model_id": "<infarct_model_uuid>",                     │
│             "study_uid": "...",                                     │
│             "nifti_series_paths": [...],    ← Direct Mode          │
│             "raw_dicom_series_paths": [...], ← Convert Mode        │
│             "rename_dicom_paths": [...],    ← NEW! (rename_dicom)  │
│             "path_params": {                                        │
│               "path_root": "/mnt/e/pipeline/sean",                  │
│               "path_raw_dicom": "/mnt/e/raw_dicom",                 │
│               "path_rename_dicom": "/mnt/e/rename_dicom",           │
│               "path_rename_nifti": "/mnt/e/rename_nifti",           │
│               "path_json": "/mnt/e/path_json",                      │
│               ...                                                   │
│             }                                                       │
│           }                                                         │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Task Queue (RabbitMQ)                                      │
├─────────────────────────────────────────────────────────────────────┤
│ from code_ai.task.task_pipeline import task_pipeline_inference     │
│ task_pipeline_inference.push(func_params)                          │
│                                                                     │
│ Queue: task_pipeline_inference (qps=1)                             │
│ ├─ GPU mutual exclusion via funboost qps=1                         │
│ └─ FIFO scheduling for all inference tasks                         │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Worker Receives Task                                       │
├─────────────────────────────────────────────────────────────────────┤
│ code_ai/task/task_pipeline.py:                                     │
│ @boost(..., qps=1)                                                  │
│ def task_pipeline_inference(func_params: Dict):                    │
│     # Data structure determines behavior (Linus philosophy)        │
│     if 'series_uids' in func_params:                                │
│         return _task_series_pipeline_inference(func_params)         │
│     else:                                                           │
│         return _task_study_pipeline_inference(func_params)          │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Series Level Inference Handler                             │
├─────────────────────────────────────────────────────────────────────┤
│ _task_series_pipeline_inference(func_params)                       │
│                                                                     │
│ 4.1 Extract Parameters:                                            │
│     series_uids = ["series_A", "series_B"]                          │
│     model_id = "<infarct_model_uuid>"                               │
│     study_uid, study_id, inference_id                               │
│     path_root, path_rename_nifti, path_json, path_process          │
│                                                                     │
│ 4.2 Check Model Type:                                              │
│     is_batch_model = _is_batch_inputs_model(model_id)               │
│     # Returns True for Infarct                                     │
│     # Looks up: pipelines[InferenceEnum.Infarct].batch_inputs      │
│                                                                     │
│ 4.3 Determine Mode (Mixed/Direct/Convert):                         │
│     nifti_series_paths = func_params.get("nifti_series_paths")     │
│     raw_dicom_series_paths = func_params.get("raw_dicom_series_paths")│
│                                                                     │
│     if nifti_series_paths and raw_dicom_series_paths:               │
│         mode = "Mixed Mode"  ← Some Direct + Some Convert          │
│     elif nifti_series_paths:                                        │
│         mode = "Direct Mode" ← All NIfTI ready                     │
│     else:                                                           │
│         mode = "Convert Mode" ← Need conversion                    │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: Batch Inputs Mode Processing (is_batch_model=True)         │
├─────────────────────────────────────────────────────────────────────┤
│ Line 947-1056 in task_pipeline.py                                  │
│                                                                     │
│ 5.1 Validate All NIfTI Files:                                      │
│     valid_nifti_paths = []                                          │
│     missing_series = []                                             │
│     for series_uid, nifti_path in zip(series_uids, nifti_paths):   │
│         if nifti_path and os.path.exists(nifti_path):               │
│             valid_nifti_paths.append(nifti_path)                    │
│         else:                                                       │
│             missing_series.append(series_uid)                       │
│                                                                     │
│ 5.2 Create Output Directory:                                       │
│     # Use inference_id (not series_uid) for batch inference        │
│     output_dir = os.path.join(path_json, inference_id)              │
│     os.makedirs(output_dir, exist_ok=True)                          │
│     # Example: /mnt/e/path_json/<inference_id>/                    │
│                                                                     │
│ 5.3 Find First Valid DICOM Path:                                   │
│     # Linus: "Don't use None when you have valid data"             │
│     dicom_dir = None                                                │
│     for path in dicom_series_paths:                                 │
│         if path is not None:                                        │
│             dicom_dir = path  ← First non-None rename_dicom path   │
│             break                                                   │
│     # Example: /mnt/e/rename_dicom/<study_id>/<series_A>           │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 6: Build Inference Command                                    │
├─────────────────────────────────────────────────────────────────────┤
│ inference_cmd = _build_series_inference_cmd(                       │
│     nifti_paths=valid_nifti_paths,  ← All valid NIfTI paths        │
│     model_id=model_id,                                              │
│     output_dir=output_dir,          ← /path_json/<inference_id>    │
│     dicom_dir=dicom_dir,            ← First valid rename_dicom     │
│     study_id=study_id,                                              │
│     path_root=path_root             ← For Chuan path resolution    │
│ )                                                                   │
│                                                                     │
│ Line 638-747 in task_pipeline.py                                   │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 7: _build_series_inference_cmd() Implementation               │
├─────────────────────────────────────────────────────────────────────┤
│ 7.1 Resolve Model → InferenceEnum:                                 │
│     inference_enum = _resolve_model_id_to_inference_enum(model_id)  │
│     # Returns: InferenceEnum.Infarct                                │
│                                                                     │
│ 7.2 Get PipelineConfig:                                            │
│     from code_ai.pipeline import pipelines                          │
│     pipeline_config = pipelines[InferenceEnum.Infarct]              │
│     # PipelineConfig("pipeline_infarct.sh", "Infarct", batch_inputs=True)│
│                                                                     │
│ 7.3 Resolve study_id:                                              │
│     # Extract study_id from first NIfTI path if not provided       │
│     if not study_id:                                                │
│         study_path = Path(nifti_paths[0]).parent                    │
│         resolved_study_id = study_path.name                         │
│                                                                     │
│ 7.4 Sort NIfTI Paths (if batch_inputs=True):                       │
│     if pipeline_config.batch_inputs and len(nifti_paths) > 1:      │
│         # Use check_study_mapping_inference for correct order      │
│         mapping_result = check_study_mapping_inference(study_path)  │
│         # Ensures input order matches config.yaml definition       │
│         sorted_nifti_paths = [ordered paths from mapping]           │
│                                                                     │
│ 7.5 Generate Output Files:                                         │
│     task_output_files = generate_output_files(                     │
│         sorted_nifti_paths,                                         │
│         inference_enum.value,  # "Infarct"                          │
│         str(study_path)        # Base output path                  │
│     )                                                               │
│                                                                     │
│ 7.6 Create Task Object:                                            │
│     task = Task(                                                    │
│         intput_path_list=sorted_nifti_paths,                        │
│         output_path=str(study_path),                                │
│         output_path_list=task_output_files                          │
│     )                                                               │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 8: PipelineConfig.generate_cmd()                              │
├─────────────────────────────────────────────────────────────────────┤
│ cmd_str = pipeline_config.generate_cmd(                            │
│     study_id=resolved_study_id,                                     │
│     task=task,                                                      │
│     input_dicom_dir=dicom_dir,  ← rename_dicom path                │
│     path_root=path_root         ← For dual deployment support      │
│ )                                                                   │
│                                                                     │
│ code_ai/pipeline/__init__.py:145-156                               │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 9: Command Builder Resolution                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 9.1 Extract Input Paths:                                           │
│     input_paths = [str(path) for path in task.input_path_list]     │
│     # ["/.../study_id/series_A.nii.gz", "/.../series_B.nii.gz"]   │
│                                                                     │
│ 9.2 Collect DICOM Args:                                            │
│     dicom_args = self._collect_dicom_args(                          │
│         input_dicom_dir=dicom_dir,  # Single path                   │
│         input_dicom_dirs=None       # List of paths                 │
│     )                                                               │
│     # Returns: ["/mnt/e/rename_dicom/<study_id>/<series_A>"]       │
│                                                                     │
│ 9.3 Resolve Command Builder:                                       │
│     builder = self._resolve_command_builder()                       │
│     # Check: "Infarct" in chuan_root_data_key?                     │
│     # YES → return self._build_chuan_command                        │
│                                                                     │
│ 9.4 Build Command:                                                  │
│     return builder(study_id, task, input_paths, dicom_args, path_root)│
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 10: _build_chuan_command() Execution                          │
├─────────────────────────────────────────────────────────────────────┤
│ code_ai/pipeline/__init__.py:176-203                               │
│                                                                     │
│ 10.1 Resolve PATH_ROOT (Dual Deployment Support):                  │
│      if path_root is not None:                                     │
│          PATH_ROOT = pathlib.Path(path_root)                        │
│      else:                                                          │
│          PATH_ROOT = pathlib.Path(os.getenv("PATH_ROOT") or "")     │
│      # Example: /mnt/e/pipeline/sean                                │
│                                                                     │
│ 10.2 Calculate Chuan Paths:                                        │
│      chuan_root = PATH_ROOT.parent.joinpath("chuan")                │
│      # /mnt/e/pipeline/chuan                                        │
│                                                                     │
│      chuan_code = chuan_root.joinpath("code")                       │
│      # /mnt/e/pipeline/chuan/code                                   │
│                                                                     │
│ 10.3 Build Command Parts:                                          │
│      command_parts = [                                              │
│          f"cd {chuan_code}",                                        │
│          "&&",                                                      │
│          f"bash {chuan_code}/{self.script_name}",  ← pipeline_infarct.sh│
│          study_id,                                 ← Study ID       │
│          *input_paths,                             ← NIfTI paths    │
│          *dicom_args,                              ← DICOM path     │
│          task.output_path,                         ← Output dir     │
│      ]                                                              │
│                                                                     │
│ 10.4 Join Command:                                                  │
│      return " ".join(command_parts)                                 │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 11: Final Command String                                      │
├─────────────────────────────────────────────────────────────────────┤
│ cd /mnt/e/pipeline/chuan/code && \                                  │
│ bash /mnt/e/pipeline/chuan/code/pipeline_infarct.sh \               │
│ <study_id> \                                                        │
│ /mnt/e/rename_nifti/<study_id>/series_A.nii.gz \                   │
│ /mnt/e/rename_nifti/<study_id>/series_B.nii.gz \                   │
│ /mnt/e/rename_dicom/<study_id>/<series_A> \                        │
│ /mnt/e/rename_nifti/<study_id>                                      │
│                                                                     │
│ ┌─────────────────────────────────────────────┐                    │
│ │ Command Arguments (positional):             │                    │
│ ├─────────────────────────────────────────────┤                    │
│ │ $1: <study_id>                              │                    │
│ │ $2: /path/series_A.nii.gz                   │                    │
│ │ $3: /path/series_B.nii.gz                   │                    │
│ │ $4: /path/rename_dicom/<study_id>/<series>  │ ← --InputsDicomDir│
│ │ $5: /path/rename_nifti/<study_id>           │ ← Output folder   │
│ └─────────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 12: Create InferenceCmdItem                                   │
├─────────────────────────────────────────────────────────────────────┤
│ inference_item = InferenceCmdItem(                                  │
│     study_id=resolved_study_id,                                     │
│     name=inference_enum,           # InferenceEnum.Infarct          │
│     cmd_str=cmd_str,               # Full bash command              │
│     input_list=sorted_nifti_paths, # All NIfTI inputs               │
│     output_list=task.output_path_list,                              │
│     input_dicom_dir=dicom_dir or "",                                │
│ )                                                                   │
│                                                                     │
│ return InferenceCmd(cmd_items=[inference_item])                    │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 13: Execute Command (Batch Mode)                              │
├─────────────────────────────────────────────────────────────────────┤
│ Line 1008-1056 in task_pipeline.py                                 │
│                                                                     │
│ process = subprocess.Popen(                                         │
│     args=cmd_str,                                                   │
│     shell=True,                                                     │
│     stdout=subprocess.PIPE,                                         │
│     stderr=subprocess.PIPE,                                         │
│     cwd=path_process                                                │
│ )                                                                   │
│ stdout, stderr = process.communicate(timeout=600)                   │
│                                                                     │
│ if process.returncode == 0:                                         │
│     # Success: Parse prediction.json                                │
│     prediction_file = os.path.join(output_dir, "prediction.json")   │
│     # All series in batch share same result                         │
│     for series_uid in series_uids:                                  │
│         result_list.append({                                        │
│             "series_uid": series_uid,                               │
│             "status": "success",                                    │
│             "prediction": prediction,                               │
│             "output_dir": output_dir,  ← inference_id based         │
│             "batch_inference": True                                 │
│         })                                                          │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 14: Send DCOP Event                                           │
├─────────────────────────────────────────────────────────────────────┤
│ Line 1210-1241 in task_pipeline.py                                 │
│                                                                     │
│ if all_success:                                                     │
│     ope_no = DCOPStatus.SERIES_INFERENCE_COMPLETE.value             │
│ else:                                                               │
│     ope_no = DCOPStatus.SERIES_INFERENCE_FAILED.value               │
│                                                                     │
│ dcop_event = DCOPEventRequest(                                      │
│     study_uid=study_uid,                                            │
│     series_uid=None,  # Batch mode: no single series               │
│     study_id=study_id,                                              │
│     ope_no=ope_no,                                                  │
│     tool_id="SERIES_INFERENCE_TOOL",                                │
│     params_data={                                                   │
│         "series_uids": series_uids,                                 │
│         "model_id": model_id,                                       │
│         "batch_inference": True,                                    │
│         "inference_id": inference_id                                │
│     },                                                              │
│     result_data={                                                   │
│         "results": result_list,                                     │
│         "inference_cmd": all_inference_cmd_items,                   │
│         "output_dir": output_dir                                    │
│     }                                                               │
│ )                                                                   │
│ requests.post(UPLOAD_DCOP_EVENT_API_URL, json=dcop_event.dict())   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 关键路径数据流

### Input Paths (NIfTI)
```python
# Step 1: Backend extracts from DCOP events
nifti_paths = [
    "/mnt/e/rename_nifti/<study_id>/series_A.nii.gz",
    "/mnt/e/rename_nifti/<study_id>/series_B.nii.gz"
]

# Step 7.4: Worker sorts paths (if batch_inputs=True)
sorted_nifti_paths = check_study_mapping_inference(study_path)
# Ensures: SWAN before T1BRAVO, etc.

# Step 10.3: Passed to bash script as $2, $3, ...
```

### DICOM Path (rename_dicom)
```python
# Step 1: Backend extracts from SERIES_CONVERSION_COMPLETE event
rename_dicom_paths = [
    "/mnt/e/rename_dicom/<study_id>/<series_A>",
    "/mnt/e/rename_dicom/<study_id>/<series_B>"
]

# Step 5.3: Worker selects first non-None
dicom_dir = "/mnt/e/rename_dicom/<study_id>/<series_A>"

# Step 10.3: Passed to bash script as $4 (--InputsDicomDir)
```

### Output Path
```python
# Step 5.2: Batch mode uses inference_id
output_dir = "/mnt/e/path_json/<inference_id>"

# Step 10.3: Passed to bash script as $5
task.output_path = "/mnt/e/rename_nifti/<study_id>"
```

---

## 与其他模型的差异

### vs. Aneurysm (Chuan, Non-Batch)
| 特性 | Infarct | Aneurysm |
|------|---------|----------|
| batch_inputs | ✅ True | ❌ False |
| 输入数量 | 多个 series | 单个 series |
| 命令生成 | _build_chuan_command | _build_chuan_command |
| Output Dir | inference_id | study_id |

### vs. CMB (Python, Batch)
| 特性 | Infarct | CMB |
|------|---------|-----|
| batch_inputs | ✅ True | ✅ True |
| 脚本类型 | Bash (chuan) | Python (code_ai) |
| 命令生成 | _build_chuan_command | _build_python_command |
| PATH_ROOT | 需要 (Chuan 路径) | 不需要 |

### vs. WMH (Chuan, Non-Batch)
| 特性 | Infarct | WMH |
|------|---------|-----|
| batch_inputs | ✅ True | ❌ False |
| 命令生成 | _build_chuan_command | _build_chuan_command |
| 脚本名称 | pipeline_infarct.sh | pipeline_wmh.sh |

---

## Linus 哲学在 Infarct 流程中的应用

### 1. Data Structure Drives Behavior
```python
# 数据结构决定行为，无需 if-else 分支
if 'series_uids' in func_params:
    # Series Level
elif pipeline_config.batch_inputs:
    # Batch Mode
```

### 2. Single Source of Truth
```python
# PATH_ROOT 的单一来源：参数 > 环境变量
if path_root is not None:
    PATH_ROOT = pathlib.Path(path_root)  # From parameter (dual deployment)
else:
    PATH_ROOT = pathlib.Path(os.getenv("PATH_ROOT"))  # Fallback
```

### 3. Eliminate Special Cases
```python
# 不使用 None，寻找第一个有效数据
for path in dicom_series_paths:
    if path is not None:
        dicom_dir = path  # Use first valid path
        break
```

### 4. Fail Fast and Fail Loud
```python
# 立即检测并报告缺失文件
if missing_series:
    logger.error(f"Missing NIFTI files for series: {missing_series}")
    all_success = False
```

---

## 常见问题排查

### Q1: `inference_item_cmd` 为空
**原因**: `path_root` 参数缺失，无法解析 chuan_code 路径

**解决**: 确保 `path_root` 传递到 `generate_cmd()`
```python
cmd_str = pipeline_config.generate_cmd(
    ...,
    path_root=path_root  # ← 必须传递！
)
```

### Q2: `--InputsDicomDir` 收到 None
**原因**:
1. Backend 未提取 rename_dicom_paths
2. Worker 未传递到 dicom_series_paths

**解决**:
- Backend: 使用 `_extract_rename_dicom_path()` 提取
- Worker: 传递 `func_params["rename_dicom_paths"]`

### Q3: NIfTI 输入顺序错误
**原因**: Batch inputs 模型需要特定顺序（如 SWAN + T1BRAVO）

**解决**: `check_study_mapping_inference()` 自动排序
```python
if pipeline_config.batch_inputs and len(nifti_paths) > 1:
    sorted_nifti_paths = check_study_mapping_inference(study_path)
```

### Q4: Output directory 混乱
**原因**: Batch mode 应使用 inference_id，非 series_uid

**解决**:
```python
# Correct: inference_id for batch
output_dir = os.path.join(path_json, inference_id)

# Wrong: series_uid for batch
output_dir = os.path.join(path_json, series_uid)  # ❌
```

---

## 实际命令示例

### 示例 1: 双 Series Infarct 检测
```bash
# Input
series_uids: ["1.2.3.4.5.1", "1.2.3.4.5.2"]
model_id: "<infarct_model_uuid>"

# Generated Command
cd /mnt/e/pipeline/chuan/code && \
bash /mnt/e/pipeline/chuan/code/pipeline_infarct.sh \
14914694_20220905 \
/mnt/e/rename_nifti/14914694_20220905/DWI.nii.gz \
/mnt/e/rename_nifti/14914694_20220905/ADC.nii.gz \
/mnt/e/rename_dicom/14914694_20220905/1.2.3.4.5.1 \
/mnt/e/rename_nifti/14914694_20220905

# Output
/mnt/e/path_json/<inference_id>/prediction.json
```

### 示例 2: Mixed Mode (Direct + Convert)
```bash
# Input
series_A: Direct (NIfTI ready)
series_B: Convert (needs dcm2niix)

# Process
1. Backend: validate_series_ready()
   - series_A: Extract from CONVERSION_COMPLETE
   - series_B: Extract from TRANSFER_COMPLETE

2. Worker: _batch_convert_series_to_nifti()
   - Convert series_B: raw_dicom → rename_nifti

3. Command Generation:
   nifti_paths = [series_A_nifti, series_B_nifti]
   dicom_dir = series_A_rename_dicom  # First valid path

# Generated Command
bash .../pipeline_infarct.sh \
<study_id> \
/path/series_A.nii.gz \  ← Direct
/path/series_B.nii.gz \  ← Converted
/rename_dicom/series_A \ ← rename_dicom (not raw_dicom!)
/rename_nifti/<study_id>
```

---

## 总结

Infarct 命令生成流程的核心特点：

1. **Batch Inputs Mode**: 多个 series → 单次推论 → 共享 output_dir (inference_id)
2. **Chuan Integration**: 使用 Chuan 项目的 bash 脚本，需要 PATH_ROOT 解析
3. **rename_dicom Path**: 使用 rename_dicom（保留完整 metadata）而非 raw_dicom
4. **Path Ordering**: batch_inputs=True 自动排序确保输入顺序正确
5. **Dual Deployment**: path_root 参数支持多环境共享 GPU worker

关键文件:
- `code_ai/pipeline/__init__.py`: PipelineConfig 和命令构建
- `code_ai/task/task_pipeline.py`: Worker 任务处理和批量模式
- `backend/app/inference/service.py`: Backend 验证和队列管理
