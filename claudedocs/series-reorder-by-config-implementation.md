# Series 重新排序功能实现文档

## 问题描述

在 Series Level 推理中，后端传入的 `target_labels` 顺序可能与 `config.yaml` 中定义的顺序不一致。例如：

- **Config.yaml 定义**：CMB 模型需要 `["SWAN", "T1BRAVO_AXI"]`
- **后端传入**：可能是 `["T1BRAVO_AXI", "SWAN"]` 或 `["SWAN", "T1BRAVO_AXI"]`

由于 `--InputsDicomDir` 参数取的是第一个位置的 DICOM 路径，顺序不一致会导致取到错误的 DICOM 目录。

## 解决方案

在 `_task_series_pipeline_inference` 函数开始处理之前，根据 `config.yaml` 中定义的顺序重新排列所有相关列表。

### 实现位置

**文件**：`code_ai/task/task_pipeline.py`

### 核心改动

#### 1. 新增 `_reorder_series_by_config` 函数（Line 923-1070）

该函数负责：
1. 读取 `config.yaml` 文件
2. 解析 `model_id` 到模型名称（如 "CMB"）
3. 获取模型定义的 series 顺序（如 `["SWAN", "T1BRAVO_AXI"]`）
4. 根据 config 定义的顺序重新排列所有传入的列表

**函数签名**：
```python
def _reorder_series_by_config(
    target_labels: List[str],
    series_uids: List[str],
    model_id: str,
    nifti_series_paths: Optional[List[str]] = None,
    raw_dicom_series_paths: Optional[List[str]] = None,
    rename_dicom_paths: Optional[List[str]] = None,
) -> tuple:
```

**返回值**：
```python
(
    sorted_target_labels,      # 重新排序的 target_labels
    sorted_series_uids,         # 重新排序的 series_uids
    sorted_nifti_paths,         # 重新排序的 nifti_series_paths
    sorted_raw_dicom_paths,     # 重新排序的 raw_dicom_series_paths
    sorted_rename_dicom_paths,  # 重新排序的 rename_dicom_paths
)
```

#### 2. 在 `_task_series_pipeline_inference` 中调用（Line 1148-1173）

在提取参数之后、执行转换之前调用该函数：

```python
# Step 2.6: 根據 config.yaml 重新排序 target_labels 和相關列表
# 確保 --InputsDicomDir 取到正確的第一個序列
(
    target_labels,
    series_uids,
    sorted_nifti_paths,
    sorted_raw_dicom_paths,
    sorted_rename_dicom_paths,
) = _reorder_series_by_config(
    target_labels=target_labels,
    series_uids=series_uids,
    model_id=model_id,
    nifti_series_paths=func_params.get("nifti_series_paths"),
    raw_dicom_series_paths=func_params.get("raw_dicom_series_paths"),
    rename_dicom_paths=func_params.get("rename_dicom_paths"),
)

# 更新 func_params 中的排序後列表
func_params["series_uids"] = series_uids
func_params["target_labels"] = target_labels
if sorted_nifti_paths is not None:
    func_params["nifti_series_paths"] = sorted_nifti_paths
if sorted_raw_dicom_paths is not None:
    func_params["raw_dicom_series_paths"] = sorted_raw_dicom_paths
if sorted_rename_dicom_paths is not None:
    func_params["rename_dicom_paths"] = sorted_rename_dicom_paths
```

## 实现细节

### 1. Config.yaml 解析

从 `config.yaml` 中读取模型定义：
```yaml
model_mapping_series:
  CMB:
    - ["MRSeriesRenameEnum.SWAN", "T1SeriesRenameEnum.T1BRAVO_AXI"]
    - ["MRSeriesRenameEnum.SWAN", "T1SeriesRenameEnum.T1FLAIR_AXI"]
```

提取第一个配置项，并去掉 Enum 前缀：
- `"MRSeriesRenameEnum.SWAN"` → `"SWAN"`
- `"T1SeriesRenameEnum.T1BRAVO_AXI"` → `"T1BRAVO_AXI"`

### 2. 索引映射

建立从 `target_label` 到原始索引的映射：
```python
label_to_index = {label: i for i, label in enumerate(target_labels)}
# 例如：{"T1BRAVO_AXI": 0, "SWAN": 1}
```

根据 config 定义的顺序查找索引：
```python
sorted_indices = []
for config_label in config_labels:  # ["SWAN", "T1BRAVO_AXI"]
    if config_label in label_to_index:
        sorted_indices.append(label_to_index[config_label])
# 结果：[1, 0] （SWAN 在原始列表的索引 1，T1BRAVO_AXI 在索引 0）
```

### 3. 重新排列所有列表

使用 `sorted_indices` 重新排列所有列表：
```python
sorted_target_labels = [target_labels[i] for i in sorted_indices]
sorted_series_uids = [series_uids[i] for i in sorted_indices]
sorted_nifti_paths = [nifti_series_paths[i] for i in sorted_indices]
# ...
```

### 4. 容错处理

- 如果 config.yaml 中没有该模型的配置，保持原顺序
- 如果无法匹配所有 config labels，保持原顺序
- 如果发生任何异常，保持原顺序并记录警告日志

## 测试示例

### 输入（后端传入）：
```python
target_labels = ["T1BRAVO_AXI", "SWAN"]
series_uids = [
    "2c4bc8a9-2cd5576d-776c485f-180b3212-5c708eb1",  # T1BRAVO_AXI
    "fb9ec858-3790e92e-ddf5d73b-8cfcea20-63d5499f",  # SWAN
]
nifti_paths = [
    "/path/T1BRAVO_AXI.nii.gz",
    "/path/SWAN.nii.gz",
]
rename_dicom_paths = [
    "/path/T1BRAVO_AXI",
    "/path/SWAN",
]
```

### 输出（重新排序后）：
```python
sorted_target_labels = ["SWAN", "T1BRAVO_AXI"]
sorted_series_uids = [
    "fb9ec858-3790e92e-ddf5d73b-8cfcea20-63d5499f",  # SWAN
    "2c4bc8a9-2cd5576d-776c485f-180b3212-5c708eb1",  # T1BRAVO_AXI
]
sorted_nifti_paths = [
    "/path/SWAN.nii.gz",
    "/path/T1BRAVO_AXI.nii.gz",
]
sorted_rename_dicom_paths = [
    "/path/SWAN",
    "/path/T1BRAVO_AXI",
]
```

### 结果：
- `--InputsDicomDir` 现在取的是 `/path/SWAN`（第一个位置）✅
- 这对应的是 SWAN 序列，与 config.yaml 定义一致 ✅

## 日志输出

当重新排序发生时，会输出日志：
```
INFO: Reordered series by config.yaml: ["T1BRAVO_AXI", "SWAN"] -> ["SWAN", "T1BRAVO_AXI"]
```

当无法重新排序时，会输出警告：
```
WARNING: No series mapping found in config.yaml for model CMB, keeping original order
WARNING: Cannot match all config labels to target_labels, keeping original order
WARNING: Failed to reorder series by config.yaml: <error>, keeping original order
```

## 向后兼容性

- ✅ 如果后端已经传入正确顺序，重新排序不会改变结果
- ✅ 如果 `config.yaml` 中没有该模型配置，保持原顺序
- ✅ 所有错误情况都会 fallback 到保持原顺序

## 适用模型

该功能适用于所有多序列输入的模型：
- **CMB**: `["SWAN", "T1BRAVO_AXI"]`
- **Infarct**: `["DWI0", "DWI1000", "ADC"]`
- 未来新增的多序列模型

## 未来改进

当前实现是临时方案，未来可以考虑：
1. 在 `config.yaml` 中添加 `InputsDicomDir` 配置，显式指定使用哪个序列的 DICOM 目录
2. 支持更灵活的排序规则（例如按优先级而非固定顺序）

## 相关文件

- `code_ai/task/task_pipeline.py` - 核心实现
- `code_ai/utils/inference/config.yaml` - 模型配置文件
- `test_reorder_series.py` - 测试脚本（需要 tensorflow 环境）
