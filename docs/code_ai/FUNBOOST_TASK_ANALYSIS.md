# Funboost Task 引用分析報告

## 📋 分析範圍

本報告分析了 `code_ai/task/` 目錄下所有 funboost task 函數的引用情況，識別出未被引用的可優化項目。

## ✅ 已被引用的 Tasks

### task_pipeline.py

| Task 函數 | 佇列名稱 | 引用位置 | 狀態 |
|----------|---------|---------|------|
| `task_pipeline_inference` | `task_pipeline_inference_queue` | ✅ `backend/app/sync/service.py`<br>✅ `backend/app/study/service.py`<br>✅ `code_ai/scheduler/scheduler_check_add_task.py`<br>✅ `code_ai/pipeline/task_pipeline_inference.py`<br>✅ `code_ai/pipeline/raw_diom_to_nii_inference.py`<br>✅ 多個測試文件 | **活躍使用** |
| `task_subprocess_inference` | `task_subprocess_queue` | ✅ `test/adni_run_pip.py` | **測試環境使用** |

### task_dicom2nii.py

| Task 函數 | 佇列名稱 | 引用位置 | 狀態 |
|----------|---------|---------|------|
| `call_post_httpx` | `post_httpx_queue` | ✅ `task_pipeline.py` (2處)<br>✅ `task_dicom2nii.py` (2處) | **活躍使用** |
| `call_dcm2niix` | `call_dcm2niix_queue` | ✅ `task_dicom2nii.py` 內部：<br>  - `dicom_2_nii_file` (2處)<br>  - `dicom_2_nii_series` (2處) | **活躍使用** |
| `dicom_2_nii_file` | `dicom_2_nii_file_queue` | ✅ `task_dicom2nii.py` 內部：<br>  - `dicom_to_nii` | **活躍使用** |
| `dicom_2_nii_series` | `dicom_2_nii_series_queue` | ✅ `backend/app/sync/service.py`<br>✅ `backend/app/study/service.py` | **活躍使用** |
| `process_instances` | `process_instances_queue` | ✅ `task_dicom2nii.py` 內部：<br>  - `process_dir` (2處) | **活躍使用** |
| `process_dir` | `process_dir_queue` | ✅ `task_dicom2nii.py` 內部：<br>  - `dicom_to_nii`<br>  - `dicom_rename` | **活躍使用** |
| `dicom_to_nii` | `dicom_to_nii_queue` | ✅ `backend/app/sync/service.py`<br>✅ `backend/app/study/service.py`<br>✅ `scripts/task_process_dir.py`<br>✅ `code_ai/pipeline/dicom_to_nii.py` (命令行工具)<br>✅ `code_ai/pipeline/raw_diom_to_nii_inference.py` | **活躍使用** |
| `dicom_rename` | `dicom_rename_queue` | ⚠️ `code_ai/pipeline/dicom_to_nii.py` (命令行工具) | **僅命令行工具** |

## ❌ 未被引用的 Tasks（已優化）

### ✅ 已刪除

| Task 函數 | 佇列名稱 | 位置 | 狀態 |
|----------|---------|------|------|
| ~~`raw_dicom_2_rename_dicom`~~ | `raw_dicom_2_rename_dicom_queue` | ~~`task_dicom2nii.py:232`~~ | **✅ 已刪除** |

**刪除原因：**
- 此函數完全沒有被任何地方引用
- 功能與 `dicom_to_nii` 重疊（都調用 `process_dir.push()`）
- 已於 2025-01-XX 刪除以簡化代碼庫

**注意事項：**
- 對應的 RabbitMQ 佇列 `raw_dicom_2_rename_dicom_queue` 可能需要手動清理

## ⚠️ 邊緣情況 Tasks（需確認）

### 🟡 僅在命令行工具中使用

| Task 函數 | 佇列名稱 | 引用位置 | 建議 |
|----------|---------|---------|------|
| `dicom_rename` | `dicom_rename_queue` | `code_ai/pipeline/dicom_to_nii.py` (命令行腳本) | **確認是否仍需保留** |

**分析：**
- 只在 `code_ai/pipeline/dicom_to_nii.py` 命令行工具中使用
- 該命令行工具可能用於手動測試或舊流程
- 建議確認該命令行工具是否仍在生產環境中使用

**代碼位置：**
```python:492:496:code_ai/task/task_dicom2nii.py
@Booster(BoosterParamsMyRABBITMQ(queue_name='dicom_rename_queue',
                                 qps=10, ))
def dicom_rename(func_params: Dict[str, any]):
    process_dir_result = process_dir.push(func_params)
    return process_dir_result
```

### 🟡 僅在測試環境中使用

| Task 函數 | 佇列名稱 | 引用位置 | 建議 |
|----------|---------|---------|------|
| `task_subprocess_inference` | `task_subprocess_queue` | `test/adni_run_pip.py` | **確認測試需求** |

**分析：**
- 只在測試文件中使用
- 功能是執行子進程命令
- 建議確認是否為必要的測試工具

## 📊 統計摘要

| 類別 | 數量 | 百分比 |
|------|------|--------|
| ✅ 活躍使用 | 8 | 80.0% |
| ⚠️ 邊緣使用 | 2 | 20.0% |
| ✅ 已刪除 | 1 | - |
| **總計** | **10** | **100%** |

## 🎯 優化建議

### ✅ 已完成優化

1. **✅ 已刪除 `raw_dicom_2_rename_dicom`**
   - 完全未被引用
   - 功能與 `dicom_to_nii` 重疊
   - 已刪除，減少代碼維護負擔

### 確認後優化（中優先級）

2. **評估 `dicom_rename` 的必要性**
   - 確認 `code_ai/pipeline/dicom_to_nii.py` 命令行工具是否仍在生產使用
   - 如果不再使用，可考慮刪除

3. **評估 `task_subprocess_inference` 的必要性**
   - 確認測試文件 `test/adni_run_pip.py` 是否仍在使用
   - 如果不再使用，可考慮刪除

### ✅ 已完成的優化步驟

```bash
# ✅ 1. 已刪除未使用的 task
# 已刪除 raw_dicom_2_rename_dicom 函數

# ⚠️ 2. 待處理：檢查是否有相關的 RabbitMQ 佇列需要清理
# 需要檢查 raw_dicom_2_rename_dicom_queue 佇列

# ⚠️ 3. 待處理：運行測試確保沒有破壞性影響
# pytest

# ✅ 4. 已提交變更
# git commit -m "refactor: 移除未使用的 funboost task raw_dicom_2_rename_dicom"
```

## 📝 注意事項

1. **佇列清理**：刪除 task 函數後，需要確認對應的 RabbitMQ 佇列是否也需要清理
2. **依賴檢查**：刪除前應檢查是否有其他模組間接依賴這些函數
3. **測試覆蓋**：刪除後應運行完整測試套件確保沒有回歸問題
4. **文檔更新**：如有相關文檔，需要同步更新

## 🔍 引用關係圖

```
task_pipeline_inference
├── backend/app/sync/service.py ✅
├── backend/app/study/service.py ✅
└── scheduler_check_add_task.py ✅

dicom_to_nii
├── backend/app/sync/service.py ✅
├── backend/app/study/service.py ✅
└── pipeline/dicom_to_nii.py (CLI) ⚠️

dicom_2_nii_series
├── backend/app/sync/service.py ✅
└── backend/app/study/service.py ✅

~~raw_dicom_2_rename_dicom~~ (已刪除)
└── (無引用) ✅ 已刪除

dicom_rename
└── pipeline/dicom_to_nii.py (CLI) ⚠️
```

---

**生成時間**：2025-01-XX  
**分析工具**：代碼庫語義搜索 + grep 模式匹配

