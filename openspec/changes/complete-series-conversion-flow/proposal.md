# 變更提案: 完整實現 Series Level 轉換流程

## 為什麼需要這個變更

目前 Series Level 推論 API 只實現了「直接模式」（假設 NIfTI 檔案已存在），但沒有完整實現從 raw_dicom 到 inference 的完整流程：

```
raw_dicom → rename_dicom → rename_nifti → inference
```

**現狀問題**：
1. `inference/service.py` 僅從 DCOP events 提取已存在的 `nifti_series_paths`
2. 當 Series 尚未轉換時，使用者無法透過 API 觸發完整流程
3. 需要手動確保 NIfTI 檔案存在後才能調用 API

**好消息**：Worker 層 (`task_pipeline.py`) 已實現完整轉換邏輯：
- `_convert_single_series_to_nifti`: 單個 series 轉換
- `_batch_convert_series_to_nifti`: 批量轉換
- `needs_conversion=True` 模式已存在

## Linus 三問

### 1. 這是真實問題還是想像問題？

✅ **真實問題**：使用者希望透過單一 API 完成「raw_dicom → inference」，而非手動準備 NIfTI 檔案。

### 2. 有更簡單的方法嗎？

✅ **資料結構驅動行為**：只需在 `inference/service.py` 中傳遞正確的參數：
- 已轉換 → `nifti_series_paths` (現有邏輯)
- 未轉換 → `needs_conversion=True` + `raw_dicom_series_paths`

**無需修改 Worker 層**，僅修改 Service 層的參數傳遞邏輯。

### 3. 這會破壞什麼？

✅ **不會破壞任何東西**：
- 原有的「直接模式」完全保留
- 新增「轉換模式」作為補充
- 向後兼容

## 變更內容

### 核心思想：讓資料結構決定行為

**Linus**: "Bad programmers worry about the code. Good programmers worry about data structures."

目前 Worker 已實現：
```python
if needs_conversion:
    # 轉換模式: raw_dicom → rename_dicom → nifti → inference
    nifti_paths = _batch_convert_series_to_nifti(...)
else:
    # 直接模式: nifti → inference
    nifti_paths = func_params['nifti_series_paths']
```

Service 層只需傳遞正確的參數即可啟用轉換模式。

### 具體修改

**修改 1: `inference/schemas.py`**
- 新增 `needs_conversion` 可選欄位
- 新增 `raw_dicom_series_paths` 可選欄位

**修改 2: `inference/service.py`**
- 增強 `queue_series_inference` 方法
- 當 series 未轉換完成時，從 DCOP events 獲取 raw_dicom 路徑
- 設置 `needs_conversion=True` + `raw_dicom_series_paths`

**修改 3: `backend/app/config/task_paths.py`**
- 新增 `PATH_RENAME_DICOM` 和 `PATH_RENAME_NIFTI` 配置

### 破壞性變更

無 - 完全向後兼容。

## 影響範圍

### 受影響的程式碼

**修改檔案** (3 個檔案):
- `backend/app/inference/schemas.py` - 新增可選欄位 (~10 行)
- `backend/app/inference/service.py` - 增強參數傳遞邏輯 (~30 行)
- `backend/app/config/task_paths.py` - 新增路徑配置 (~5 行)

**不修改**：
- ✅ `code_ai/task/task_pipeline.py` - 已實現完整邏輯
- ✅ `backend/app/sync/` - 不影響
- ✅ 任何既有功能

### 設計原則

1. **最小修改原則**: 重用 Worker 已有邏輯，只修改參數傳遞
2. **資料結構驅動**: 讓 `needs_conversion` flag 決定行為
3. **向後兼容**: 原有 API 調用方式不變
4. **Fail Fast**: 明確的參數驗證和錯誤訊息
