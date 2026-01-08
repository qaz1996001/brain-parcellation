# Inference Parameters Design - Linus Style

## 核心原則

> "Bad programmers worry about the code. Good programmers worry about data structures."

數據結構本身應該表達意圖，不需要額外的 `type` 或 `level` 字段。

## 數據結構定義

### Study 級別參數（舊邏輯）

```python 
StudyInferenceParams = {
    # 必需字段
    'study_uid': str,           # Study 唯一標識
    'study_id': str,            # Study ID
    'nifti_study_path': str,    # NIFTI study 路徑
    'dicom_study_path': str,    # DICOM study 路徑

    # 路徑配置（可選，有環境變數 fallback）
    'path_process': str,        # 處理路徑
    'path_json': str,           # JSON 輸出路徑
    'path_log': str,            # 日誌路徑

    # API 配置（可選，有環境變數 fallback）
    'upload_data_api_url': str, # 回調 API URL
}
```

**判斷邏輯**：
```python
def is_study_level(func_params: dict) -> bool:
    return 'series_uids' not in func_params
```

### Series 級別參數（新邏輯）

```python
SeriesInferenceParams = {
    # 必需字段
    'study_uid': str,                # Study 唯一標識
    'study_id': str,                 # Study ID
    'series_uids': List[str],        # 🔑 Series 唯一標識列表
    'model_id': str,                 # 🔑 模型 UUID 或標識
    'nifti_study_path': str,    # NIFTI study 路徑
    'dicom_study_path': str,    # DICOM study 路徑

    # 可選：模型選擇（二選一）
    'model_name': str,               # 模型名稱
    'model_version': str,            # 模型版本

    # 路徑配置（可選，有環境變數 fallback）
    'path_process': str,
    'path_json': str,
    'path_log': str,

    # API 配置（可選，有環境變數 fallback）
    'upload_data_api_url': str,

    # 推論追蹤
    'inference_id': str,             # 推論任務 ID（用於回調）
}
```

**判斷邏輯**：
```python
def is_series_level(func_params: dict) -> bool:
    return 'series_uids' in func_params
```

## 向後兼容性保證

### 兼容性矩陣

| 調用方式 | series_uids | nifti_study_path | 行為 | 兼容性 |
|---------|-------------|------------------|------|--------|
| 舊代碼 | ❌ 無 | ✅ 有 | Study 級別 | ✅ 100% |
| 新代碼(study) | ❌ 無 | ✅ 有 | Study 級別 | ✅ 100% |
| 新代碼(series) | ✅ 有 | ❌ 無 | Series 級別 | ✅ 新功能 |
| 錯誤調用 | ✅ 有 | ✅ 有 | ❌ 拋異常 | ⚠️ 快速失敗 |

### 互斥性驗證

```python
def validate_inference_params(func_params: dict) -> None:
    """驗證參數互斥性，確保不會混淆 study 和 series 級別

    Raises:
        ValueError: 當同時提供 study 和 series 級別參數時
    """
    has_study_path = 'nifti_study_path' in func_params
    has_series_uids = 'series_uids' in func_params

    # Linus: "Good code has no special cases"
    # 這裡的驗證消除了歧義，使代碼路徑清晰
    if has_study_path and has_series_uids:
        raise ValueError(
            "Cannot specify both study-level (nifti_study_path) "
            "and series-level (series_uids) parameters. "
            "Choose one inference level."
        )

    if has_series_uids:
        # Series 級別必需字段驗證
        required = ['series_uids', 'nifti_series_paths', 'model_id']
        missing = [f for f in required if f not in func_params]
        if missing:
            raise ValueError(f"Series-level inference missing: {missing}")

        # 驗證列表長度一致
        if len(func_params['series_uids']) != len(func_params['nifti_series_paths']):
            raise ValueError(
                "series_uids and nifti_series_paths must have same length"
            )
```

## 實際使用示例

### 場景 1: 現有代碼繼續工作（Study 級別）

```python
# backend/app/sync/service.py (現有代碼，無需修改)
func_params = {
    'study_uid': 'ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30',
    'study_id': '10089413_20210201_MR_21002010079',
    'nifti_study_path': '/data/nifti/10089413_20210201_MR_21002010079',
    'dicom_study_path': '/data/dicom/10089413_20210201_MR_21002010079',
    'path_process': '/workspace/process',
    'upload_data_api_url': 'http://localhost:8000/api/v1',
}

# 推送到舊 queue（現有行為）
task_pipeline_inference.push(func_params)
```

### 場景 2: 新功能（Series 級別）

```python
# backend/app/inference/service.py (新代碼)
func_params = {
    'study_uid': 'ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30',
    'study_id': '10089413_20210201_MR_21002010079',
    'series_uids': [
        '0c0a1444-9238e5ad-fbcd0251-335322e7-9af7b058',
        '4a128011-eb094052-044f0a9b-dad8e303-d1bd2514',
    ],
    'model_id': 'uuid-model-123',
    'inference_id': 'inf-20240101-001',
    'path_process': '/workspace/process',
    'upload_data_api_url': 'http://localhost:8000/api/v1',
}

# 推送到新 queue（新功能）
task_series_inference.push(func_params)
```

### 場景 3: Service 層智能分派

```python
# 統一的 service 方法（可選）
def queue_inference_task(func_params: dict) -> str:
    """根據參數自動選擇正確的 queue

    Linus: "Let the data structure do the talking"
    """
    validate_inference_params(func_params)  # 快速失敗

    if 'series_uids' in func_params:
        # Series 級別
        task_series_inference.push(func_params)
        return func_params['inference_id']
    else:
        # Study 級別
        task_pipeline_inference.push(func_params)
        return func_params.get('study_uid', 'unknown')
```

## 關鍵設計決策

### ✅ 為什麼用 `series_uids` 作為判斷標誌？

1. **語義清晰**：有 `series_uids` = series 級別，沒有 = study 級別
2. **自然互斥**：Study 級別不需要指定 series（處理所有）
3. **零歧義**：不可能同時有意義地提供兩者
4. **向後兼容**：舊代碼中沒有這個字段，自動走 study 邏輯

### ✅ 為什麼不添加 `inference_level` 字段？

```python
# ❌ 冗餘設計（Linus 會罵）
func_params = {
    'inference_level': 'series',  # 多餘！
    'series_uids': [...],         # 已經說明了是 series 級別
}

# ✅ 簡潔設計
func_params = {
    'series_uids': [...],  # 數據結構本身就是文檔
}
```

> "If you need a comment to explain what the data structure means, your data structure is wrong."

### ✅ 為什麼要驗證互斥性？

```python
# 這種情況應該是程序錯誤，不是用戶錯誤
func_params = {
    'nifti_study_path': '/study',   # Study 級別？
    'series_uids': ['1.2.3'],       # Series 級別？
    # → 快速失敗，避免隱藏的 bug
}
```

> "Errors should never pass silently. Unless explicitly silenced."

## 遷移路徑

### Phase 1: 添加新功能（零破壞）
- 創建 `task_series_inference.py`
- 新 service 使用新參數格式
- 舊代碼完全不受影響

### Phase 2: 統一（可選）
- 如果需要，可以創建統一的 dispatcher
- 但不是必需的！簡單就是美

### Phase 3: 監控與驗證
- 監控兩個 queue 的使用情況
- 驗證沒有誤用（study/series 參數混用）

## Linus 的最後建議

1. **Keep It Simple**: 不要過度設計，一個 `if` 就夠了
2. **Data Speaks**: 讓數據結構表達意圖，不要依賴註釋或文檔
3. **Fail Fast**: 參數錯誤時立即拋異常，不要嘗試"聰明"地猜測
4. **Never Break**: 任何現有調用都必須繼續工作

---

*"Talk is cheap. Show me the code." - Linus Torvalds*
