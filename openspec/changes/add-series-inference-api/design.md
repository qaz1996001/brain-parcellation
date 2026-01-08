# Series Inference API 設計文件

> "Bad programmers worry about the code. Good programmers worry about data structures and their relationships."
> — Linus Torvalds

## 設計哲學

本設計嚴格遵循 Linus Torvalds 的程式設計哲學：

1. **資料結構優先**：正確的資料結構讓程式碼自然而然地簡單
2. **消除特殊情況**：用不同角度看問題，讓邊緣案例變成正常情況
3. **不過度設計**：從小開始，讓程式碼在實際使用中演化
4. **簡單明瞭**：函數要短，做一件事，做好它

---

## 最終設計：統一入口模式

### 核心洞見

> "Good taste is about understanding the problem well enough that the solution becomes obvious."

**問題**：Study Level 與 Series Level 推論需要共用 GPU 資源，如何避免衝突？

**早期錯誤方案**：
- Redis 鎖
- DB Semaphore
- 複雜的回呼機制

**最終方案**：使用單一佇列 + 資料結構決定行為

```
┌─────────────────────────────────────────────────────────────────────┐
│                    統一入口設計 (Unified Entry Point)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Study Level Request          Series Level Request                  │
│   (sync/service.py)            (inference/service.py)                │
│          │                              │                            │
│          └──────────┬───────────────────┘                            │
│                     │                                                │
│                     ▼                                                │
│   ┌─────────────────────────────────────────────┐                    │
│   │       task_pipeline_inference.push()        │                    │
│   │         (queue: qps=1, 自然互斥)             │                    │
│   └─────────────────────────────────────────────┘                    │
│                     │                                                │
│                     ▼                                                │
│   ┌─────────────────────────────────────────────┐                    │
│   │    if 'series_uids' in func_params:         │                    │
│   │        → _task_series_pipeline_inference()  │   ← 資料結構       │
│   │    else:                                    │     決定行為        │
│   │        → _task_study_pipeline_inference()   │                    │
│   └─────────────────────────────────────────────┘                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 設計優點

| 特性 | 說明 |
|------|------|
| **GPU 互斥** | 單一佇列 `qps=1` 提供自然的 GPU 資源互斥 |
| **零協調成本** | 無需 Redis 鎖、DB Semaphore 或外部協調 |
| **向後相容** | 既有 Study Level 邏輯完全不變 |
| **Linus 風格** | 資料結構（`series_uids` 存在與否）決定行為 |

---

## 資料結構設計

### Study Level 參數（既有）

```python
func_params_study = {
    'study_uid': str,
    'study_id': str,
    'path_process': str,
    'path_json': str,
    'path_log': str,
    'upload_data_api_url': str,
    # 注意：沒有 'series_uids' 欄位
}
```

### Series Level 參數（新增）

```python
func_params_series = {
    # === 識別資訊（新增）===
    'series_uids': List[str],     # ← 關鍵欄位：存在即表示 Series Level
    'model_id': str,
    'inference_id': str,

    # === Study 上下文 ===
    'study_uid': str,
    'study_id': str,

    # === 配置（與 Study Level 相同）===
    'path_process': str,
    'path_json': str,
    'path_log': str,
    'upload_data_api_url': str,
}
```

### Linus 原則驗證

```python
# Linus: "Data structure is the documentation"

def task_pipeline_inference(func_params: Dict[str, any]):
    """統一入口：資料結構決定行為"""
    if 'series_uids' in func_params:
        # Series Level: 參數包含 series_uids
        return _task_series_pipeline_inference(func_params)
    else:
        # Study Level: 傳統推論（原有邏輯不變）
        return _task_study_pipeline_inference(func_params)
```

---

## 實作細節

### task_pipeline.py 結構

```python
# code_ai/task/task_pipeline.py

@Booster(BoosterParamsMyAI(queue_name='task_pipeline_inference_queue', qps=1))
def task_pipeline_inference(func_params: Dict[str, any]):
    """
    統一入口點：處理 Study Level 和 Series Level 推論。

    GPU 互斥由 qps=1 自然實現，無需額外協調機制。

    Linus 原則：資料結構決定行為
    - 'series_uids' in func_params → Series Level
    - 'series_uids' not in func_params → Study Level
    """
    if 'series_uids' in func_params:
        return _task_series_pipeline_inference(func_params)
    else:
        return _task_study_pipeline_inference(func_params)


def _task_study_pipeline_inference(func_params: Dict[str, any]):
    """
    Study Level 推論（原有邏輯）。

    完全保留既有實作，不做任何修改。
    """
    # ... 原有 500+ 行邏輯 ...


def _task_series_pipeline_inference(func_params: Dict[str, any]):
    """
    Series Level 推論（新增邏輯）。

    處理指定 series 的推論請求。
    """
    # 1. 驗證必要參數
    _validate_series_params(func_params)

    # 2. 記錄狀態變更
    _update_series_status(func_params, DCOPStatus.SERIES_INFERENCE_RUNNING)

    # 3. 執行推論
    try:
        result = _run_series_inference(func_params)
        _update_series_status(func_params, DCOPStatus.SERIES_INFERENCE_COMPLETE)
        return result
    except Exception as e:
        _update_series_status(func_params, DCOPStatus.SERIES_INFERENCE_FAILED)
        raise
```

### inference/service.py 調整

```python
# backend/app/inference/service.py

async def queue_series_inference(self, request: SeriesInferenceRequest):
    # ... 驗證邏輯 ...

    func_params = {
        'series_uids': accepted_series,  # ← 關鍵：此欄位存在決定 Series Level
        'model_id': model_id,
        'inference_id': str(inference_id),
        'study_uid': request.study_uid,
        # ... 其他參數 ...
    }

    # 使用統一入口
    from code_ai.task.task_pipeline import task_pipeline_inference
    task_pipeline_inference.push(func_params)
```

---

## DCOPStatus 狀態碼

### 既有狀態（Study Level）

| 狀態碼 | 名稱 | 說明 |
|--------|------|------|
| 300.110 | STUDY_INFERENCE_READY | Study 準備推論 |
| 300.120 | STUDY_INFERENCE_QUEUED | Study 已排隊 |
| 300.130 | STUDY_INFERENCE_RUNNING | Study 推論中 |
| 300.140 | STUDY_INFERENCE_COMPLETE | Study 推論完成 |
| 300.150 | STUDY_INFERENCE_FAILED | Study 推論失敗 |

### 新增狀態（Series Level）

| 狀態碼 | 名稱 | 說明 |
|--------|------|------|
| 300.115 | SERIES_INFERENCE_READY | Series 準備推論 |
| 300.125 | SERIES_INFERENCE_QUEUED | Series 已排隊 |
| 300.155 | SERIES_INFERENCE_RUNNING | Series 推論中 |
| 300.165 | SERIES_INFERENCE_COMPLETE | Series 推論完成 |
| 300.175 | SERIES_INFERENCE_FAILED | Series 推論失敗 |

### tool_id 區分

| tool_id | 說明 |
|---------|------|
| `INFERENCE_TOOL` | Study Level 推論 |
| `SERIES_INFERENCE_TOOL` | Series Level 推論 |

---

## 檔案修改清單

### 修改的檔案

| 檔案 | 修改內容 |
|------|----------|
| `code_ai/task/task_pipeline.py` | 重構為統一入口，原邏輯移至 `_task_study_pipeline_inference` |
| `backend/app/inference/service.py` | 改用 `task_pipeline_inference.push()` |

### 刪除的檔案

| 檔案 | 原因 |
|------|------|
| `code_ai/task/task_series_inference.py` | 邏輯已整合至 `task_pipeline.py` |

### 不修改的檔案

| 檔案 | 原因 |
|------|------|
| `backend/app/sync/service.py` | Study Level 既有流程不受影響 |
| `code_ai/task/task_dicom2nii.py` | 轉換邏輯獨立運作 |

---

## GPU 衝突解決方案比較

| 方案 | 複雜度 | 可靠性 | 採用 |
|------|--------|--------|------|
| Redis 分布式鎖 | 高 | 中 | ❌ |
| DB Semaphore | 高 | 中 | ❌ |
| 獨立佇列 + 外部協調 | 高 | 低 | ❌ |
| **單一佇列 qps=1** | **低** | **高** | ✅ |

### 單一佇列方案的優勢

1. **funboost 原生支援**：`qps=1` 確保同一時間只有一個任務執行
2. **無需外部依賴**：不需要額外的 Redis 鎖或 DB 操作
3. **自然順序**：先進先出，公平調度
4. **簡單可靠**：減少故障點，易於除錯

---

## 設計驗證（Linus 品味檢查）

| 檢查項目 | 狀態 | 說明 |
|----------|------|------|
| 資料結構設計清楚？ | ✅ | `series_uids` 欄位決定行為 |
| 消除特殊情況？ | ✅ | 單一 `if` 處理兩種模式 |
| 函數簡短？ | ✅ | 入口點僅 4 行 |
| 無隱藏依賴？ | ✅ | 所有參數顯式傳遞 |
| 不過度設計？ | ✅ | 重用既有佇列機制 |
| 向後相容？ | ✅ | Study Level 邏輯完全保留 |

---

## 總結

本設計的核心洞見是：

> 「不需要複雜的協調機制，只需要正確的資料結構。」

透過將 `series_uids` 作為判斷依據，我們：
1. 使用單一入口點處理兩種推論模式
2. 利用 funboost 的 `qps=1` 實現自然的 GPU 互斥
3. 保持程式碼簡潔且可維護

這正是 Linus 所說的「好品味」：

> "Sometimes the elegant implementation is just a function. Not a method. Not a class. Not a framework. Just a function."
