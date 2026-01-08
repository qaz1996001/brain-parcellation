# Design: Fix DCOP Event study_uid Validation Error

## Problem Statement

在 Series Level 推理執行中,`_batch_convert_series_to_nifti()` 函數發送 DCOP 轉換事件時,傳遞 `study_uid=None` 導致 Pydantic 驗證失敗:

```
Failed to send conversion event for DWI0:
1 validation error for DCOPEventRequest
study_uid
  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
```

## Root Cause Analysis

### 調用鏈分析

```
_task_series_pipeline_inference() [Line 892-1000]
  ├─ 提取 study_uid from func_params [Line 892-896]
  │    study_uid = func_params.get("study_uid")  # ✅ 存在
  │
  └─ 調用 _batch_convert_series_to_nifti() [Line 990]
       ├─ 未傳遞 study_uid 參數  # ❌ 斷鏈點
       │
       └─ 函數內部 [Line 540]
            dcop_event = DCOPEventRequest(
                study_uid=None,  # ❌ 硬編碼 None
                ...
            )
```

### 錯誤假設

在 UID/Label 分離重構中,添加了註釋 "Series Level 可能沒有 study_uid",但此假設**不正確**:

1. **Series 總是屬於 Study**: 醫學影像數據模型中,Series 必定隸屬於某個 Study
2. **func_params 總是包含 study_uid**: Backend 調度任務時總是傳遞 study_uid
3. **Database 約束**: series 表有 study_id 外鍵,無法存在無 Study 的 Series

## Design Solution

### 核心設計原則

**Knuth 原則應用**:
1. **精確性** (第三原則): 明確定義 study_uid 必須傳遞,不容許 None
2. **可追溯性**: 事件鏈完整,Backend 可追溯到 Study
3. **可讀性** (第二原則): 參數傳遞鏈清晰,無隱式假設

### 設計決策

#### 決策 1: 參數傳遞 vs 全局變量

**選項 A**: 通過函數參數傳遞 study_uid ✅ (選擇)
- 優點: 純函數設計,可測試性高
- 優點: 遵循 Pure Function 重構模式
- 優點: 顯式依賴,易於追蹤

**選項 B**: 使用全局變量或環境變量 ❌
- 缺點: 隱式依賴,難以測試
- 缺點: 違反 Pure Function 原則
- 缺點: 可能導致多線程問題

**結論**: 選擇選項 A,遵循項目現有的 Pure Function 重構方向。

#### 決策 2: 參數類型 - str vs Optional[str]

**選項 A**: `study_uid: str` ✅ (選擇)
- 優點: 匹配 DCOPEventRequest Pydantic 模型要求
- 優點: 強制調用者提供有效值
- 優點: 無需處理 None 情況

**選項 B**: `study_uid: Optional[str]` ❌
- 缺點: 需要在函數內部處理 None 情況
- 缺點: 不匹配 Pydantic 模型要求
- 缺點: 增加代碼複雜度

**結論**: 選擇選項 A,study_uid 總是存在且有效。

#### 決策 3: 驗證位置

**選項 A**: 在調用點驗證 (父函數) ❌
```python
if study_uid is None:
    raise ValueError("study_uid is required")
```

**選項 B**: 依賴 Pydantic 驗證 ✅ (選擇)
- 優點: 統一驗證邏輯
- 優點: 減少重複代碼
- 優點: 驗證錯誤信息標準化

**結論**: 選擇選項 B,依賴現有的 Pydantic 驗證機制。

## Architecture Impact

### 修改範圍

```
影響層級: Worker Layer (code_ai/task/)
修改文件: code_ai/task/task_pipeline.py
修改行數: ~10 行 (3 處修改)
向後兼容: ✅ 是 (只有一處調用點)
```

### 依賴關係

```
DCOPEventRequest (Pydantic 模型)
  ↑ 依賴
_batch_convert_series_to_nifti() [新增 study_uid 參數]
  ↑ 調用
_task_series_pipeline_inference() [傳遞 study_uid]
  ↑ 調用
task_pipeline_inference() [funboost 任務入口]
```

### 數據流

```
Backend 調度:
  queue_series_inference() → func_params["study_uid"] = db_study.study_uid

RabbitMQ 傳輸:
  func_params → JSON → RabbitMQ Queue

Worker 消費:
  func_params → _task_series_pipeline_inference()
    → study_uid = func_params.get("study_uid")
    → _batch_convert_series_to_nifti(study_uid=study_uid)
      → DCOPEventRequest(study_uid=study_uid)

Backend 接收事件:
  event.study_uid → 查詢數據庫 → 更新 Study 狀態
```

## Implementation Details

### 修改 1: 函數簽名

**File**: `code_ai/task/task_pipeline.py`
**Line**: 467-474

```python
def _batch_convert_series_to_nifti(
    raw_dicom_paths: List[str],
    series_uids: List[str],
    target_labels: List[str],
    output_dicom_base: str,
    output_nifti_base: str,
    study_id: str,
    study_uid: str,  # ✅ 新增
    upload_data_api_url: Optional[str] = None,
) -> tuple:
    """
    批量轉換 DICOM series 到 NIfTI 格式

    Args:
        ...
        study_uid: Study UID (用於 DCOP 事件追溯)
        ...
    """
```

**變更類型**: 簽名修改 (新增參數)
**風險**: 低 (只有一處調用)

### 修改 2: 調用傳遞

**File**: `code_ai/task/task_pipeline.py`
**Line**: 990-998

```python
# 提取 study_uid (Line 892-896 已存在)
study_uid = func_params.get("study_uid")

# 調用時傳遞
dicom_series_paths, nifti_paths = _batch_convert_series_to_nifti(
    raw_dicom_paths=raw_dicom_paths,
    series_uids=series_uids,
    target_labels=target_labels,
    output_dicom_base=path_rename_dicom,
    output_nifti_base=path_rename_nifti,
    study_id=study_id,
    study_uid=study_uid,  # ✅ 傳遞
    upload_data_api_url=upload_data_api_url,
)
```

**變更類型**: 參數傳遞
**風險**: 低 (study_uid 總是存在)

### 修改 3: 使用參數

**File**: `code_ai/task/task_pipeline.py`
**Line**: 540-556

```python
dcop_event = DCOPEventRequest(
    study_uid=study_uid,  # ✅ 修改: None → study_uid
    series_uid=series_uid,
    study_id=study_id,
    ope_no=status,
    tool_id="NIFTI_TOOL",
    result_data={
        "raw_dicom_path": raw_path,
        "rename_dicom_path": dicom_path,
        "nifti_path": nifti_path,
        "target_label": target_label,
    },
)
```

**變更類型**: 值修改 (None → 參數)
**風險**: 低 (Pydantic 自動驗證)

### 修改 4: 移除誤導性註釋

**File**: `code_ai/task/task_pipeline.py`
**Line**: 540

```python
# 移除: # Series Level 可能沒有 study_uid
```

**變更類型**: 註釋清理
**風險**: 無

## Testing Strategy

### 單元測試

```python
def test_batch_convert_requires_study_uid():
    """測試函數簽名要求 study_uid"""
    import inspect
    sig = inspect.signature(_batch_convert_series_to_nifti)
    params = sig.parameters

    assert "study_uid" in params
    assert params["study_uid"].annotation == str
    assert params["study_uid"].default == inspect.Parameter.empty  # 非可選

def test_dcop_event_with_study_uid():
    """測試 DCOP 事件包含 study_uid"""
    with patch('code_ai.task.task_pipeline._send_dcop_event') as mock_send:
        _batch_convert_series_to_nifti(
            raw_dicom_paths=["/path/DWI"],
            series_uids=["308454c5-..."],
            target_labels=["DWI0"],
            output_dicom_base="/output/dicom",
            output_nifti_base="/output/nifti",
            study_id="14914694_...",
            study_uid="e9364c14-...",  # ✅ 傳遞
            upload_data_api_url="http://localhost:8000",
        )

    call_args = mock_send.call_args_list[0][0][0]
    assert call_args.study_uid == "e9364c14-..."
```

### 集成測試

```bash
# 觸發 Series Level 推理
curl -X POST http://localhost:8000/api/v1/inference/series \
  -H "Content-Type: application/json" \
  -d '{
    "study_uid": "e9364c14-...",
    "series_uids": ["308454c5-...", "86364c14-..."],
    "model_id": "infarct_v1"
  }'

# 檢查日誌
tail -f /path/logs/task_pipeline_inference_*.log | grep "Failed to send conversion event"
# 應該無輸出 ✅
```

### 端到端測試

1. 觸發完整的 Study 同步 → Series 轉換 → 推理流程
2. 驗證 Backend 數據庫中 Study 狀態更新
3. 確認所有 DCOP 事件成功記錄

## Error Handling

### 現有錯誤處理保持不變

```python
try:
    _send_dcop_event(dcop_event, upload_data_api_url)
except Exception as e:
    logger.warning(
        f"Failed to send conversion event for {target_label}: {e}"
    )
    # 不中斷轉換流程
```

### 新增的隱式錯誤處理

- **Pydantic 驗證**: 如果 study_uid 無效,Pydantic 會在構造時拋出 ValidationError
- **捕獲處理**: 現有的 `except Exception` 會捕獲 ValidationError
- **影響**: 日誌記錄錯誤,轉換流程繼續

## Performance Impact

### 性能分析

- **參數傳遞開銷**: O(1) 指針傳遞,~8 bytes
- **無額外計算**: 無循環或複雜邏輯
- **內存影響**: 可忽略 (<0.01% 增長)

### 基準測試 (可選)

```python
import time

# Before: study_uid=None
start = time.perf_counter()
for _ in range(1000):
    DCOPEventRequest(..., study_uid=None)  # ❌ 驗證失敗
end = time.perf_counter()
print(f"Before: {end - start:.4f}s")

# After: study_uid="e9364c14-..."
start = time.perf_counter()
for _ in range(1000):
    DCOPEventRequest(..., study_uid="e9364c14-...")  # ✅ 驗證通過
end = time.perf_counter()
print(f"After: {end - start:.4f}s")
```

**預期結果**: 性能改善 (驗證成功比失敗更快)

## Security Considerations

### study_uid 注入風險

- **來源**: Backend 從數據庫提取 study_uid
- **驗證**: Pydantic 模型驗證字符串類型
- **傳輸**: RabbitMQ JSON 序列化 (無注入風險)
- **結論**: 無安全風險

### 數據追溯性提升

- **Before**: study_uid=None → Backend 無法追溯
- **After**: study_uid=真實值 → Backend 可審計事件鏈
- **合規性**: 提升 IEC 62304 可追溯性要求

## Rollback Strategy

### 觸發條件

- Type checking 失敗且無法修復
- 端到端測試失敗
- 發現未知的調用點導致錯誤

### Rollback 步驟

1. **恢復函數簽名** (移除 study_uid 參數)
2. **恢復調用點** (移除傳遞)
3. **恢復 DCOPEventRequest** (study_uid=None)
4. **運行測試** (確保穩定性)

### Rollback 風險

- **低風險**: 修改範圍小,只有一處調用
- **可逆性**: 完全可逆,無數據庫遷移

## Documentation Updates

### 代碼文檔

- [x] `proposal.md`: Knuth 式數學分析
- [x] `tasks.md`: 任務清單
- [x] `design.md`: 本文檔
- [ ] `spec.md`: 規格說明 (如需要)

### CLAUDE.md 更新

建議添加到 "Common Pitfalls" 章節:

```markdown
**DCOP Event 完整性**:
- 所有 DCOP 事件必須包含有效的 study_uid
- study_uid 必須從 func_params 顯式傳遞
- 不允許 study_uid=None (Pydantic 驗證失敗)
```

## Success Metrics

### 定量指標

- ✅ DCOP 事件發送成功率: 0% → 100%
- ✅ Pydantic 驗證錯誤: 3/3 → 0/3
- ✅ Backend 事件追溯率: 0% → 100%

### 定性指標

- ✅ 代碼可讀性: 清晰的參數傳遞鏈
- ✅ 可維護性: 無隱式假設
- ✅ 可測試性: 純函數設計

## Timeline Estimate

- Task 1-3 (代碼修改): ~30 分鐘
- Task 4 (質量檢查): ~10 分鐘
- Task 5 (端到端驗證): ~20 分鐘
- Task 6 (回歸測試,可選): ~1 小時
- **Total**: ~2 小時 (含測試)

## Approval Checklist

- [x] 問題分析完整
- [x] 設計方案符合項目架構
- [x] 修改範圍明確
- [x] 測試策略充分
- [x] 風險評估完成
- [x] Rollback 方案準備
- [ ] 技術評審通過
- [ ] 用戶驗收測試通過

---

**設計者**: Claude Sonnet 4.5
**日期**: 2026-01-07
**版本**: 1.0
**狀態**: 等待評審
