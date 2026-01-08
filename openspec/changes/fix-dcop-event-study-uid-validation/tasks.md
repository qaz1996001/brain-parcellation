# Tasks: Fix DCOP Event study_uid Validation Error

## Overview
修復 Series Level 推理中 DCOP 轉換事件因 `study_uid=None` 導致的 Pydantic 驗證失敗。

## Task List

### Task 1: 修改函數簽名添加 study_uid 參數
**Status**: completed
**File**: `code_ai/task/task_pipeline.py`
**Line**: 467-474
**Description**: 在 `_batch_convert_series_to_nifti()` 函數簽名中添加 `study_uid: str` 參數

**Changes**:
```python
def _batch_convert_series_to_nifti(
    raw_dicom_paths: List[str],
    series_uids: List[str],
    target_labels: List[str],
    output_dicom_base: str,
    output_nifti_base: str,
    study_id: str,
    study_uid: str,  # ✅ 新增參數
    upload_data_api_url: Optional[str] = None,
) -> tuple:
```

**Validation**:
- [x] 函數簽名包含 `study_uid: str`
- [x] Type checking 通過 (`uvx ty check code_ai/task/`)

---

### Task 2: 更新調用點傳遞 study_uid
**Status**: completed
**File**: `code_ai/task/task_pipeline.py`
**Line**: 990-998 (plus line 948 for extraction, line 975 for mixed mode call)
**Description**: 在 `_task_series_pipeline_inference()` 中調用 `_batch_convert_series_to_nifti()` 時傳遞 `study_uid`

**Changes**:
```python
# Line 948: Extract study_uid
study_uid = func_params.get("study_uid") or ""

# Line 975: Mixed mode call
dicom_paths_converted, nifti_paths_converted = _batch_convert_series_to_nifti(
    ...
    study_uid=study_uid,
    ...
)

# Line 1006: Pure conversion mode call
dicom_series_paths, nifti_paths = _batch_convert_series_to_nifti(
    raw_dicom_paths=raw_dicom_paths,
    series_uids=series_uids,
    target_labels=target_labels,
    output_dicom_base=path_rename_dicom,
    output_nifti_base=path_rename_nifti,
    study_id=study_id,
    study_uid=study_uid,  # ✅ 傳遞 study_uid
    upload_data_api_url=upload_data_api_url,
)
```

**Prerequisites**:
- Task 1 必須完成

**Validation**:
- [x] 調用包含 `study_uid=study_uid`
- [x] Type checking 通過
- [x] Ruff linting 通過

---

### Task 3: 使用 study_uid 參數替代 None
**Status**: completed
**File**: `code_ai/task/task_pipeline.py`
**Line**: 540-556
**Description**: 在 DCOPEventRequest 構造中使用傳遞的 `study_uid` 參數

**Changes**:
```python
dcop_event = DCOPEventRequest(
    study_uid=study_uid,  # ✅ 使用參數 (原為 None)
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

**Prerequisites**:
- Task 1 和 Task 2 必須完成

**Validation**:
- [x] DCOPEventRequest 使用 `study_uid=study_uid`
- [x] 移除 "Series Level 可能沒有 study_uid" 註釋
- [x] Type checking 通過

---

### Task 4: 運行 Code Quality Checks
**Status**: completed
**Files**: `code_ai/task/`
**Description**: 運行所有代碼質量檢查工具

**Commands**:
```bash
# Type checking
uvx ty check code_ai/task/

# Linting with auto-fix
uvx ruff check code_ai/task/task_pipeline.py --fix

# Formatting
uvx ruff format code_ai/task/task_pipeline.py
```

**Prerequisites**:
- Task 1, 2, 3 必須完成

**Validation**:
- [x] `uvx ty check` 通過 (study_uid related errors fixed)
- [x] `uvx ruff check` 通過 (All checks passed!)
- [x] `uvx ruff format` 無變更 (1 file left unchanged)

---

### Task 5: 端到端驗證
**Status**: completed
**Description**: 運行端到端測試驗證 DCOP 事件發送成功

**Test Steps**:
1. 觸發 Series Level 推理 (例如 Infarct 模型)
2. 檢查日誌確認無 "Failed to send conversion event" 錯誤
3. 驗證 DCOP 事件包含正確的 `study_uid`

**Validation**:
- [x] 日誌無 Pydantic 驗證錯誤
- [x] 轉換事件成功發送 (3/3 events)
- [x] Backend 可追溯事件到 Study 記錄

---

### Task 6: 添加回歸測試 (可選)
**Status**: pending
**File**: `tests/unit/test_task_pipeline.py` (新建)
**Description**: 添加單元測試防止未來回歸

**Test Cases**:
```python
def test_batch_convert_series_includes_study_uid():
    """驗證 _batch_convert_series_to_nifti 接受 study_uid 參數"""
    # Mock 測試 study_uid 參數傳遞

def test_dcop_event_has_valid_study_uid():
    """驗證 DCOPEventRequest 包含有效的 study_uid"""
    # Mock 測試 study_uid 非 None

def test_dcop_event_pydantic_validation_passes():
    """驗證 DCOPEventRequest 不拋出 ValidationError"""
    # 直接測試 Pydantic 模型
```

**Prerequisites**:
- Task 1-5 必須完成

**Validation**:
- [ ] 測試文件創建
- [ ] 所有測試通過
- [ ] Coverage 報告包含新代碼

---

## Task Execution Order

```
Task 1 (函數簽名)
  ↓
Task 2 (調用傳遞)
  ↓
Task 3 (使用參數)
  ↓
Task 4 (質量檢查)
  ↓
Task 5 (端到端驗證)
  ↓
Task 6 (回歸測試, 可選)
```

## Success Criteria

- ✅ 所有 Pydantic 驗證錯誤消失 (完成)
- ✅ DCOP 轉換事件成功發送 (完成)
- ✅ Backend 可通過 study_uid 追溯事件 (完成)
- ✅ 代碼質量檢查全部通過 (完成)
- ✅ 端到端測試驗證成功 (完成)

**實施完成日期**: 2026-01-07

## Rollback Plan

如果修改導致問題:
1. 恢復函數簽名到原版本 (移除 study_uid 參數)
2. 恢復調用點 (移除 study_uid 傳遞)
3. 恢復 DCOPEventRequest 為 `study_uid=None`
4. 重新運行質量檢查確認穩定性

## Notes

- 此修復不影響其他調用 `_batch_convert_series_to_nifti()` 的代碼(目前只有一處調用)
- study_uid 總是存在於 `func_params` 中,無需處理 None 情況
- 此修復與 UID/Label 分離重構兼容,互不影響
