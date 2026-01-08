# 任務清單: 完整實現 Series Level 轉換流程

## 前置條件

- [x] Worker 已實現 `needs_conversion` 模式
- [x] `_convert_single_series_to_nifti` 函數存在
- [x] `_batch_convert_series_to_nifti` 函數存在
- [x] DCOP 狀態碼已定義

---

## 階段 1: 配置層 (Config Layer)

### Task 1.1: 新增轉換路徑配置

**檔案**: `backend/app/config/task_paths.py`

**修改內容**:
```python
result = {
    "path_process": ...,
    "path_json": ...,
    "path_log": ...,
    "path_root": ...,
    # 新增
    "path_rename_dicom": paths.get("path_rename_dicom") or os.getenv("PATH_RENAME_DICOM"),
    "path_rename_nifti": paths.get("path_rename_nifti") or os.getenv("PATH_RENAME_NIFTI"),
}
```

**驗證**:
- [x] `uvx ty check backend/app/config/task_paths.py` ✅ All checks passed
- [x] 設置環境變數後可正確讀取

**依賴**: 無

---

## 階段 2: Schema 層 (Request/Response Models)

### Task 2.1: 擴充 SeriesInferenceRequest

**檔案**: `backend/app/inference/schemas.py`

**新增欄位**:
```python
class SeriesInferenceRequest(BaseModel):
    # 現有欄位...

    # 新增（可選）
    needs_conversion: Optional[bool] = Field(
        default=None,
        description="是否需要 DICOM 轉換。None=自動偵測, True=強制轉換, False=直接模式"
    )
    raw_dicom_series_paths: Optional[List[str]] = Field(
        default=None,
        description="原始 DICOM 路徑列表（needs_conversion=True 時必須）"
    )
```

**驗證**:
- [x] `uvx ty check backend/app/inference/` ✅ All checks passed
- [x] `uvx ruff check backend/app/inference/ --fix` ✅ Passed

**依賴**: 無

---

## 階段 3: Service 層 (Business Logic)

### Task 3.1: 新增 TRANSFER_COMPLETE 查詢方法

**檔案**: `backend/app/inference/service.py`

**新增方法**:
```python
async def _find_event_by_status(
    self, session, study_uid: str, series_uid: str, ope_no: str
) -> Optional[DCOPEventModel]:
    """查詢指定狀態的 DCOP 事件（通用方法）"""
```

**驗證**:
- [x] 方法可正確查詢 SERIES_TRANSFER_COMPLETE 和 SERIES_CONVERSION_COMPLETE

**依賴**: Task 1.1

---

### Task 3.2: 新增 raw_dicom 路徑提取方法

**檔案**: `backend/app/inference/service.py`

**新增方法**:
```python
def _extract_raw_dicom_path(self, event: DCOPEventModel) -> str:
    """從 TRANSFER_COMPLETE 事件提取 raw_dicom 路徑"""
```

**實際 result_data 格式** (from `task_dicom2nii.py:419-420`):
```python
result_data = {
    'raw_dicom_path': str(os.path.dirname(raw_dicom_path)),
    'rename_dicom_path': str(os.path.dirname(rename_dicom_path)),
}
```

**驗證**:
- [x] 路徑提取邏輯正確（優先從 result_data.raw_dicom_path 提取）
- [x] 2026-01-05: 更新方法以正確讀取 result_data.raw_dicom_path

**依賴**: Task 3.1

---

### Task 3.3: 擴充 validate_series_ready

**檔案**: `backend/app/inference/service.py`

**修改返回值**:
```python
async def validate_series_ready(
    self, study_uid: str, series_uids: List[str]
) -> Tuple[
    List[str],  # accepted_direct (已轉換)
    List[str],  # accepted_convert (需轉換)
    List[Dict[str, str]],  # rejected
    List[str],  # nifti_paths
    List[str],  # raw_dicom_paths
]:
```

**邏輯**:
1. 先查 CONVERSION_COMPLETE → 直接模式
2. 若無，查 TRANSFER_COMPLETE → 轉換模式
3. 若都無 → 拒絕

**驗證**:
- [x] 兩種模式都能正確識別
- [x] 混合狀態批次正確處理
- [x] 2026-01-05: 增強文件存在驗證 - 若 NIfTI 文件不存在，自動回退到轉換模式

**增強** (2026-01-05):
```python
# 驗證 NIfTI 文件實際存在
if nifti_path and os.path.exists(nifti_path):
    # Direct mode
else:
    # Fall through to conversion mode
    logger.warning(f"NIfTI file missing: {nifti_path}, trying conversion mode")

# 如果沒有任何事件，嘗試從配置推斷 raw_dicom 路徑
# Ken Thompson: "When in doubt, use brute force."
inferred_path = self._infer_raw_dicom_path(study_uid, series_uid)  # 修正: 傳入 series_uid
if inferred_path:
    # Inferred conversion mode
```

**重要修正** (2026-01-05):
- `_infer_raw_dicom_path` 需要同時接收 `study_uid` 和 `series_uid`
- 路徑結構: `{PATH_RAW_DICOM}/{study_uid}/{series_uid}` (series level)
- 之前錯誤只用 study_uid，導致轉換 study 下所有 DICOM

**依賴**: Task 3.1, Task 3.2

---

### Task 3.4: 修改 queue_series_inference 參數構建

**檔案**: `backend/app/inference/service.py`

**修改邏輯**:
```python
# 判斷是否需要轉換
if accepted_convert:
    func_params['needs_conversion'] = True
    func_params['raw_dicom_series_paths'] = raw_dicom_paths
    func_params['path_rename_dicom'] = task_paths['path_rename_dicom']
    func_params['path_rename_nifti'] = task_paths['path_rename_nifti']
else:
    func_params['nifti_series_paths'] = nifti_paths
```

**驗證**:
- [x] 直接模式參數正確
- [x] 轉換模式參數正確
- [x] 混合模式參數正確（同時包含兩種路徑）

**依賴**: Task 1.1, Task 2.1, Task 3.3

---

## 階段 4: 整合測試

### Task 4.1: API 測試 - 直接模式

**方法**: curl

**測試場景**: Series 已有 CONVERSION_COMPLETE

**預期結果**:
- status: "queued"
- func_params 包含 nifti_series_paths
- func_params 不包含 needs_conversion

**狀態**: ⏳ 待執行（需要實際測試環境）

**依賴**: Task 3.4

---

### Task 4.2: API 測試 - 轉換模式

**方法**: curl

**測試場景**: Series 只有 TRANSFER_COMPLETE

**預期結果**:
- status: "queued"
- func_params 包含 needs_conversion=true
- func_params 包含 raw_dicom_series_paths

**狀態**: ⏳ 待執行（需要實際測試環境）

**依賴**: Task 3.4

---

### Task 4.3: Worker 執行測試

**方法**: 檢查 worker 日誌

**測試場景**: 轉換模式任務執行

**預期結果**:
- raw_dicom → rename_dicom 成功
- rename_dicom → nifti 成功
- 推論執行成功

**狀態**: ⏳ 待執行（需要 Worker 環境）

**依賴**: Task 4.2

---

## 階段 5: Worker 命令生成

### Task 5.0: 修復 _build_series_inference_cmd 使用 PipelineConfig

**檔案**: `code_ai/task/task_pipeline.py`

**問題**: `_build_series_inference_cmd` 原本使用假設性路徑 `/workspace/models/{model_id}/`，
需要改用 `code_ai/pipeline/__init__.py` 中的 `pipelines` 配置。

**修改內容**:
1. 新增 `_resolve_model_id_to_inference_enum()` 函數：
   - 將 model_id (UUID 或字串) 映射到 InferenceEnum
   - 支持 UUID 映射（例如 `3fa85f64-5717-4562-b3fc-2c963f66afa6` → CMB）
   - 支持字串名稱（例如 'CMB', 'Aneurysm'）

2. 修改 `_build_series_inference_cmd()` 函數：
   - 使用 `pipelines[inference_enum].generate_cmd()` 生成命令
   - 新增 `study_id` 和 `path_root` 參數支持雙部署架構

**UUID 映射表** (位於 `_resolve_model_id_to_inference_enum`):
```python
MODEL_UUID_MAPPING = {
    # CMB (Cerebral Microbleed) - Swagger 示例 UUID
    '3fa85f64-5717-4562-b3fc-2c963f66afa6': InferenceEnum.CMB,
    # 可在此添加更多 UUID 映射
}
```

**驗證**:
- [x] 新函數使用正確的 pipeline 配置
- [x] UUID 和字串名稱都能正確映射
- [x] `path_root` 參數正確傳遞給 `PipelineConfig.generate_cmd()`
- [x] 類型檢查通過（除預先存在的類型問題外）
- [x] `uvx ruff format` 完成

**依賴**: Task 3.4

---

## 階段 6: 代碼品質

### Task 6.1: 類型檢查

```bash
uvx ty check backend/app/inference/
uvx ty check backend/app/config/task_paths.py
uvx ty check code_ai/task/task_pipeline.py
```

**結果**:
- [x] `backend/app/inference/` ✅ All checks passed!
- [x] `backend/app/config/task_paths.py` ✅ All checks passed!
- [x] `code_ai/task/task_pipeline.py` ✅ 新增函數無類型錯誤（預先存在的類型問題不影響）

**依賴**: 所有前置任務

---

### Task 6.2: Linting & Formatting

```bash
uvx ruff check backend/app/inference/ --fix
uvx ruff format backend/app/inference/
uvx ruff format code_ai/task/task_pipeline.py
```

**結果**:
- [x] `ruff check --fix` ✅ Found 1 error (1 fixed, 0 remaining)
- [x] `ruff format` ✅ 2 files reformatted, 5 files left unchanged
- [x] `code_ai/task/task_pipeline.py` ✅ 1 file reformatted

**依賴**: Task 6.1

---

## 總結

| 階段 | 任務數 | 狀態 |
|-----|--------|------|
| Config | 1 | ✅ 完成 |
| Schema | 1 | ✅ 完成 |
| Service | 4 | ✅ 完成 |
| 測試 | 3 | ⏳ 待執行 |
| Worker 命令生成 | 1 | ✅ 完成 |
| 品質 | 2 | ✅ 完成 |
| **總計** | **12** | **9/12 完成** |

**設計原則遵循**:
- ✅ 最小修改原則
- ✅ 重用現有 Worker 邏輯
- ✅ 向後兼容
- ✅ Fail Fast 驗證

**實作完成日期**: 2026-01-05

**更新紀錄**:
- 2026-01-05: 新增 Task 5.0 - 修復 `_build_series_inference_cmd` 使用 PipelineConfig
  - 新增 `_resolve_model_id_to_inference_enum()` 函數
  - 修改 `_build_series_inference_cmd()` 使用 `pipelines` 字典
  - UUID `3fa85f64-5717-4562-b3fc-2c963f66afa6` 現在正確映射到 CMB 模型
