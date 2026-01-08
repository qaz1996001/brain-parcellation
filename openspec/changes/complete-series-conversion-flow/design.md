# 設計文檔: 完整實現 Series Level 轉換流程

## Linus 式問題分解

### 第一層：資料結構分析

**核心資料流**：
```
raw_dicom (原始DICOM目錄)
    ↓ rename_dicom_file()
rename_dicom (重命名後的DICOM)
    ↓ dcm2niix
rename_nifti (NIfTI檔案)
    ↓ AI pipeline
inference_result (推論結果)
```

**資料所有權**：
- `raw_dicom`: 外部系統提供（Orthanc/平台）
- `rename_dicom`: 由轉換流程產生，儲存在 `PATH_RENAME_DICOM`
- `rename_nifti`: 由轉換流程產生，儲存在 `PATH_RENAME_NIFTI`
- `inference_result`: 由推論流程產生，上傳至平台

**DCOP Event 追蹤**：
```
SERIES_NEW → SERIES_TRANSFERRING → SERIES_TRANSFER_COMPLETE
    → SERIES_CONVERTING → SERIES_CONVERSION_COMPLETE
    → SERIES_INFERENCE_READY → SERIES_INFERENCE_QUEUED
    → SERIES_INFERENCE_RUNNING → SERIES_INFERENCE_COMPLETE
```

### 第二層：邊界案例識別

**目前的 if/else 分支**：

1. **Series 已轉換** (`SERIES_CONVERSION_COMPLETE` exists):
   - 直接模式：從 DCOP event 提取 `nifti_path`
   - 現有實現 ✅

2. **Series 未轉換** (只有 `SERIES_TRANSFER_COMPLETE`):
   - 轉換模式：從 DCOP event 提取 `raw_dicom_path`
   - 設置 `needs_conversion=True`
   - **這是缺失的部分** ❌

3. **Series 不存在** (無任何 DCOP event):
   - 拒絕請求，返回錯誤
   - 現有實現 ✅

**Linus 問題**: 這些分支是真實業務邏輯還是設計缺陷？

**答案**: 這是真實業務邏輯。Series 可能處於不同狀態，API 需要根據狀態選擇適當的處理方式。

### 第三層：複雜度審查

**功能本質（一句話）**：
> 根據 Series 狀態，選擇「直接推論」或「先轉換再推論」。

**目前解決方案使用的概念**：
1. DCOP event 狀態查詢
2. 參數驗證
3. 任務派發

**可以減半嗎？**
不行。這三個概念都是必要的，且已經是最小集合。

**複雜度評估**: 🟢 **可接受** - 每個概念都有明確用途

### 第四層：破壞性分析

**受影響的功能**：
| 功能 | 影響 |
|-----|------|
| 直接模式 API 調用 | 不影響 ✅ |
| Study Level 推論 | 不影響 ✅ |
| DCOP 狀態追蹤 | 不影響 ✅ |
| Worker 任務執行 | 不影響 ✅ |

**依賴關係**：
- `inference/service.py` 依賴 `task_pipeline.py` 的轉換邏輯
- 轉換邏輯已存在且測試通過 ✅

### 第五層：實用性驗證

**問題是否真實存在於生產環境？**
✅ 是。使用者希望透過 API 觸發完整流程，而非手動準備 NIfTI。

**受影響的使用者數量？**
所有使用 Series Level API 的 radax 客戶端。

**解決方案的複雜度是否匹配問題嚴重性？**
✅ 是。修改量極小（~45行），重用現有邏輯。

---

## 核心判斷

✅ **值得做**：這解決了真實問題，修改量小，向後兼容。

---

## 關鍵洞察

### 資料結構

**最關鍵的資料關係**：

```python
# Worker 期望的資料結構（已實現）
func_params = {
    # 模式 1: 直接
    'nifti_series_paths': [...],  # 已存在的 NIfTI 路徑

    # 模式 2: 轉換
    'needs_conversion': True,
    'raw_dicom_series_paths': [...],  # 原始 DICOM 路徑
}
```

**資料結構驅動行為** - Worker 根據欄位存在性決定流程，這是好設計。

### 複雜度

**可消除的複雜度**：無。目前設計已是最簡形式。

### 風險點

**最大風險**：`raw_dicom` 路徑不存在
- **緩解措施**: Worker 已有 `os.path.exists()` 檢查
- **失敗處理**: 返回明確錯誤訊息

---

## Linus 式解決方案

### Step 1: 簡化資料結構（不需要，已是最簡）

Worker 的資料結構設計已符合 Linus 原則。

### Step 2: 消除特殊案例（不需要，這是真實業務邏輯）

「已轉換」vs「未轉換」是真實的業務狀態，不是設計缺陷。

### Step 3: 用最笨但最清晰的方式實現

```python
# inference/service.py
async def validate_series_ready(self, study_uid: str, series_uids: List[str]):
    """查詢 Series 狀態並返回適當的參數"""

    for series_uid in series_uids:
        # 嘗試找 CONVERSION_COMPLETE (已轉換)
        complete_event = await self._find_event(
            series_uid, DCOPStatus.SERIES_CONVERSION_COMPLETE
        )

        if complete_event:
            # 直接模式
            nifti_path = self._extract_nifti_path(complete_event)
            yield {'mode': 'direct', 'nifti_path': nifti_path}
        else:
            # 嘗試找 TRANSFER_COMPLETE (未轉換但已傳輸)
            transfer_event = await self._find_event(
                series_uid, DCOPStatus.SERIES_TRANSFER_COMPLETE
            )

            if transfer_event:
                # 轉換模式
                raw_path = self._extract_raw_dicom_path(transfer_event)
                yield {'mode': 'convert', 'raw_dicom_path': raw_path}
            else:
                # 尚未準備好
                yield {'mode': 'rejected', 'reason': 'Series not transferred'}
```

### Step 4: 確保零破壞

| 檢查項 | 狀態 |
|-------|------|
| 直接模式 API 保持不變 | ✅ |
| 現有 Worker 邏輯不修改 | ✅ |
| DCOP 追蹤機制不變 | ✅ |
| 錯誤處理保持一致 | ✅ |

---

## 架構決策

### ADR-001: 使用狀態機模式處理 Series 生命週期

**狀態**: 已採用

**決策**: Series 狀態透過 DCOP events 追蹤，Service 層根據狀態選擇處理方式。

**原因**:
- 符合現有架構（DCOP 事件驅動）
- 無需新增資料庫表或欄位
- Worker 已實現對應的處理邏輯

### ADR-002: 參數注入而非環境變數

**狀態**: 已採用（延續現有設計）

**決策**: 轉換路徑 (`PATH_RENAME_DICOM`, `PATH_RENAME_NIFTI`) 透過任務參數傳遞。

**原因**:
- 支援雙重部署（Production/Testing 使用不同路徑）
- 符合 Pure Function 設計原則
- 與現有 `path_process`, `path_json`, `path_log` 模式一致

---

## 實現順序

1. **Config 層**: 新增 `PATH_RENAME_DICOM`, `PATH_RENAME_NIFTI` 到 `task_paths.py`
2. **Schema 層**: 新增 `needs_conversion`, `raw_dicom_series_paths` 欄位
3. **Service 層**: 增強 `validate_series_ready` 支援兩種模式
4. **測試**: 驗證兩種模式都能正確觸發
