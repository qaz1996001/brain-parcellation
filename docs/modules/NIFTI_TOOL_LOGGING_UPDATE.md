# NIFTI Tool Logging 與 Multi-output 邏輯更新說明

## 2025-12-16 更新摘要

### 1. 專用日誌記錄 (Dedicated Logging)
為了避免 NIFTI 轉換邏輯的日誌被淹沒在系統主日誌中，我們實作了獨立的日誌記錄器。

*   **日誌檔案位置**: `logs/YYYY-MM-DD.0001.DCOPEventDicomService-nifti_tool_get_series_info.log`
*   **記錄內容**: 包含完整的執行追蹤，從 Multi-output 檢測、路徑提取、檔案系統掃描、任務建立到資料庫寫入。
*   **優點**: 開發者可以專注於 NIFTI 轉換流程的除錯，不受其他服務干擾。

### 2. Multi-output Series (DWI) 處理邏輯增強
針對 DWI 等需要產生多個輸出 (如 DWI0, DWI1000) 的 Series，我們發現資料庫 `result_data` 有時可能資訊不完整（例如只記錄了其中一個路徑），導致轉換任務遺漏。

**修正方案**:
1.  **Series Description 提取增強 (New)**:
    *   當資料庫中的 `result_data` 缺少明確的 `series_description` (key: `result`) 時，系統會自動嘗試從 `rename_dicom_path` 解析最後一層目錄名稱作為替補。
    *   這解決了因缺少描述而導致無法識別 Multi-output Series (如 DWI) 的問題。
2.  **檔案系統掃描 (File System Scan)**: 當檢測到 Multi-output Series (如 DWI) 時，系統不再僅依賴資料庫記錄。
    *   **自動搜尋**: 系統會自動掃描 `rename_dicom_path` 的目錄及其父目錄，尋找符合 `DWI0`, `DWI1000` 命名規則的資料夾。
    *   **確保完整性**: 只要磁碟上存在對應的資料夾，系統就會建立對應的轉換任務，解決了 "只隨機轉換一筆" 的問題。

### 3. 其他錯誤修復
*   **AttributeError 修復**: 修正了日誌中存取 `DCOPEventModel` 時使用錯誤屬性名稱 (`id` -> `VsPrimaryKey`) 的問題。
*   **重複路徑過濾**: 修正了 Single-output Series 會重複提取相同路徑並發出警告的問題。

## 如何解讀日誌

打開 `logs/` 下的最新日誌檔案，您可以搜尋以下關鍵字：

*   `[NIFTI_TOOL]`: 流程開始與結束
*   `[SERIES]`: 每個 Series 的處理細節
*   `[DETECTION]`: Multi-output 檢測結果
*   `🔀 Multi-output series detected!`: 表示進入多重輸出處理流程
*   `File System Scan Results`: 顯示從磁碟掃描到的實際輸出資料夾
*   `[QUEUE]`: 任務發送到佇列的狀態

### 範例：DWI 處理成功

```text
[SERIES 10] ⚠️ 未找到明確的 series_description，使用路徑名稱作為替補: 'DWI0'
[SERIES 10] Multi-output 檢測: is_multi=True, required_outputs=['DWI0', 'DWI1000']
[SERIES 10] 🔀 Multi-output series detected! Checking file system for outputs...
[SERIES 10]    - Scanning directories: [...]
[SERIES 10]    - ✅ Found DWI0 on disk: /path/to/study/DWI0
[SERIES 10]    - ✅ Found DWI1000 on disk: /path/to/study/DWI1000
[SERIES 10] 🔀 File System Scan Results: Found 2/2 outputs
[SERIES 10] [OUTPUT 1/2] 建立 NIFTI 任務...
[SERIES 10] [OUTPUT 2/2] 建立 NIFTI 任務...
```
