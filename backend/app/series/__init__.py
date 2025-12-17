"""
Series 模組 - DICOM 序列分析和識別服務。

此模組提供 DICOM 序列（Series）的分析和識別功能，包括：
1. DICOM 序列類型識別
2. 影像方向分析（軸位、矢狀位、冠狀位）
3. 批次檔案處理
4. 多種輸入方式支援（檔案路徑、HTTP 上傳）

核心功能
--------
✨ **序列類型識別**
   - 結構影像序列（T1, T2, DWI, ADC 等）
   - 特殊序列（MRA, SWAN, eSWAN）
   - 灌注序列（DSC, ASL）
   - 功能序列（RESTING, CVR, DTI）

✨ **影像方向分析**
   - 自動識別軸位（AXI）、冠狀位（COR）、矢狀位（SAG）
   - 支援重格式化（REFORMATTED）序列
   - 識別對比增強（CE）序列

✨ **高效能處理**
   - 僅讀取 DICOM 標頭，不載入像素資料
   - 批次處理支援（最多 100 個檔案）
   - 記憶體友善的串流處理

模組結構
--------
- **routers.py**: HTTP API 端點定義
- **schemas.py**: 資料序列化方案和序列類型定義
- **urls.py**: API 路由路徑
- **deps.py**: 依賴注入配置

API 端點
--------
GET /series
    健康檢查端點

GET /series/types
    取得所有支援的序列類型列表

POST /series/dicom/analyze/by-path
    透過檔案路徑分析 DICOM 檔案

POST /series/dicom/analyze/by-upload
    透過 HTTP 上傳分析 DICOM 檔案

Examples
--------
取得所有支援的序列類型：

    >>> import httpx
    >>> async with httpx.AsyncClient() as client:
    ...     response = await client.get("/series/types")
    ...     types = response.json()
    ...     print(types)
    ['ADC', 'DWI0', 'DWI1000', 'T1_AXI', 'T1_COR', ...]

透過檔案路徑分析 DICOM：

    >>> file_paths = ["/path/to/scan1.dcm", "/path/to/scan2.dcm"]
    >>> response = await client.post(
    ...     "/series/dicom/analyze/by-path",
    ...     json=file_paths
    ... )
    >>> results = response.json()
    >>> for result in results:
    ...     print(f"{result['file_name']}: {result['series_type']}")

透過上傳分析 DICOM：

    >>> files = [
    ...     ("dicom_file_list", open("scan1.dcm", "rb")),
    ...     ("dicom_file_list", open("scan2.dcm", "rb"))
    ... ]
    >>> response = await client.post(
    ...     "/series/dicom/analyze/by-upload",
    ...     files=files
    ... )
    >>> results = response.json()

See Also
--------
backend.app.study : 研究級別的管理
backend.app.sync : DICOM 同步服務
code_ai.dicom2nii.convert : DICOM 轉換管理器

Notes
-----
此模組專注於 DICOM 序列的識別和分析，不涉及：
- 影像像素資料的處理
- DICOM 檔案的轉換
- 序列的儲存和管理

設計特點：
- 輕量級：僅讀取標頭資訊
- 高效能：批次處理和快取機制
- 易擴展：模組化的序列類型定義
"""

from .routers import router

__all__ = ["router"]
