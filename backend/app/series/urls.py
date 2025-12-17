"""
Series 模組的 API 路由路徑定義。

此模組集中定義了所有與 DICOM 序列分析相關的 API 端點路徑。
使用集中管理的方式可以：
- 避免硬編碼路徑
- 簡化路由維護
- 提高代碼可讀性
- 方便路徑變更

API 路由結構
-----------
所有序列相關的端點都使用 /series 前綴。

路由列表
--------
- GET /series
    健康檢查端點

- GET /series/types
    取得所有支援的序列類型列表

- POST /series/dicom/analyze/by-path
    透過檔案路徑分析 DICOM 檔案

- POST /series/dicom/analyze/by-upload
    透過 HTTP 上傳分析 DICOM 檔案

Examples
--------
健康檢查：

    GET /series

取得序列類型：

    GET /series/types

分析檔案（路徑）：

    POST /series/dicom/analyze/by-path
    Body: ["/path/to/file1.dcm", "/path/to/file2.dcm"]

分析檔案（上傳）：

    POST /series/dicom/analyze/by-upload
    Content-Type: multipart/form-data
    Files: dicom_file_list

Notes
-----
路由路徑遵循 RESTful 設計原則：
- 使用名詞表示資源（series）
- GET 用於查詢
- POST 用於創建/處理
- 路徑層級清晰（/series/dicom/analyze/by-path）
"""

# API 路由前綴
SERIES_GET_HEALTH_CHECK = "/series"

# 取得所有支援的序列類型
SERIES_GET_AVAILABLE_SERIES_TYPES = f"{SERIES_GET_HEALTH_CHECK}/types"

# 透過檔案路徑分析 DICOM 檔案
SERIES_ANALYZE_DICOM_FILES_BY_PATH = f"{SERIES_GET_HEALTH_CHECK}/dicom/analyze/by-path"

# 透過 HTTP 上傳分析 DICOM 檔案
SERIES_ANALYZE_DICOM_FILES_BY_UPLOAD = (
    f"{SERIES_GET_HEALTH_CHECK}/dicom/analyze/by-upload"
)
