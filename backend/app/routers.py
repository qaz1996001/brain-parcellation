"""
路由聚合模組 - 整合所有子模組的路由。

此模組是應用程式的路由聚合點，將所有子模組的路由整合到一個
主路由器中。使用 FastAPI 的標籤系統組織 API 文檔。

路由結構
--------
所有子模組的路由都通過此模組聚合：
- /api/v1/series/* : Series 管理相關路由
- /api/v1/rerun/* : 重跑任務相關路由
- /api/v1/sync/* : DICOM 同步相關路由
- /api/v1/study/* : Study 管理相關路由

Attributes
----------
router : APIRouter
    主路由器實例，包含所有子模組的路由。
    在 server.py 中通過 prefix="/api/v1" 掛載。

Notes
-----
路由組織原則：
    - 每個子模組使用獨立的標籤（tags）
    - 統一的前綴路徑（/api/v1）
    - 清晰的職責分離

TYPE_CHECKING 使用：
    用於類型檢查時避免循環導入，運行時不執行。

Examples
--------
在主應用程式中使用：

>>> from backend.app.routers import router
>>> from fastapi import FastAPI
>>> 
>>> app = FastAPI()
>>> app.include_router(router, prefix="/api/v1")
>>> # 所有子模組路由現在可通過 /api/v1/* 訪問

訪問特定模組的路由：

>>> # Series 路由: /api/v1/series/*
>>> # Sync 路由: /api/v1/sync/*
>>> # Rerun 路由: /api/v1/rerun/*
>>> # Study 路由: /api/v1/study/*

See Also
--------
backend.app.server : 應用程式主實例
backend.app.sync : DICOM 同步模組
backend.app.series : Series 管理模組
backend.app.rerun : 重跑任務模組
backend.app.study : Study 管理模組
"""

from typing import TYPE_CHECKING
from fastapi import APIRouter, Request

# TYPE_CHECKING 用於類型檢查時避免循環導入
# 運行時此塊不執行，避免實際導入
if TYPE_CHECKING:
    pass

# 導入所有子模組的路由器
from backend.app import series, sync, rerun, study

# 建立主路由器
router = APIRouter()

# 整合所有子模組的路由
# 每個模組使用獨立的標籤，便於 API 文檔組織
router.include_router(series.router, tags=["series"])
router.include_router(rerun.router, tags=["rerun"])
router.include_router(sync.router, tags=["sync"])
router.include_router(study.router, tags=["study"])


@router.post("/upload_json")
async def upload_json(request: Request):
    """
    上傳 JSON 資料的臨時端點。
    
    此端點用於接收和處理 JSON 格式的請求資料。
    目前為調試用途，直接打印接收到的資料。
    
    Parameters
    ----------
    request : Request
        FastAPI 請求物件，包含請求資料。
    
    Returns
    -------
    None
        目前不返回任何內容，僅打印資料。
    
    Notes
    -----
    此端點為臨時實現，生產環境應：
    - 添加資料驗證（Pydantic 模型）
    - 實現業務邏輯處理
    - 返回適當的響應
    - 添加錯誤處理
    - 記錄日誌而非使用 print
    
    Examples
    --------
    >>> curl -X POST http://localhost:8000/api/v1/upload_json \
    ...      -H "Content-Type: application/json" \
    ...      -d '{"key": "value"}'
    """
    # 解析請求中的 JSON 資料
    json_data = await request.json()
    
    # 調試輸出（生產環境應使用日誌記錄）
    print(json_data)
