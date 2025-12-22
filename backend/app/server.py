"""
FastAPI 應用程式主模組 - 伺服器配置和初始化。

此模組建立和配置 FastAPI 應用程式實例，包括：
- 應用程式生命週期管理
- Redis 快取初始化
- CORS 中間件配置
- 路由註冊
- 資料庫集成

應用程式結構
------------
- 標題: SHH AI API
- 版本: 1.0.0
- API 前綴: /api/v1
- 快取: Redis (資料庫 6)
- CORS: 允許所有來源（開發環境）

生命週期管理
-----------
使用 lifespan 上下文管理器處理：
- 啟動: 初始化 Redis 快取
- 關閉: 清理資源（目前為空）

Attributes
----------
app : FastAPI
    配置完成的 FastAPI 應用程式實例。
    包含所有中間件、路由和生命週期處理。

Notes
-----
CORS 配置：
    目前允許所有來源（allow_origins=["*"]），
    生產環境應限制為特定域名。

Redis 配置：
    使用環境變數配置連接參數。
    預設使用資料庫 6 作為快取存儲。

路由前綴：
    所有路由通過 /api/v1 前綴訪問，
    便於版本管理和反向代理配置。

Examples
--------
直接使用應用程式實例：

>>> from backend.app.server import app
>>> # app 已配置完成，可直接使用

使用 Uvicorn 啟動：

>>> uvicorn backend.app.server:app --host 0.0.0.0 --port 8000

在測試中使用：

>>> from fastapi.testclient import TestClient
>>> from backend.app.server import app
>>> 
>>> client = TestClient(app)
>>> response = client.get("/api/v1/health")

See Also
--------
backend.app.database : 資料庫配置
backend.app.routers : 路由聚合
fastapi.FastAPI : FastAPI 主類
fastapi_cache : FastAPI 快取擴展
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

from .routers import router
from .database import alchemy


async def init_cache() -> None:
    """
    初始化 Redis 快取後端。
    
    此函數從環境變數讀取 Redis 連接參數，建立連接並初始化
    FastAPICache。用於應用程式啟動時的快取系統初始化。
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    
    Environment Variables
    ---------------------
    REDIS_HOST : str
        Redis 伺服器主機地址。
    REDIS_USERNAME : str
        Redis 認證用戶名。
    REDIS_PASSWORD : str
        Redis 認證密碼。
    REDIS_PORT : str
        Redis 伺服器端口。
    REDIS_DB_FASTAPI_CACHE : int, optional
        Redis 資料庫編號，預設 6。
    
    Notes
    -----
    連接字串格式：
        redis://username:password@host:port/database
    
    編碼配置：
        - encoding="utf8": 字串編碼
        - decode_responses=True: 自動解碼響應
    
    快取前綴：
        使用 "fastapi-cache" 作為鍵前綴，避免與其他應用衝突。
    
    Raises
    ------
    ConnectionError
        若無法連接到 Redis 伺服器。
    
    Examples
    --------
    在應用程式啟動時自動調用：
    
    >>> # 在 lifespan 中調用
    >>> await init_cache()
    >>> # Redis 快取現在可用於所有路由
    """
    # 從環境變數讀取 Redis 連接參數
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_USERNAME = os.getenv("REDIS_USERNAME")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_PORT = os.getenv("REDIS_PORT")
    REDIS_DB = os.getenv("REDIS_DB_FASTAPI_CACHE", 6)

    # 構建 Redis 連接 URL
    REDIS_URL = f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    # 建立 Redis 非同步連接
    redis = aioredis.from_url(
        REDIS_URL,
        encoding="utf8",        # 字串編碼
        decode_responses=True   # 自動解碼響應為字串
    )
    
    # 初始化 FastAPI 快取，使用 Redis 作為後端
    FastAPICache.init(
        RedisBackend(redis),
        prefix="fastapi-cache"  # 鍵前綴，避免衝突
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    應用程式生命週期管理器。
    
    此上下文管理器處理 FastAPI 應用程式的啟動和關閉事件。
    確保資源正確初始化和清理。
    
    Parameters
    ----------
    app : FastAPI
        FastAPI 應用程式實例。
    
    Yields
    ------
    None
        在應用程式運行期間保持上下文。
    
    Lifecycle Events
    ----------------
    Startup:
        1. 初始化 Redis 快取
        2. （可選）啟動任務調度器
    
    Shutdown:
        1. （可選）停止任務調度器
        2. 清理資源
    
    Notes
    -----
    使用 asynccontextmanager 裝飾器實現異步上下文管理器。
    yield 之前為啟動邏輯，之後為關閉邏輯。
    
    目前實現：
        - 啟動: 初始化快取
        - 關閉: 空（未來可添加清理邏輯）
    
    Examples
    --------
    在 FastAPI 應用程式中使用：
    
    >>> app = FastAPI(lifespan=lifespan)
    >>> # 應用程式啟動時自動執行啟動邏輯
    >>> # 應用程式關閉時自動執行關閉邏輯
    """
    # ========== 啟動事件 ==========
    # 初始化 Redis 快取系統
    await init_cache()
    
    # （可選）啟動任務調度器
    # asyncio.create_task(task_scheduler.start())
    
    # 應用程式運行期間
    yield
    
    # ========== 關閉事件 ==========
    # （可選）停止任務調度器
    # await task_scheduler.stop()


# FastAPI 應用程式實例
# 配置應用程式元資料、生命週期和基本設置
app = FastAPI(
    title="SHH AI API",           # API 標題（用於文檔）
    description="API for SHH AI",  # API 描述
    version="1.0.0",              # API 版本
    lifespan=lifespan,            # 生命週期管理器
    # root_path="/api/v1"         # 根路徑（可選，用於反向代理）
)

# CORS 中間件配置
# 允許跨域請求（開發環境配置，生產環境應限制）
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],          # 允許所有來源（生產環境應限制）
    allow_credentials=True,       # 允許憑證
    allow_methods=["*"],          # 允許所有 HTTP 方法
    allow_headers=["*"],          # 允許所有請求頭
)

# 註冊路由
# 所有子模組的路由通過 /api/v1 前綴訪問
app.include_router(router, prefix="/api/v1")

# 初始化資料庫集成
# 將 Advanced Alchemy 集成到 FastAPI 應用程式
alchemy.init_app(app)
