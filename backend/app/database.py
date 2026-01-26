"""
資料庫配置模組 - SQLAlchemy 非同步 ORM 設置。

此模組配置 Advanced Alchemy（基於 SQLAlchemy 的 FastAPI 擴展），
提供非同步資料庫連接和會話管理。使用 PostgreSQL 作為主要資料庫。

配置說明
--------
- 連接字串: PostgreSQL + asyncpg 驅動
- 會話模式: 自動提交（autocommit）
- 會話配置: 提交後不過期物件（expire_on_commit=False）
- 自動建立: 啟用（create_all=True）

Attributes
----------
sqlalchemy_config : SQLAlchemyAsyncConfig
    SQLAlchemy 非同步配置實例，包含連接字串、會話配置等。
    
alchemy : AdvancedAlchemy
    Advanced Alchemy 實例，用於初始化 FastAPI 應用程式。
    提供依賴注入、儲存庫模式等功能。

Notes
-----
連接字串格式：
    postgresql+asyncpg://username:password@host:port/database
    
環境變數建議：
    生產環境應使用環境變數而非硬編碼連接字串。
    
會話管理：
    expire_on_commit=False 確保提交後物件仍可訪問，
    適合 FastAPI 的非同步場景。

Examples
--------
在 FastAPI 應用程式中使用：

>>> from backend.app.database import alchemy
>>> from fastapi import FastAPI
>>> 
>>> app = FastAPI()
>>> alchemy.init_app(app)
>>> # 現在可以在路由中使用依賴注入獲取資料庫會話

在服務類中使用：

>>> from backend.app.database import alchemy
>>> from backend.app.service import BaseRepositoryService
>>> 
>>> class MyService(BaseRepositoryService[MyModel]):
>>>     # 服務類自動使用 alchemy 配置
>>>     pass

See Also
--------
advanced_alchemy.extensions.fastapi.AdvancedAlchemy : Advanced Alchemy 主類
backend.app.service.BaseRepositoryService : 基礎服務類
"""

from advanced_alchemy.extensions.fastapi import (
    AdvancedAlchemy,
    AsyncSessionConfig,
    SQLAlchemyAsyncConfig,
)
from advanced_alchemy.extensions.starlette.config import EngineConfig

# SQLAlchemy 非同步配置
# 使用 PostgreSQL + asyncpg 驅動進行非同步資料庫操作
sqlalchemy_config = SQLAlchemyAsyncConfig(
    # 開發環境可選：SQLite 配置（已註解）
    # connection_string="sqlite+aiosqlite:///test.sqlite",
    
    # PostgreSQL 連接字串
    # 格式: postgresql+asyncpg://username:password@host:port/database
    connection_string="postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom",

    # 連接池配置：解決 QueuePool 連接耗盡問題
    engine_config=EngineConfig(
        pool_size=20,        # 基本連接數（預設 5）
        max_overflow=30,     # 額外連接數（預設 10）
        pool_timeout=60,     # 等待連接超時（秒）
        pool_pre_ping=True,  # 連接健康檢查
    ),

    # 會話配置：提交後物件不過期
    # 這確保提交後仍可訪問物件屬性，適合非同步場景
    session_config=AsyncSessionConfig(expire_on_commit=False),
    
    # 自動提交模式：每個操作自動提交
    # 簡化事務管理，適合大多數場景
    commit_mode="autocommit",
    
    # 自動建立表結構：根據模型自動建立資料庫表
    # 開發環境啟用，生產環境建議使用遷移工具
    create_all=True,
)

# Advanced Alchemy 實例
# 提供 FastAPI 集成、依賴注入、儲存庫模式等功能
alchemy = AdvancedAlchemy(
    config=sqlalchemy_config,
)
