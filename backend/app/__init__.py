"""
Backend 應用程式主模組。

此模組是 FastAPI 應用程式的根模組，提供應用程式的核心配置和初始化。
包含資料庫連接、路由聚合、伺服器設置等基礎設施。

模組結構
--------
- database.py: SQLAlchemy 非同步資料庫配置
- server.py: FastAPI 應用程式實例和生命週期管理
- routers.py: 路由聚合，整合所有子模組的路由
- service.py: 基礎服務類，提供統一的會話管理
- main.py: 應用程式入口點

子模組
------
- sync: DICOM 同步服務
- series: Series 管理服務
- rerun: 重跑任務服務
- study: Study 管理服務

Notes
-----
此模組遵循 FastAPI 最佳實踐：
- 使用 Advanced Alchemy 進行 ORM 管理
- 非同步資料庫操作
- 模組化路由設計
- 統一的服務基類

Examples
--------
從此模組導入應用程式實例：

>>> from backend.app.server import app
>>> # app 是配置完成的 FastAPI 實例

導入資料庫配置：

>>> from backend.app.database import alchemy
>>> # alchemy 是 AdvancedAlchemy 實例
"""

