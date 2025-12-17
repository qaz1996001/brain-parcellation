"""
研究 (Study) 模組的 API 路由路徑定義。

此模組集中定義了所有與研究管理相關的 API 端點路徑。
使用集中管理的方式可以：
- 避免硬編碼路徑
- 簡化路由維護
- 提高代碼可讀性

API 路由結構
-----------
所有研究相關的端點都使用 /study 前綴。

Examples
--------
查詢研究事件列表：
    
    GET /study/list
    
詳細的查詢參數見 backend.app.study.routers 模組。

Notes
-----
路由路徑遵循 RESTful 設計原則：
- 使用複數名詞表示資源集合
- GET 用於查詢
- POST 用於創建（在 routers 中實現）
"""

# API 路由前綴
prefix = "/study"

# 查詢研究事件列表
# 支援多條件搜索、過濾、排序和分頁
STUDY_GET_LIST = f"{prefix}/list"
