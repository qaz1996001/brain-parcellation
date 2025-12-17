"""
應用程式入口點 - 啟動 FastAPI 伺服器。

此模組是應用程式的主入口點，負責載入環境變數並啟動 Uvicorn ASGI 伺服器。
使用模組字符串引用應用程式實例，避免循環導入問題。

工作流程
--------
1. 載入環境變數（.env 檔案）
2. 讀取應用程式端口配置
3. 啟動 Uvicorn 伺服器
4. 載入 FastAPI 應用程式實例

配置
----
- 主機: 0.0.0.0 (監聽所有網路介面)
- 端口: 由環境變數 APP_PORT 決定，預設 8000
- 重載: 禁用（生產環境建議）

環境變數
--------
APP_PORT : int, optional
    應用程式監聽端口，預設 8000。
    例如: APP_PORT=8080

Notes
-----
使用模組字符串而非直接導入：
    uvicorn.run("backend.app.server:app", ...)
    
優點：
    - 避免循環導入
    - 延遲載入應用程式
    - 更好的錯誤處理

生產環境建議：
    - 使用 Gunicorn + Uvicorn workers
    - 配置反向代理（Nginx）
    - 啟用日誌記錄
    - 設置健康檢查端點

Examples
--------
直接執行：

>>> python -m backend.app.main
# 或
>>> python backend/app/main.py

使用環境變數：

>>> APP_PORT=8080 python -m backend.app.main
# 應用程式將在端口 8080 啟動

使用 Uvicorn CLI：

>>> uvicorn backend.app.server:app --host 0.0.0.0 --port 8000

See Also
--------
backend.app.server : FastAPI 應用程式實例
uvicorn : ASGI 伺服器實現
"""

import os
import uvicorn
from code_ai import load_dotenv


if __name__ == "__main__":
    # 載入環境變數（從 .env 檔案或系統環境）
    load_dotenv()
    
    # 讀取應用程式端口配置，預設 8000
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    
    # 啟動 Uvicorn ASGI 伺服器
    # 使用模組字符串引用應用程式，避免循環導入
    uvicorn.run(
        "backend.app.server:app",  # 應用程式模組路徑
        host="0.0.0.0",            # 監聽所有網路介面
        port=APP_PORT,             # 應用程式端口
        reload=False               # 禁用自動重載（生產環境）
    )
