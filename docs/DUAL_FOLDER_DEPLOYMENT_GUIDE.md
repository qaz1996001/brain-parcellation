# 雙資料夾部署指南 (Dual Folder Deployment Guide)

## 概述

本指南說明如何在**兩個獨立資料夾**部署 Production 和 Testing 環境，僅通過改變環境變數即可快速啟動。

## 架構分析

### 依賴關係圖

```
專案根目錄/
├── backend/              # FastAPI 後端服務
│   ├── app/
│   │   ├── config/      # ✅ 環境配置模組 (get_environment)
│   │   ├── database.py  # ✅ 使用 backend.app.config
│   │   ├── main.py      # ✅ 入口，使用 code_ai.load_dotenv()
│   │   └── server.py    # FastAPI 應用
│
├── code_ai/              # AI 推理任務模組
│   ├── __init__.py      # ✅ load_dotenv() 函數
│   ├── task/
│   │   ├── params.py    # ✅ 分布式控頻配置
│   │   └── task_pipeline.py  # Funboost 任務
│   ├── pipeline/        # ⚠️ 導入 backend 模組
│   └── utils/           # ⚠️ 導入 backend 模組
│
├── resource/             # ❌ 不依賴 backend 或 code_ai
│   └── script/          # 獨立腳本
│
├── funboost_cli_user.py  # ✅ Funboost CLI (使用 code_ai.load_dotenv)
├── docker-compose.yml    # ✅ 環境變數 ${ENV}
├── .env.production       # Production 配置
└── .env.testing          # Testing 配置
```

### 🚨 關鍵架構決策：必須共享 RabbitMQ

### 為什麼兩個環境必須使用同一個 RabbitMQ？

**設計原理**：Solution B 架構採用「統一隊列 + 分布式控頻」來防止 GPU 資源競爭

```yaml
✅ 正確架構 (Solution B):
  - 2 FastAPI instances (8000, 8001)
  - 2 Databases (dicom, dicom_testing)
  - 2 Funboost workers (separate processes)
  - 1 Unified Queue ← 關鍵！共享 RabbitMQ
  - 1 Distributed frequency control

分布式控頻工作原理:
  - qps_per_worker = qps / active_consumer_num
  - 例如：2 workers × (1/2) qps = 全局 1 qps ✅
  - GPU 安全：同時最多 1 個推理任務執行
```

**如果使用分離的 RabbitMQ 會發生什麼？** ❌

```yaml
錯誤架構:
  - Production worker → RabbitMQ:5672
  - Testing worker → RabbitMQ:5673
  - 兩個 worker 看不到彼此 ❌
  - 分布式控頻失效 ❌
  - 實際結果：2 workers × 1 qps = 2 qps 總計
  - GPU OOM 風險！❌
```

**正確配置**：
- 兩個環境使用相同的 `RABBITMQ_PORT=5672`
- 通過不同的 `RABBITMQ_VIRTUAL_HOST` 組織隔離（optional）
- 統一隊列名稱（不加環境後綴）

### 關鍵依賴關係

1. **backend → backend.app.config** ✅
   - `backend/app/database.py` 使用 `get_environment()`
   - 環境感知，支援 `ENV=production|testing`

2. **backend → code_ai** ✅
   - `backend/app/main.py` 使用 `code_ai.load_dotenv()`
   - 啟動時載入 `.env` 檔案

3. **code_ai → backend** ⚠️ 循環依賴
   - `code_ai/pipeline/*.py` 導入 `backend.app` 模組
   - `code_ai/utils/database.py` 導入 `backend.app.database`

4. **resource → 無依賴** ✅
   - `resource/` 目錄可獨立複製

## 兩資料夾部署方案

### 方案 A：符號連結共享 (推薦)

```bash
# 目錄結構
/home/david/
├── brain-parcellation/          # Production 資料夾
│   ├── backend/
│   ├── code_ai/
│   ├── resource/
│   ├── .env -> .env.production  # 符號連結
│   ├── .env.production
│   └── docker-compose.yml
│
└── brain-parcellation-testing/  # Testing 資料夾
    ├── backend -> ../brain-parcellation/backend/  # 符號連結
    ├── code_ai -> ../brain-parcellation/code_ai/  # 符號連結
    ├── resource -> ../brain-parcellation/resource/  # 符號連結
    ├── .env -> .env.testing       # 符號連結
    ├── .env.testing
    └── docker-compose.yml
```

**優點**：
- ✅ 節省磁碟空間 (共享 backend、code_ai、resource)
- ✅ 程式碼更新同步 (修改一次，兩環境生效)
- ✅ 首次部署快速 (< 1 分鐘)

**缺點**：
- ⚠️ 無法測試不同版本程式碼
- ⚠️ Windows 需要管理員權限建立符號連結

### 方案 B：完整複製

```bash
# 目錄結構
/home/david/
├── brain-parcellation/          # Production 資料夾
│   ├── backend/
│   ├── code_ai/
│   ├── resource/
│   ├── .env -> .env.production
│   └── docker-compose.yml
│
└── brain-parcellation-testing/  # Testing 資料夾 (完整複製)
    ├── backend/                 # 完整複製
    ├── code_ai/                 # 完整複製
    ├── resource/                # 完整複製
    ├── .env -> .env.testing
    └── docker-compose.yml
```

**優點**：
- ✅ 完全獨立，可測試不同版本
- ✅ 不需要特殊權限

**缺點**：
- ❌ 佔用雙倍磁碟空間
- ❌ 首次複製耗時 (視專案大小，約 2-5 分鐘)

## 快速部署步驟 (方案 A - 符號連結)

### 步驟 1: 準備 Production 資料夾

```bash
# 已存在的 Production 資料夾
cd /home/david/brain-parcellation

# 確認環境變數檔案
ls -la .env*
# 應該看到：.env.production, .env.testing

# 建立 .env 符號連結指向 production
ln -sf .env.production .env

# 確認
cat .env | head -1
# 應顯示：ENV=production
```

### 步驟 2: 建立 Testing 資料夾 (符號連結)

```bash
# 建立 Testing 資料夾
mkdir -p /home/david/brain-parcellation-testing
cd /home/david/brain-parcellation-testing

# 建立符號連結到 Production 的程式碼
ln -s ../brain-parcellation/backend ./backend
ln -s ../brain-parcellation/code_ai ./code_ai
ln -s ../brain-parcellation/resource ./resource
ln -s ../brain-parcellation/funboost_cli_user.py ./funboost_cli_user.py
ln -s ../brain-parcellation/funboost_config.py ./funboost_config.py

# 複製必要的獨立檔案
cp ../brain-parcellation/docker-compose.yml ./
cp ../brain-parcellation/.env.testing ./
cp ../brain-parcellation/.env.production ./
cp ../brain-parcellation/pyproject.toml ./
cp ../brain-parcellation/brain-parcellation-start.sh ./
cp ../brain-parcellation/brain-parcellation-stop.sh ./

# 建立 .env 符號連結指向 testing
ln -sf .env.testing .env

# 確認環境
cat .env | head -1
# 應顯示：ENV=testing
```

**⏱️ 預估時間**：< 1 分鐘

### 步驟 3: 驗證符號連結

```bash
cd /home/david/brain-parcellation-testing

# 檢查符號連結
ls -la
# 應該看到：
# backend -> ../brain-parcellation/backend
# code_ai -> ../brain-parcellation/code_ai
# .env -> .env.testing

# 驗證環境變數
grep "^ENV=" .env
# 輸出：ENV=testing

grep "^APP_PORT=" .env
# 輸出：APP_PORT=8001

grep "^RABBITMQ_PORT=" .env
# 輸出：RABBITMQ_PORT=5673
```

### 步驟 4: 啟動 Testing 環境

```bash
cd /home/david/brain-parcellation-testing

# 啟動 Docker 服務
docker-compose up -d

# 確認容器運行
docker ps
# 應該看到 3 個容器：rabbitmq_server, redis_server, db_server

# 啟動 Testing 環境
ENV=testing ./brain-parcellation-start.sh testing

# 檢查日誌
tail -f /var/log/brain-parcellation/startup.log
# 應該看到：
# [日期] Starting in testing environment
# [日期] ENV=testing
```

**⏱️ 預估時間**：
- Docker 容器啟動：30-60 秒
- Backend + Funboost 啟動：10-20 秒
- **總計**：約 1-2 分鐘

## 環境變數差異對照表

| 配置項 | Production (.env.production) | Testing (.env.testing) | 說明 |
|--------|------------------------------|------------------------|------|
| **ENV** | `production` | `testing` | 環境識別 |
| **APP_PORT** | `8000` | `8001` | 不同端口避免衝突 |
| **RABBITMQ_PORT** | `5672` | `5672` | ⭐ 共享同一個 RabbitMQ |
| **RABBITMQ_UI_PORT** | `15672` | `15672` | ⭐ 共享同一個管理界面 |
| **RABBITMQ_VIRTUAL_HOST** | `prod_vhost` | `test_vhost` | 不同 vhost 組織隔離 |
| **REDIS_PORT** | `6379` | `6379` | ⭐ 共享同一個 Redis |
| **REDIS_DB** | `2` | `4` | 不同 DB 數據隔離 |
| **REDIS_DB_FILTER_AND_RPC_RESULT** | `3` | `5` | 不同 DB 數據隔離 |
| **POSTGRES_DB** | `dicom` | `dicom_testing` | 不同資料庫名稱 |
| **PATH_PROCESS** | `D:/00_Chen/Task04_git/process` | `D:/00_Chen/Task04_git_test/process` | ⭐ 推理執行路徑 |
| **PATH_JSON** | `D:/00_Chen/Task04_git/json` | `D:/00_Chen/Task04_git_test/json` | ⭐ JSON 輸出路徑 |
| **PATH_LOG** | `D:/00_Chen/Task04_git/logs` | `D:/00_Chen/Task04_git_test/logs` | ⭐ 日誌文件路徑 |

## 任務路徑參數化配置 (Task Path Parameterization)

### 🆕 新架構：共享 GPU Worker 與路徑參數注入

從 2024-12-24 開始，系統支援**任務路徑參數化**，實現真正的雙環境共享 GPU Worker：

```yaml
新架構 (Parameter Injection):
  ✅ Production Backend (D:\00_Chen\Task04_git)
      ├─> .env: PATH_PROCESS=/prod/process
      ├─> 調用 get_task_execution_paths()
      └─> task_dict = {path_process: /prod/process, ...}
          └─> RabbitMQ Queue

  ✅ Testing Backend (D:\00_Chen\Task04_git_test)
      ├─> .env: PATH_PROCESS=/test/process
      ├─> 調用 get_task_execution_paths()
      └─> task_dict = {path_process: /test/process, ...}
          └─> RabbitMQ Queue

  ✅ GPU Worker (單一進程，可在任一資料夾啟動)
      ├─> 從 task_dict 參數讀取路徑 (不依賴自身 .env)
      ├─> 根據 task dispatcher 指定的路徑執行
      └─> Production 任務 → /prod 路徑
      └─> Testing 任務 → /test 路徑
```

### Path Parameterization Configuration

**New Architecture**: Starting from 2024-12-24, the system supports **task path parameterization**, enabling true dual-environment GPU worker sharing:

```yaml
New Architecture (Parameter Injection):
  ✅ Production Backend (D:\00_Chen\Task04_git)
      ├─> .env: PATH_PROCESS=/prod/process
      ├─> Calls get_task_execution_paths()
      └─> task_dict = {path_process: /prod/process, ...}
          └─> RabbitMQ Queue

  ✅ Testing Backend (D:\00_Chen\Task04_git_test)
      ├─> .env: PATH_PROCESS=/test/process
      ├─> Calls get_task_execution_paths()
      └─> task_dict = {path_process: /test/process, ...}
          └─> RabbitMQ Queue

  ✅ GPU Worker (Single process, can start from either folder)
      ├─> Reads paths from task_dict parameters (not its own .env)
      ├─> Executes in path specified by dispatcher
      └─> Production tasks → /prod paths
      └─> Testing tasks → /test paths
```

### 關鍵優勢 (Key Benefits)

| 特性 | 舊架構 (環境變數) | 新架構 (參數注入) |
|------|-------------------|-------------------|
| **GPU Worker 數量** | 2 個 (各環境一個) | ⭐ 1 個 (共享) |
| **資源效率** | 50% 利用率 | ⭐ 100% 利用率 |
| **路徑配置** | Worker 的 .env 決定 | ⭐ Backend 的 .env 決定 |
| **函數純度** | 依賴外部環境 (impure) | ⭐ 純函數 (pure) |
| **測試性** | 需要修改環境變數 | ⭐ 參數覆蓋即可 |
| **部署複雜度** | Worker 需配置環境 | ⭐ Worker 無需配置 |

### 配置範例 (Configuration Example)

**Production Backend (.env.production)**:
```bash
# 路徑配置 - Production 環境
PATH_PROCESS=D:/00_Chen/Task04_git/process
PATH_JSON=D:/00_Chen/Task04_git/json
PATH_LOG=D:/00_Chen/Task04_git/logs
```

**Testing Backend (.env.testing)**:
```bash
# 路徑配置 - Testing 環境
PATH_PROCESS=D:/00_Chen/Task04_git_test/process
PATH_JSON=D:/00_Chen/Task04_git_test/json
PATH_LOG=D:/00_Chen/Task04_git_test/logs
```

**GPU Worker**:
```bash
# GPU Worker 不再需要 PATH_* 環境變數
# 路徑從任務參數中獲取，由 Backend dispatcher 指定
```

### 程式碼實現 (Code Implementation)

**Backend Dispatcher** (自動注入路徑參數):
```python
from backend.app.config.task_paths import get_task_execution_paths

# 獲取當前環境的路徑配置
task_paths = get_task_execution_paths()

# 準備任務參數
params_data = {
    'nifti_study_path': '/data/nifti/study',
    'dicom_study_path': '/data/dicom/study',
    'path_process': task_paths['path_process'],  # 從 Backend .env
    'path_json': task_paths['path_json'],        # 從 Backend .env
    'path_log': task_paths['path_log']           # 從 Backend .env
}

# 發送到 RabbitMQ
task_pipeline_inference.push(params_data)
```

**GPU Worker Task Function** (從參數讀取路徑):
```python
def task_pipeline_inference(func_params: Dict):
    # 優先從參數讀取，fallback 到環境變數 (向後兼容)
    path_process = _extract_path_from_params(
        func_params, 'path_process', 'PATH_PROCESS'
    )
    path_json = _extract_path_from_params(
        func_params, 'path_json', 'PATH_JSON'
    )
    path_log = _extract_path_from_params(
        func_params, 'path_log', 'PATH_LOG'
    )

    # 在指定路徑執行推理任務
    # ...
```

### 監控與驗證 (Monitoring and Validation)

**檢查路徑路由 (Check Path Routing)**:
```bash
# 查看 Worker 日誌，確認路徑來源
tail -f logs/task_pipeline_inference_queue.log

# 應該看到 (如果從參數獲取):
# [INFO] Using path_process from task parameters: D:/00_Chen/Task04_git/process

# 或者看到 (如果 fallback 到環境變數):
# [WARNING] Path parameter 'path_process' not found, falling back to PATH_PROCESS
```

**驗證雙環境路徑隔離 (Verify Path Isolation)**:
```bash
# Production 推理應該在 Production 路徑
ls D:/00_Chen/Task04_git/json/
# 應該看到 Production 的 JSON 輸出

# Testing 推理應該在 Testing 路徑
ls D:/00_Chen/Task04_git_test/json/
# 應該看到 Testing 的 JSON 輸出
```

### 相關文檔 (Related Documentation)

- 📖 [任務路徑遷移指南 (Migration Guide)](./MIGRATION_TASK_PATHS.md)
- 📖 [API 參考文檔 (API Reference)](./API_REFERENCE.md)
- 📖 [架構決策記錄 (ADR)](./adr/ADR-TASK-PATH-PARAMETERIZATION.md)

### 常見問題 (FAQ)

**Q: 舊的 Worker 還能工作嗎？**

A: 可以！新架構保持向後兼容。如果任務參數中沒有路徑，Worker 會 fallback 到環境變數，並記錄警告日誌。

**Q: 需要重啟 Worker 嗎？**

A: 是的。更新程式碼後需要重啟 Worker 以啟用新的參數提取邏輯。

**Q: 如何測試路徑路由是否正確？**

A: 檢查輸出文件位置。Production 的輸出應該在 `D:/00_Chen/Task04_git/json/`，Testing 的應該在 `D:/00_Chen/Task04_git_test/json/`。

**Q: 能否動態切換路徑？**

A: 可以！使用 override 參數：
```python
test_paths = {
    'path_process': '/custom/test/process',
    'path_json': '/custom/test/json',
    'path_log': '/custom/test/logs'
}
paths = get_task_execution_paths(override=test_paths)
```

## 首次部署時間分析

### 情境 1：Production 已部署，新增 Testing (符號連結)

| 步驟 | 時間 |
|------|------|
| 建立 Testing 資料夾和符號連結 | < 1 分鐘 |
| 複製必要檔案 | < 30 秒 |
| 啟動 Docker Compose | 30-60 秒 |
| 啟動 Backend + Funboost | 10-20 秒 |
| **總計** | **約 2-3 分鐘** ✅ |

### 情境 2：Production 已部署，新增 Testing (完整複製)

| 步驟 | 時間 |
|------|------|
| 複製整個專案資料夾 | 2-5 分鐘 (視專案大小) |
| 修改 .env 符號連結 | < 10 秒 |
| 啟動 Docker Compose | 30-60 秒 |
| 啟動 Backend + Funboost | 10-20 秒 |
| **總計** | **約 3-7 分鐘** |

### 情境 3：全新部署 (Production + Testing)

| 步驟 | 時間 |
|------|------|
| 安裝依賴 (pyproject.toml) | 5-10 分鐘 (首次) |
| 下載 Docker 映像檔 | 2-5 分鐘 (首次) |
| Production 設定 + 啟動 | 1-2 分鐘 |
| Testing 設定 + 啟動 (符號連結) | 2-3 分鐘 |
| **總計** | **約 10-20 分鐘** (首次) |

## 依賴關係優化建議

### 問題：循環依賴 (code_ai ↔ backend)

```python
# code_ai/utils/database.py
from backend.app.database import ...  # ⚠️ 循環依賴

# backend/app/main.py
from code_ai import load_dotenv  # ⚠️ 循環依賴
```

### 解決方案：提取共享配置模組

建議創建 `shared/` 模組：

```
專案根目錄/
├── shared/              # 新增：共享配置模組
│   ├── __init__.py
│   ├── config.py        # load_dotenv, get_environment
│   └── constants.py     # 共享常數
│
├── backend/
│   └── app/
│       ├── main.py      # from shared import load_dotenv
│       └── database.py  # from shared.config import get_environment
│
└── code_ai/
    ├── __init__.py      # from shared import load_dotenv
    └── utils/
        └── database.py  # from shared.config import get_environment
```

**優點**：
- ✅ 解除循環依賴
- ✅ 符號連結更安全 (shared/ 也可符號連結)
- ✅ 程式碼組織更清晰

**實作時間**：約 30 分鐘

## 快速切換環境腳本

創建 `switch-env.sh`：

```bash
#!/bin/bash
# switch-env.sh - 快速切換環境變數

ENV_TARGET="${1:-production}"

if [[ "$ENV_TARGET" != "production" && "$ENV_TARGET" != "testing" ]]; then
    echo "❌ 無效環境。使用: production 或 testing"
    exit 1
fi

# 切換 .env 符號連結
ln -sf .env.${ENV_TARGET} .env

# 確認
echo "✅ 環境已切換至: $ENV_TARGET"
echo "當前配置："
grep "^ENV=" .env
grep "^APP_PORT=" .env
grep "^RABBITMQ_PORT=" .env
```

**使用方式**：

```bash
# 在任一資料夾執行
./switch-env.sh production  # 切換到 Production
./switch-env.sh testing     # 切換到 Testing
```

## 驗證清單

### Production 環境驗證

```bash
cd /home/david/brain-parcellation

# 1. 檢查環境變數
grep "^ENV=" .env
# 預期：ENV=production

# 2. 檢查端口
grep "^APP_PORT=" .env
# 預期：APP_PORT=8000

# 3. 檢查服務運行
curl http://localhost:8000/health
# 預期：{"status": "healthy"}

# 4. 檢查 Funboost Worker
ps aux | grep funboost_cli_user
# 應該看到運行的進程
```

### Testing 環境驗證

```bash
cd /home/david/brain-parcellation-testing

# 1. 檢查環境變數
grep "^ENV=" .env
# 預期：ENV=testing

# 2. 檢查端口
grep "^APP_PORT=" .env
# 預期：APP_PORT=8001

# 3. 檢查服務運行
curl http://localhost:8001/health
# 預期：{"status": "healthy"}

# 4. 檢查符號連結
readlink backend
# 預期：../brain-parcellation/backend
```

## 常見問題

### Q1: 符號連結在 Windows 上如何建立？

**A**: Windows 需要管理員權限：

```powershell
# 以管理員身份開啟 PowerShell
cd D:\brain-parcellation-testing

# 建立目錄符號連結
mklink /D backend ..\brain-parcellation\backend
mklink /D code_ai ..\brain-parcellation\code_ai

# 建立檔案符號連結
mklink .env .env.testing
```

### Q2: 如何避免端口衝突？

**A**: `.env.testing` 已配置不同端口：

- APP_PORT: 8000 → 8001
- RABBITMQ_PORT: 5672 → 5673
- RABBITMQ_UI_PORT: 15672 → 15673
- REDIS_PORT: 6379 → 6380

### Q3: 兩個環境可以同時運行嗎？

**A**: 可以！架構設計支援：

```bash
# Terminal 1: Production
cd /home/david/brain-parcellation
ENV=production ./brain-parcellation-start.sh production

# Terminal 2: Testing
cd /home/david/brain-parcellation-testing
ENV=testing ./brain-parcellation-start.sh testing
```

分布式控頻確保 GPU 不競爭：
- Production Worker: 0.5 qps
- Testing Worker: 0.5 qps
- 總計: 1.0 qps (GPU 安全) ✅

## 總結

### 推薦配置

- **方案**：符號連結 (方案 A)
- **首次部署時間**：< 3 分鐘
- **切換環境時間**：< 10 秒
- **磁碟空間**：節省 50%

### 關鍵要點

1. ✅ **僅需改變數**：透過 `.env` 符號連結實現
2. ✅ **快速部署**：符號連結建立 < 1 分鐘
3. ✅ **依賴關係清晰**：
   - resource: 獨立
   - code_ai: 依賴 backend.app.config
   - backend: 依賴 code_ai.load_dotenv
4. ✅ **GPU 安全**：分布式控頻防止資源競爭

### 下一步

1. 執行步驟 2 建立 Testing 資料夾
2. 驗證符號連結和環境變數
3. 啟動 Testing 環境並測試
4. (可選) 重構為 `shared/` 模組解除循環依賴
