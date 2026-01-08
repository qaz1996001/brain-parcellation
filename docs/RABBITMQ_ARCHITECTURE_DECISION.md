# RabbitMQ 架構決策：共享 vs 分離

## ❓ 問題

**兩個環境要用同一個 RABBITMQ 還是不用？**

## ✅ 答案：必須共享同一個 RabbitMQ

## 🎯 決策理由

### 1. 設計文檔的明確要求

從 `openspec/changes/integrate-dual-deployment-gpu-solution/design.md` 的 **Solution B** 架構：

```yaml
架構組件：
- 2 FastAPI instances (8000, 8001)
- 2 Databases (dicom, dicom_testing)
- 2 Funboost workers (separate processes)
- 1 Unified Queue ← 關鍵！統一的 RabbitMQ
- 1 Distributed frequency control
```

### 2. 分布式控頻的工作原理

**code_ai/task/params.py:17-23**：
```python
is_using_distributed_frequency_control: bool = True

# 工作原理：
# qps_per_worker = qps / active_consumer_num
# 例如：2 workers × (1/2) qps = 全局 1 qps ✅
```

**關鍵機制**：
- Funboost 的分布式控頻依賴 **Redis** 來追蹤 active consumers
- 所有 workers 必須連接到**同一個 RabbitMQ 隊列**才能被 Redis 統計
- 如果 workers 連接不同的 RabbitMQ，Redis 無法統計總數

### 3. 分離 RabbitMQ 的嚴重後果 ❌

**錯誤配置**：
```bash
# .env.production
RABBITMQ_PORT=5672

# .env.testing
RABBITMQ_PORT=5673  # ← 啟動第二個 RabbitMQ！
```

**實際行為**：
1. Production worker 連接到 RabbitMQ:5672
2. Testing worker 連接到 RabbitMQ:5673
3. Redis 只能看到**每個 RabbitMQ 各自的 consumers**
4. 分布式控頻對每個 RabbitMQ 分別計算：
   - Production: 1 consumer → 1 qps
   - Testing: 1 consumer → 1 qps
5. **實際總 QPS = 2** (而非預期的 1)
6. **GPU OOM 風險** ❌

### 4. 正確配置的效果 ✅

**正確配置**：
```bash
# .env.production
RABBITMQ_PORT=5672
RABBITMQ_VIRTUAL_HOST=prod_vhost

# .env.testing
RABBITMQ_PORT=5672  # ← 共享同一個 RabbitMQ
RABBITMQ_VIRTUAL_HOST=test_vhost
```

**實際行為**：
1. Production worker 連接到 RabbitMQ:5672/prod_vhost
2. Testing worker 連接到 RabbitMQ:5672/test_vhost
3. **兩個 workers 都在同一個 RabbitMQ 實例**
4. Redis 可以統計到**總共 2 個 active consumers**
5. 分布式控頻生效：
   - 全局 qps = 1
   - 每個 worker: 1/2 qps
   - **實際總 QPS = 1** ✅
6. **GPU 安全** ✅

## 📋 已完成的配置修正

### 1. .env.production 修正

**修正前**：
```bash
RABBITMQ_DEFAULT_VHOST=prod_vhost
RABBITMQ_DEFAULT_USER=admin
RABBITMQ_DEFAULT_PASS=prod_password
RABBITMQ_PORT=5672
```

**修正後**：
```bash
# Funboost 需要的變數名稱
RABBITMQ_HOST=localhost
RABBITMQ_USER=admin
RABBITMQ_PASS=prod_password
RABBITMQ_VIRTUAL_HOST=prod_vhost
RABBITMQ_PORT=5672

# Docker Compose 需要的變數名稱
RABBITMQ_DEFAULT_VHOST=prod_vhost
RABBITMQ_DEFAULT_USER=admin
RABBITMQ_DEFAULT_PASS=prod_password

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=2  # funboost消息队列
REDIS_DB_FILTER_AND_RPC_RESULT=3  # 分布式控頻使用
```

### 2. .env.testing 修正

**修正前**：
```bash
RABBITMQ_PORT=5673  # ← 錯誤！啟動第二個 RabbitMQ
RABBITMQ_UI_PORT=15673
```

**修正後**：
```bash
# ⭐ 重要：兩個環境共享同一個 RabbitMQ (port 5672) 以啟用分布式控頻
RABBITMQ_HOST=localhost
RABBITMQ_USER=admin
RABBITMQ_PASS=test_password
RABBITMQ_VIRTUAL_HOST=test_vhost
RABBITMQ_PORT=5672  # ← 修正：使用相同的 port
RABBITMQ_UI_PORT=15672

# Docker Compose 需要的變數名稱
RABBITMQ_DEFAULT_VHOST=test_vhost
RABBITMQ_DEFAULT_USER=admin
RABBITMQ_DEFAULT_PASS=test_password

# Redis 配置（不同的 Redis 實例）
REDIS_HOST=localhost
REDIS_PORT=6380  # ← 不同的 port
REDIS_DB=2
REDIS_DB_FILTER_AND_RPC_RESULT=3
```

### 3. 隊列配置驗證 ✅

**code_ai/task/task_pipeline.py:23**：
```python
@Booster(BoosterParamsMyAI(queue_name='task_pipeline_inference_queue',
                           ...))
```

**確認**：
- ✅ 隊列名稱是硬編碼，**不加環境後綴**
- ✅ 兩個環境使用**統一隊列**
- ✅ 符合 Solution B 設計

## 🔍 技術細節

### vhost 的作用

**RabbitMQ Virtual Host (vhost)**：
- 類似於命名空間，用於邏輯隔離
- 每個 vhost 有獨立的 exchanges, queues, bindings
- **但是**：Funboost 的統一隊列策略不使用 vhost 隔離
- 在此架構中，vhost 僅用於**組織管理**，不影響分布式控頻

### Redis 的角色

**分布式控頻需要 Redis**：
```python
# funboost_config.py:28-29
# 如果@boost装饰器设置 is_using_distributed_frequency_control 为 True
# 则需要把 redis 连接配置好
```

**Redis 追蹤**：
- 記錄每個 worker 的心跳
- 統計 active consumers 數量
- 協調 QPS 分配

### ⚠️ 重要更新：Redis 也必須共享！

**之前的誤解**：
- 原本以為 Redis 可以分離
- Production: REDIS_PORT=6379
- Testing: REDIS_PORT=6380

**實際情況** ❌：
- Funboost 使用 Redis 追蹤 **active consumers**
- 如果使用分離的 Redis，每個 Redis 只能看到自己的 workers
- 分布式控頻失效：每個環境獨立計算 QPS
- 結果：2 workers × 1 qps = 2 qps 總計 → GPU OOM！

**正確配置** ✅：
```bash
# 兩個環境共享同一個 Redis
Production: REDIS_PORT=6379, REDIS_DB=2
Testing: REDIS_PORT=6379, REDIS_DB=4

# 通過不同的 REDIS_DB 隔離數據
Production:
  - REDIS_DB=2 (消息隊列)
  - REDIS_DB_FILTER_AND_RPC_RESULT=3
  - REDIS_DB_FASTAPI_CACHE=0

Testing:
  - REDIS_DB=4 (消息隊列)
  - REDIS_DB_FILTER_AND_RPC_RESULT=5
  - REDIS_DB_FASTAPI_CACHE=1
```

**工作原理**：
- Redis 用於**控頻協調**和**消息隊列存儲**
- 兩個 workers 連接到同一個 Redis（不同 DB）
- Redis 可以統計到總共 2 個 active consumers
- 分布式控頻生效：qps_per_worker = 1 / 2 = 0.5
- 實際總 QPS = 1 ✅

## 📊 資源隔離總結

| 資源 | Production | Testing | 隔離策略 | 原因 |
|------|-----------|---------|---------|------|
| **RabbitMQ** | :5672 | :5672 | ⭐ 共享（同port） | 分布式控頻必須 |
| **Redis** | :6379 | :6379 | ⭐ 共享（同port，不同DB） | 分布式控頻必須 |
| **PostgreSQL** | dicom | dicom_testing | 分離（不同DB名） | 資料隔離 |
| **FastAPI** | :8000 | :8001 | 分離（不同port） | 避免端口衝突 |

## ✅ 驗證步驟

### 啟動後驗證分布式控頻

```bash
# 1. 啟動 Production
cd /path/to/brain-parcellation
docker-compose up -d
ENV=production ./brain-parcellation-start.sh production

# 2. 啟動 Testing
cd /path/to/brain-parcellation-testing
ENV=testing ./brain-parcellation-start.sh testing

# 3. 檢查 RabbitMQ Management UI
# http://localhost:15672
# 應該看到：
# - prod_vhost 和 test_vhost 兩個 virtual hosts
# - task_pipeline_inference_queue 在兩個 vhost 中都存在
# - 每個隊列有 1 個 consumer（總共 2 個）

# 4. 發送測試任務並監控
# 應該觀察到：
# - 同時最多 1 個任務在執行
# - GPU 使用率穩定
# - 無 OOM 錯誤
```

## 🎓 學習要點

1. **分布式控頻的核心**：
   - 依賴統計 active consumers
   - 必須在同一個隊列系統中

2. **Virtual Host 的誤區**：
   - vhost 提供邏輯隔離
   - 但不是物理隔離
   - 不能替代統一隊列需求

3. **架構設計原則**：
   - 關鍵資源（RabbitMQ）：共享
   - 資料存儲（PostgreSQL）：隔離
   - 緩存資源（Redis）：可共享可隔離
   - 應用服務（FastAPI）：分離

## 📚 參考文檔

- `openspec/changes/integrate-dual-deployment-gpu-solution/design.md` - Solution B 架構設計
- `code_ai/task/params.py` - 分布式控頻配置
- `funboost_config.py` - Funboost 中間件配置
- `docs/DUAL_FOLDER_DEPLOYMENT_GUIDE.md` - 完整部署指南
