# 任務移除工具總結

## ✅ 已創建的腳本

| 腳本名稱 | 功能 | 推薦度 | 說明 |
|---------|------|--------|------|
| `remove_task_complete.py` | **完整移除** | ⭐⭐⭐⭐⭐ | 同時處理 RabbitMQ + Redis，最完整 |
| `quick_remove_task.py` | 快速移除 Redis | ⭐⭐⭐ | 僅處理 Redis RPC 結果 |
| `remove_task_from_queue.py` | Redis 管理 | ⭐⭐⭐ | 功能完整的 Redis 管理工具 |
| `remove_task_77167315.sh` | Shell 快捷方式 | ⭐⭐ | 一鍵執行移除任務 |

---

## 🎯 核心特性

### ✅ 環境變數自動載入
所有腳本都使用 `code_ai.load_dotenv()` 自動從 `.env` 檔案載入設定：

```python
import code_ai
code_ai.load_dotenv()
```

### ✅ 完整的移除流程
`remove_task_complete.py` 會依序處理：

1. **RabbitMQ 佇列** - 移除等待中的任務
2. **Redis RPC 結果** - 移除任務執行結果
3. **Redis 推論快取** - 移除重複檢查快取

### ✅ 測試模式
所有工具都支援測試模式 (`--dry-run` 或 `DRY_RUN=True`)，確保安全操作。

---

## 📖 快速使用指南

### 情況 1: 任務卡在佇列中 (最常見)

```bash
# 使用完整工具 (推薦)
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --dry-run  # 先測試

# 確認後實際執行
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66"
```

### 情況 2: 只需要清除 Redis 結果

```bash
# 修改 quick_remove_task.py 中的 TASK_ID
vim scripts/quick_remove_task.py

# 執行
uv run python scripts/quick_remove_task.py
```

### 情況 3: 查看佇列狀態

```bash
# 列出 RabbitMQ 佇列中的所有任務
uv run python scripts/remove_task_complete.py --list
```

### 情況 4: 清除特定 Study 的推論快取

```bash
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --study-uid "1.2.840.113619.xxx" \
  --study-id "STUDY123"
```

---

## 🔧 配置檔案

所有腳本會自動從 `.env` 載入以下設定：

### RabbitMQ 設定
```bash
RABBITMQ_HOST=127.0.0.1
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASS=guest
RABBITMQ_VIRTUAL_HOST=/
```

### Redis 設定
```bash
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your_password
REDIS_DB=0                         # 推論快取
REDIS_DB_FILTER_AND_RPC_RESULT=3  # RPC 結果
```

---

## 📊 任務資料存放位置

### 1. RabbitMQ 佇列
- **位置**: `task_pipeline_inference_queue`
- **內容**: 等待執行的任務訊息
- **格式**: JSON (funboost 格式)

### 2. Redis RPC 結果
- **Key**: `task_pipeline_inference_queue_result:<UUID>`
- **DB**: `REDIS_DB_FILTER_AND_RPC_RESULT` (通常是 3)
- **TTL**: 1800 秒 (30 分鐘)
- **內容**: 任務執行結果

### 3. Redis 推論快取
- **Key**: `inference_task:<study_uid>,<study_id>`
- **DB**: `REDIS_DB` (通常是 0)
- **TTL**: 
  - 佇列中: 21600 秒 (6 小時)
  - 完成: 7200 秒 (2 小時)
- **內容**: "queued" 或 "completed"

---

## ⚙️ 技術細節

### RabbitMQ 操作原理
1. 連接到 RabbitMQ 佇列
2. 使用 `basic_get` 逐一取出訊息
3. 檢查訊息是否包含目標 task_id
4. 符合的訊息 `ack` 後不重新放回（刪除）
5. 不符合的訊息重新 `publish` 回佇列（保留）

### Redis 操作原理
1. 連接到對應的 Redis DB
2. 檢查 key 是否存在
3. 顯示 TTL 和資料預覽
4. 使用 `DELETE` 命令移除

---

## 🔍 疑難排解

### 問題 1: "任務不在 RabbitMQ 佇列中"
**可能原因**:
- 任務已經被 worker 取走正在執行
- 任務已經完成
- 任務 ID 錯誤

**解決方式**:
```bash
# 列出所有任務確認
uv run python scripts/remove_task_complete.py --list
```

### 問題 2: "Redis 連接失敗"
**檢查步驟**:
```bash
# 1. 檢查 .env 檔案
cat .env | grep REDIS

# 2. 測試 Redis 連接
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping
```

### 問題 3: "RabbitMQ 連接失敗"
**檢查步驟**:
```bash
# 1. 檢查 .env 檔案
cat .env | grep RABBITMQ

# 2. 檢查 RabbitMQ 服務狀態
systemctl status rabbitmq-server

# 3. 查看 RabbitMQ 管理介面
# http://localhost:15672 (預設帳密: guest/guest)
```

---

## 📚 相關文件

- `README_TASK_REMOVAL.md` - 詳細使用說明
- `QUICKSTART.md` - 快速開始指南
- `code_ai/task/task_pipeline.py` - 任務定義
- `funboost_config.py` - Funboost 配置

---

## 🚀 最佳實踐

1. **先用測試模式** - 使用 `--dry-run` 確認操作正確
2. **完整移除** - 使用 `remove_task_complete.py` 確保完全清除
3. **查看日誌** - 執行後檢查輸出確認移除成功
4. **清除快取** - 如有 study_uid/study_id，一併清除推論快取

---

## 📝 使用範例

### 範例 1: 完整移除流程
```bash
# 1. 先測試
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --dry-run

# 2. 確認後實際執行
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66"

# 3. 驗證已移除
uv run python scripts/remove_task_complete.py --list
```

### 範例 2: 批次清理
```bash
# 列出所有任務
uv run python scripts/remove_task_complete.py --list

# 逐一移除
for task_id in "id1" "id2" "id3"; do
  uv run python scripts/remove_task_complete.py --task-id "$task_id"
done
```

---

## ✨ 總結

您現在有一套完整的任務管理工具，可以：

✅ 從 RabbitMQ 佇列移除等待中的任務  
✅ 從 Redis 清除任務結果和快取  
✅ 使用測試模式確保安全操作  
✅ 查看佇列狀態和任務詳情  
✅ 自動載入環境變數  

建議優先使用 `remove_task_complete.py` 進行完整的任務移除！



