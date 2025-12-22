# Scripts 目錄

本目錄包含用於管理和維護系統的各種腳本。

## 📁 任務移除工具

### 🎯 快速開始
移除任務 `77167315-6579-475f-8609-0f65b9f06a66`:

```bash
# 一鍵執行 (最簡單)
./scripts/remove_task_77167315.sh

# 或使用完整工具 (推薦)
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --dry-run  # 先測試
```

### 📚 文檔索引

| 文檔 | 說明 | 適合 |
|------|------|------|
| **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** | 實際使用範例 | ⭐ 新手必看 |
| **[QUICKSTART.md](QUICKSTART.md)** | 快速開始指南 | 快速上手 |
| [SUMMARY.md](SUMMARY.md) | 工具總結 | 了解全貌 |
| [README_TASK_REMOVAL.md](README_TASK_REMOVAL.md) | 詳細說明 | 深入理解 |

### 🛠️ 工具清單

| 腳本 | 功能 | 推薦度 |
|------|------|--------|
| `remove_task_complete.py` | 完整移除 (RabbitMQ + Redis) | ⭐⭐⭐⭐⭐ |
| `quick_remove_task.py` | 快速移除 (僅 Redis) | ⭐⭐⭐ |
| `remove_task_from_queue.py` | Redis 管理工具 | ⭐⭐⭐ |
| `remove_task_77167315.sh` | 一鍵執行腳本 | ⭐⭐⭐⭐ |

---

## 🚀 核心功能

### ✅ 完整的任務移除
- 從 RabbitMQ 佇列移除等待中的任務
- 從 Redis 移除 RPC 執行結果
- 從 Redis 移除推論快取

### ✅ 自動環境配置
- 使用 `code_ai.load_dotenv()` 自動載入 `.env`
- 無需手動設定環境變數

### ✅ 安全操作
- 測試模式 (`--dry-run`)
- 詳細的操作日誌
- 執行摘要

---

## 📖 使用場景

### 場景 1: 任務卡住需要重新執行
```bash
uv run python scripts/remove_task_complete.py \
  --task-id "YOUR_TASK_ID" \
  --dry-run
```

### 場景 2: 查看佇列狀態
```bash
uv run python scripts/remove_task_complete.py --list
```

### 場景 3: 清除特定 Study 的所有資料
```bash
uv run python scripts/remove_task_complete.py \
  --task-id "YOUR_TASK_ID" \
  --study-uid "STUDY_UID" \
  --study-id "STUDY_ID"
```

---

## 🔧 環境需求

### Python 依賴
```bash
# 已包含在 pyproject.toml 中
- redis
- pika (RabbitMQ 客戶端)
```

### 環境變數 (自動從 .env 載入)
```bash
# RabbitMQ
RABBITMQ_HOST=127.0.0.1
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASS=guest
RABBITMQ_VIRTUAL_HOST=/

# Redis
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_DB_FILTER_AND_RPC_RESULT=3
```

---

## 📊 任務資料結構

### RabbitMQ 佇列
- **佇列名稱**: `task_pipeline_inference_queue`
- **訊息格式**: JSON (funboost 格式)
- **內容**: 等待執行的任務參數

### Redis RPC 結果
- **Key 格式**: `task_pipeline_inference_queue_result:<UUID>`
- **DB**: 3
- **TTL**: 1800 秒 (30 分鐘)
- **內容**: 任務執行結果

### Redis 推論快取
- **Key 格式**: `inference_task:<study_uid>,<study_id>`
- **DB**: 0
- **TTL**: 2-6 小時
- **內容**: "queued" 或 "completed"

---

## 🎯 最佳實踐

1. ✅ **先測試**: 使用 `--dry-run` 確認操作
2. ✅ **完整移除**: 使用 `remove_task_complete.py` 確保清除乾淨
3. ✅ **查看日誌**: 檢查輸出確認移除成功
4. ✅ **驗證結果**: 使用 `--list` 確認任務已移除

---

## 🆘 需要幫助？

1. 查看 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) 了解實際使用方式
2. 查看 [QUICKSTART.md](QUICKSTART.md) 快速開始
3. 查看 [SUMMARY.md](SUMMARY.md) 了解工具全貌
4. 查看 [README_TASK_REMOVAL.md](README_TASK_REMOVAL.md) 深入了解

---

## 📝 更新日誌

- **2024-12-15**: 初版發布
  - 新增 `remove_task_complete.py` (RabbitMQ + Redis)
  - 新增 `quick_remove_task.py` (僅 Redis)
  - 新增 `remove_task_from_queue.py` (Redis 管理)
  - 新增 `remove_task_77167315.sh` (一鍵執行)
  - 所有工具支援 `code_ai.load_dotenv()` 自動載入環境變數
  - 支援測試模式 (`--dry-run`)
  - 完整的使用文檔

---

## 🎉 開始使用

```bash
# 最簡單的方式
cd /mnt/d/00_Chen/Task04_git
./scripts/remove_task_77167315.sh

# 或查看範例
cat scripts/USAGE_EXAMPLES.md
```

祝您使用愉快！🚀





