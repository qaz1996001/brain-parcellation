# 快速開始 - 移除任務 77167315-6579-475f-8609-0f65b9f06a66

## 🚀 最快速的方式 (3 步驟)

### 步驟 1: 進入專案目錄
```bash
cd /mnt/d/00_Chen/Task04_git
```

### 步驟 2: 確認環境設定
```bash
# 確保在正確的 Python 環境中 (uv 專案)
uv sync

# 腳本會自動使用 code_ai.load_dotenv() 載入 .env 檔案
# 確保 .env 檔案存在並包含必要的設定
```

### 步驟 3: 執行移除腳本
```bash
# 方式 A: 使用完整移除工具 ⭐ (最推薦 - 同時處理 RabbitMQ 和 Redis)
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66"

# 方式 B: 使用快速腳本 (僅 Redis)
uv run python scripts/quick_remove_task.py

# 方式 C: 使用 shell 腳本
./scripts/remove_task_77167315.sh
```

---

## 🔍 測試模式 (建議先執行)

在實際刪除前，先用測試模式查看會發生什麼:

### 方式 1: 使用完整工具的 dry-run (推薦)
```bash
# 測試模式 - 只顯示不刪除
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --dry-run

# 確認無誤後，移除 --dry-run 實際執行
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66"
```

### 方式 2: 修改 quick_remove_task.py
```python
# 打開檔案
vim scripts/quick_remove_task.py

# 找到配置區，確認設定:
TASK_ID = "77167315-6579-475f-8609-0f65b9f06a66"  # ✓ 正確
DRY_RUN = True  # ✓ 改為 True 測試

# 保存後執行
uv run python scripts/quick_remove_task.py
```

看到輸出正常後，再改回 `DRY_RUN = False` 執行實際刪除。

---

## ⚡ 使用 redis-cli 直接刪除 (最快)

如果您已經安裝了 `redis-cli`:

```bash
# 設定 Redis 連接資訊 (根據您的環境調整)
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export REDIS_DB=3  # RPC 結果通常在 DB 3

# 直接刪除
redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB \
  DEL "task_pipeline_inference_queue_result:77167315-6579-475f-8609-0f65b9f06a66"
```

返回 `(integer) 1` 表示刪除成功，`(integer) 0` 表示 key 不存在。

---

## 📋 檢查任務是否存在

```bash
# 方法 1: 使用 Python 腳本列出所有任務
uv run python scripts/remove_task_from_queue.py --list

# 方法 2: 使用 redis-cli
redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB \
  EXISTS "task_pipeline_inference_queue_result:77167315-6579-475f-8609-0f65b9f06a66"

# 方法 3: 查看任務內容
redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB \
  GET "task_pipeline_inference_queue_result:77167315-6579-475f-8609-0f65b9f06a66"
```

---

## 🔧 環境變數設定

確保已設定以下環境變數 (或使用預設值):

```bash
# 檢查當前環境變數
echo "REDIS_HOST: $REDIS_HOST"
echo "REDIS_PORT: $REDIS_PORT"
echo "REDIS_PASSWORD: $REDIS_PASSWORD"
echo "REDIS_DB_FILTER_AND_RPC_RESULT: $REDIS_DB_FILTER_AND_RPC_RESULT"

# 如果未設定，可以臨時設定:
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export REDIS_PASSWORD=your_password
export REDIS_DB_FILTER_AND_RPC_RESULT=3
```

---

## ✅ 執行結果說明

### 成功的輸出範例:
```
==============================================================
快速移除任務腳本
==============================================================
Redis: 127.0.0.1:6379 (DB: 3)
任務 ID: 77167315-6579-475f-8609-0f65b9f06a66
模式: 實際刪除
==============================================================

✓ Redis 連接成功

找到任務:
  Key: task_pipeline_inference_queue_result:77167315-6579-475f-8609-0f65b9f06a66
  TTL: 1234 秒 (20 分鐘)
  資料大小: 567 字元
  資料預覽: {"result": [...]}

正在刪除任務...

==============================================================
✓ 任務刪除成功!
==============================================================

任務移除完成
```

### 任務不存在的輸出:
```
✗ 任務不存在: task_pipeline_inference_queue_result:77167315-6579-475f-8609-0f65b9f06a66

正在搜尋所有相關任務...

找到 5 個任務:
  1. task_pipeline_inference_queue_result:12345678-1234-1234-1234-123456789012
      TTL: 1800s
  ...
```

---

## 🆘 遇到問題?

1. **Redis 連接失敗**: 檢查 Redis 是否運行 (`redis-cli ping`)
2. **任務不存在**: 可能已過期或 ID 錯誤，使用 `--list` 查看所有任務
3. **權限不足**: 檢查 Redis 密碼是否正確
4. **Python 環境**: 確保使用 `uv run` 或正確的虛擬環境

更多詳細說明請查看 `README_TASK_REMOVAL.md`

---

## 📞 相關命令參考

```bash
# 列出所有任務
uv run python scripts/remove_task_from_queue.py --list

# 測試模式刪除
uv run python scripts/remove_task_from_queue.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --dry-run

# 實際刪除
uv run python scripts/remove_task_from_queue.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66"

# 查看幫助
uv run python scripts/remove_task_from_queue.py --help
```

