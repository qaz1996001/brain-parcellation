# 任務移除工具 - 使用示範

## 🎯 您的任務

移除任務: `77167315-6579-475f-8609-0f65b9f06a66`

---

## 📋 方法對比

| 方法 | 處理範圍 | 難度 | 推薦 |
|------|---------|------|------|
| **方法 1: 完整工具** | RabbitMQ + Redis | ⭐ 簡單 | ✅ 最推薦 |
| 方法 2: 快速腳本 | 僅 Redis | ⭐ 簡單 | ⚠️ 不完整 |
| 方法 3: Shell 腳本 | RabbitMQ + Redis | ⭐ 簡單 | ✅ 推薦 |
| 方法 4: 命令列工具 | 僅 Redis | ⭐⭐ 中等 | ⚠️ 不完整 |

---

## ⭐ 方法 1: 完整工具 (最推薦)

### 步驟 1: 測試模式
```bash
cd /mnt/d/00_Chen/Task04_git

uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --dry-run
```

### 步驟 2: 查看輸出
```
==================================================================
完整任務移除工具
==================================================================
RabbitMQ: 127.0.0.1:5672 (vhost: /)
Redis RPC: 127.0.0.1:6379 (DB: 3)
Redis Cache: 127.0.0.1:6379 (DB: 0)
模式: 🔍 測試模式 (不實際刪除)
==================================================================

✓ RabbitMQ 連接成功

[1] 檢查 RabbitMQ 佇列: task_pipeline_inference_queue
----------------------------------------------------------------------
佇列中有 5 個任務

正在檢查佇列中的任務...

✓ 找到目標任務 (第 3 個)
  訊息預覽: {"function_result_status_id": "77167315-6579-475f-8609-0f65b9f06a66", ...}
  [DRY RUN] 將移除此任務

[2] 檢查 Redis RPC 結果 (DB 3)
----------------------------------------------------------------------
✓ 找到 RPC 結果:
  Key: task_pipeline_inference_queue_result:77167315-6579-475f-8609-0f65b9f06a66
  TTL: 1234 秒 (20 分鐘)
  資料大小: 567 字元
  [DRY RUN] 將刪除此 RPC 結果

==================================================================
執行摘要
==================================================================
[測試模式] 將移除 2 個項目
==================================================================
```

### 步驟 3: 實際執行
```bash
# 確認無誤後，移除 --dry-run
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66"
```

---

## ⚡ 方法 2: 快速腳本 (僅 Redis)

**警告**: 此方法只移除 Redis 結果，不處理 RabbitMQ 佇列！

### 步驟 1: 編輯腳本
```bash
vim scripts/quick_remove_task.py

# 修改以下內容:
TASK_ID = "77167315-6579-475f-8609-0f65b9f06a66"
DRY_RUN = True  # 先測試
```

### 步驟 2: 執行
```bash
uv run python scripts/quick_remove_task.py
```

### 步驟 3: 實際刪除
```bash
# 編輯腳本，改為 DRY_RUN = False
vim scripts/quick_remove_task.py

# 再次執行
uv run python scripts/quick_remove_task.py
```

---

## 🔧 方法 3: Shell 腳本 (一鍵執行)

```bash
cd /mnt/d/00_Chen/Task04_git

# 直接執行
./scripts/remove_task_77167315.sh
```

---

## 📊 方法 4: 命令列工具 (僅 Redis)

**警告**: 此方法只移除 Redis 結果，不處理 RabbitMQ 佇列！

```bash
# 列出所有任務
uv run python scripts/remove_task_from_queue.py --list

# 測試刪除
uv run python scripts/remove_task_from_queue.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --dry-run

# 實際刪除
uv run python scripts/remove_task_from_queue.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66"
```

---

## 🔍 查看佇列狀態

```bash
# 列出 RabbitMQ 佇列中的所有任務
uv run python scripts/remove_task_complete.py --list
```

輸出範例:
```
列出 RabbitMQ 佇列: task_pipeline_inference_queue
======================================================================

✓ RabbitMQ 連接成功
佇列中有 8 個任務

1. 任務:
   {
     "function_result_status_id": "12345678-1234-1234-1234-123456789012",
     "func_name": "task_pipeline_inference",
     ...
   }

2. 任務:
   {
     "function_result_status_id": "77167315-6579-475f-8609-0f65b9f06a66",
     ...
   }

...
```

---

## ✅ 驗證移除成功

### 1. 檢查 RabbitMQ 佇列
```bash
uv run python scripts/remove_task_complete.py --list
```

應該看不到目標任務 ID。

### 2. 檢查 Redis
```bash
# 使用 redis-cli
redis-cli -h $REDIS_HOST -p $REDIS_PORT -n 3 \
  EXISTS "task_pipeline_inference_queue_result:77167315-6579-475f-8609-0f65b9f06a66"
```

應該返回 `(integer) 0` 表示不存在。

---

## 🆘 疑難排解

### 問題: "任務不在 RabbitMQ 佇列中"

可能原因：
1. 任務已經被 worker 取走正在執行
2. 任務已經完成
3. 任務 ID 錯誤

解決方式：
```bash
# 1. 列出所有任務
uv run python scripts/remove_task_complete.py --list

# 2. 檢查 Redis 是否還有結果
uv run python scripts/remove_task_from_queue.py --list

# 3. 如果任務正在執行，等待完成或重啟 worker
```

### 問題: "連接失敗"

檢查步驟：
```bash
# 1. 確認 .env 檔案
cat .env | grep -E "REDIS|RABBITMQ"

# 2. 測試連接
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping
rabbitmqctl status

# 3. 確認服務運行
systemctl status redis
systemctl status rabbitmq-server
```

---

## 💡 進階使用

### 同時清除推論快取
```bash
uv run python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --study-uid "1.2.840.113619.2.xxx.xxx" \
  --study-id "STUDY_20241215_001"
```

### 批次清理多個任務
```bash
#!/bin/bash
TASK_IDS=(
  "77167315-6579-475f-8609-0f65b9f06a66"
  "88888888-8888-8888-8888-888888888888"
  "99999999-9999-9999-9999-999999999999"
)

for task_id in "${TASK_IDS[@]}"; do
  echo "處理任務: $task_id"
  uv run python scripts/remove_task_complete.py --task-id "$task_id"
  echo "---"
done
```

---

## 📚 相關文檔

- `SUMMARY.md` - 工具總結
- `README_TASK_REMOVAL.md` - 詳細說明
- `QUICKSTART.md` - 快速開始

---

## 🎓 小結

**推薦的工作流程**:

1. ✅ 使用 `remove_task_complete.py --dry-run` 測試
2. ✅ 確認輸出正確
3. ✅ 使用 `remove_task_complete.py` 實際執行
4. ✅ 使用 `--list` 驗證移除成功

**一鍵執行**:
```bash
./scripts/remove_task_77167315.sh
```

就這麼簡單！🎉



