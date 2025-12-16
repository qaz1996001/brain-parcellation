# 任務移除工具使用說明

本目錄包含用於管理和移除 `task_pipeline_inference` 佇列任務的腳本。

## 📋 腳本列表

### 1. `remove_task_complete.py` - 完整移除工具 ⭐ (最推薦)

**用途**: 同時處理 RabbitMQ 佇列和 Redis，完整移除任務

**功能**:
- ✅ 從 RabbitMQ 佇列中移除等待中的任務
- ✅ 從 Redis 中移除 RPC 結果
- ✅ 從 Redis 中移除推論快取
- ✅ 支援測試模式 (dry-run)
- ✅ 列出 RabbitMQ 佇列中的所有任務

**使用方式**:

```bash
# 完整移除任務（測試模式）
python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --dry-run

# 完整移除任務（實際執行）
python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66"

# 同時移除任務和推論快取
python scripts/remove_task_complete.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --study-uid "1.2.840.113619.xxx" \
  --study-id "STUDY123"

# 列出 RabbitMQ 佇列中的任務
python scripts/remove_task_complete.py --list
```

---

### 2. `quick_remove_task.py` - 快速移除工具 (僅 Redis)

**用途**: 快速移除 Redis 中的任務結果，適合單次操作

**注意**: 此工具只處理 Redis RPC 結果，不處理 RabbitMQ 佇列

**使用方式**:

```bash
# 1. 編輯腳本，修改 TASK_ID
vim scripts/quick_remove_task.py

# 2. 找到配置區，修改以下內容:
TASK_ID = "77167315-6579-475f-8609-0f65b9f06a66"  # 改成你要刪除的任務 ID
DRY_RUN = False  # True: 測試模式, False: 實際刪除

# 3. 執行腳本
python scripts/quick_remove_task.py
```

**優點**:
- ✅ 簡單易用，直接修改變數即可
- ✅ 自動搜尋和顯示所有任務
- ✅ 支援測試模式 (DRY_RUN)
- ✅ 顯示詳細的任務資訊

---

### 3. `remove_task_from_queue.py` - Redis 管理工具

**用途**: 管理 Redis 中的任務結果和快取

**注意**: 此工具只處理 Redis，不處理 RabbitMQ 佇列

**使用方式**:

#### 列出所有任務
```bash
python scripts/remove_task_from_queue.py --list
```

#### 移除指定任務 (測試模式)
```bash
python scripts/remove_task_from_queue.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --dry-run
```

#### 移除指定任務 (實際執行)
```bash
python scripts/remove_task_from_queue.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66"
```

#### 移除推論快取
```bash
python scripts/remove_task_from_queue.py \
  --study-uid "1.2.840.113619.xxx" \
  --study-id "STUDY123"
```

#### 同時移除任務和快取
```bash
python scripts/remove_task_from_queue.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --study-uid "1.2.840.113619.xxx" \
  --study-id "STUDY123"
```

---

## 🔍 任務 ID 格式說明

任務 ID 有兩種格式:

1. **完整格式**: `task_pipeline_inference_queue_result:77167315-6579-475f-8609-0f65b9f06a66`
2. **簡短格式**: `77167315-6579-475f-8609-0f65b9f06a66` (僅 UUID 部分)

兩種格式都可以使用，腳本會自動處理。

---

## 🔧 環境設定

腳本會自動使用 `code_ai.load_dotenv()` 從 `.env` 檔案載入環境變數。

### Redis 設定
```bash
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your_password
REDIS_DB=0                         # 推論快取存放的 DB
REDIS_DB_FILTER_AND_RPC_RESULT=3  # RPC 結果存放的 DB
```

### RabbitMQ 設定 (用於 remove_task_complete.py)
```bash
RABBITMQ_HOST=127.0.0.1
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASS=guest
RABBITMQ_VIRTUAL_HOST=/
```

**注意**: 所有腳本都會自動從專案根目錄的 `.env` 檔案載入設定，無需手動設定環境變數。

---

## 📊 任務類型和 Redis Key 說明

### 1. RPC 結果 Key
- **格式**: `task_pipeline_inference_queue_result:<UUID>`
- **DB**: `REDIS_DB_FILTER_AND_RPC_RESULT` (通常是 DB 3)
- **用途**: 儲存任務執行結果
- **TTL**: 1800 秒 (30 分鐘)

### 2. 推論快取 Key
- **格式**: `inference_task:<study_uid>,<study_id>`
- **DB**: `REDIS_DB` (通常是 DB 0)
- **用途**: 防止重複推論
- **TTL**: 
  - 佇列中 (queued): 21600 秒 (6 小時)
  - 完成 (completed): 7200 秒 (2 小時)

---

## 💡 使用場景

### 場景 1: 任務卡住需要重新執行
```bash
# 1. 先測試查看任務
python scripts/quick_remove_task.py  # DRY_RUN = True

# 2. 確認後實際刪除
python scripts/quick_remove_task.py  # DRY_RUN = False
```

### 場景 2: 清理特定 Study 的所有相關任務
```bash
# 移除任務結果和推論快取
python scripts/remove_task_from_queue.py \
  --task-id "77167315-6579-475f-8609-0f65b9f06a66" \
  --study-uid "1.2.840.113619.xxx" \
  --study-id "STUDY123"
```

### 場景 3: 查看佇列狀態
```bash
# 列出所有等待中的任務
python scripts/remove_task_from_queue.py --list
```

---

## ⚠️ 注意事項

1. **測試模式**: 第一次使用建議先用 `--dry-run` 或 `DRY_RUN=True` 測試
2. **備份**: 刪除前確認任務 ID 正確，刪除後無法恢復
3. **權限**: 確保有 Redis 的讀寫權限
4. **連線**: 確保能連接到 Redis 伺服器

---

## 🐛 疑難排解

### 問題 1: 連接 Redis 失敗
**解決方式**:
```bash
# 檢查 Redis 是否運行
redis-cli ping

# 檢查環境變數
echo $REDIS_HOST
echo $REDIS_PORT

# 手動測試連接
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping
```

### 問題 2: 任務不存在
**可能原因**:
- 任務 ID 錯誤
- 任務已經過期 (TTL 到期)
- 任務在不同的 Redis DB

**解決方式**:
```bash
# 列出所有任務確認
python scripts/remove_task_from_queue.py --list
```

### 問題 3: 刪除後任務仍然執行
**說明**: 刪除的是 Redis 中的結果快取，不會中斷正在執行的任務

如需停止正在執行的任務:
1. 找到執行任務的 worker 進程
2. 使用 `kill` 或重啟 worker

---

## 📚 相關檔案

- **任務定義**: `code_ai/task/task_pipeline.py`
- **佇列配置**: `code_ai/task/params.py`
- **Funboost 配置**: `funboost_config.py`
- **服務層**: `backend/app/sync/service.py`

---

## 🔗 快速參考

```bash
# 最快速的使用方式 (推薦給新手)
1. vim scripts/quick_remove_task.py
2. 修改 TASK_ID = "你的任務ID"
3. python scripts/quick_remove_task.py

# 進階使用 (命令列工具)
python scripts/remove_task_from_queue.py --help
```

---

## 📝 更新日誌

- **2024-12-15**: 初版發布
  - 新增 `quick_remove_task.py` 快速移除工具
  - 新增 `remove_task_from_queue.py` 完整命令列工具
  - 支援測試模式 (dry-run)
  - 支援列出所有任務
  - 支援移除推論快取

