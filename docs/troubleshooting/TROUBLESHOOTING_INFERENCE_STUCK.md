# 🏥 Study 300.50 推理卡住問題排查指南

## 📋 問題描述

Study 狀態進入 `300.50 (STUDY_INFERENCE_READY)` 後無法繼續進行 AI 推理，需要手動執行 `sudo systemctl restart brain-parcellation.service` 才能恢復。

## 🔍 根本原因分析

### 1. **Redis 快取鎖機制問題** (主要原因)

```python
# backend/app/sync/service.py 行 1052-1059
inference_task_key = f"inference_task:{study_uid},{study_id}"
if await redis_client.get(inference_task_key):
    # ❌ 如果 key 存在就跳過，即使推理失敗也會保留 6 小時
    continue
```

**影響**：
- Redis key 設定後保留 6 小時 (`ex=21600`)
- 如果推理失敗或 consumer 崩潰，key 仍然存在
- 新的推理請求被錯誤地跳過
- 需要等待 6 小時 key 過期，或手動刪除 key

### 2. **Subprocess 無超時控制**

```python
# code_ai/task/task_pipeline.py 行 94
stdout, stderr = process.communicate()  # ❌ 無超時，可能永久阻塞
```

**影響**：
- 如果推理命令卡住（GPU 記憶體不足、模型載入失敗）
- 整個 funboost consumer 被阻塞
- 佇列中的其他任務無法處理

### 3. **Funboost Consumer 崩潰無自動恢復**

**影響**：
- Consumer 進程崩潰後不會自動重啟
- RabbitMQ 佇列中的任務累積但無人處理
- 需要手動重啟服務

## 🛠️ 已實施的修復

### ✅ 修復 1: 智能 Redis 快取管理

```python
# backend/app/sync/service.py
cached_value = await redis_client.get(inference_task_key)
if cached_value:
    ttl = await redis_client.ttl(inference_task_key)
    # 如果剩餘時間少於 1 小時，強制重新執行
    if ttl > 18000:  # 大於 5 小時
        logger.info(f"Skipping duplicate, TTL: {ttl}s")
        continue
    else:
        logger.warning(f"TTL low ({ttl}s), forcing retry")
        await redis_client.delete(inference_task_key)

# 縮短快取時間為 1 小時
await redis_client.set(inference_task_key, "queued", ex=3600)
```

**優點**：
- 自動檢測卡住的任務（TTL 異常低）
- 強制重試而非永久跳過
- 快取時間從 6 小時縮短為 1 小時

### ✅ 修復 2: Subprocess 超時控制

```python
# code_ai/task/task_pipeline.py
try:
    # 設定 30 分鐘超時
    stdout, stderr = process.communicate(timeout=1800)
    logger.info("Command completed")
except subprocess.TimeoutExpired:
    logger.error("Command timeout (30min)")
    process.kill()
    # 記錄錯誤但繼續執行，不阻塞整個 consumer
```

**優點**：
- 防止單一任務永久阻塞 consumer
- 超時任務被終止但不影響其他任務
- 記錄詳細錯誤日誌便於排查

### ✅ 修復 3: 任務完成後清理快取

```python
# code_ai/task/task_pipeline.py
# 任務完成後清理 Redis 快取，允許重新執行
if redis_client and inference_task_key:
    try:
        redis_client.delete(inference_task_key)
        logger.info(f"Cleared Redis cache key: {inference_task_key}")
    except Exception as e:
        logger.warning(f"Failed to clear Redis cache: {e}")
```

**優點**：
- 任務完成立即清理快取
- 避免快取累積
- 允許立即重新執行（如果需要）

## 🚀 快速修復指令

### 方案 A: 使用快速修復腳本（推薦）

```bash
# 給予執行權限
chmod +x scripts/quick_fix_stuck.sh

# 執行快速修復
./scripts/quick_fix_stuck.sh
```

腳本會自動：
1. ✅ 檢查 funboost consumer 狀態
2. ✅ 檢查 RabbitMQ 佇列
3. ✅ 檢查 Redis 卡住的 key
4. ✅ 提供清理選項
5. ✅ 顯示最近日誌

### 方案 B: 使用完整診斷工具

```bash
# 診斷特定 study
python scripts/diagnose_inference_stuck.py \
    --study-uid <STUDY_UID> \
    --study-id <STUDY_ID>

# 清理 Redis 快取 key
python scripts/diagnose_inference_stuck.py \
    --cleanup <STUDY_UID> <STUDY_ID>
```

### 方案 C: 手動修復步驟

#### 1. 檢查 funboost consumer 狀態

```bash
# 查看進程
ps aux | grep funboost

# 查看服務狀態
sudo systemctl status brain-parcellation.service

# 重啟服務
sudo systemctl restart brain-parcellation.service
```

#### 2. 檢查 Redis 快取

```bash
# 連接 Redis
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD

# 列出所有推理任務 key
SCAN 0 MATCH inference_task:*

# 檢查特定 key 的 TTL
TTL inference_task:<STUDY_UID>,<STUDY_ID>

# 刪除卡住的 key
DEL inference_task:<STUDY_UID>,<STUDY_ID>

# 清除所有推理任務 key（謹慎使用）
redis-cli --scan --pattern "inference_task:*" | xargs redis-cli DEL
```

#### 3. 檢查 RabbitMQ 佇列

```bash
# 列出所有佇列
rabbitmqadmin list queues

# 檢查推理佇列狀態
rabbitmqadmin get queue=task_pipeline_inference_queue count=10

# 清空佇列（謹慎使用）
rabbitmqadmin purge queue=task_pipeline_inference_queue
```

#### 4. 查看日誌

```bash
# 實時查看日誌
sudo journalctl -u brain-parcellation.service -f

# 查看最近 100 行日誌
sudo journalctl -u brain-parcellation.service -n 100

# 搜尋特定 study 的日誌
sudo journalctl -u brain-parcellation.service | grep <STUDY_ID>
```

## 📊 診斷清單

使用此清單來診斷問題：

- [ ] **Funboost Consumer 運行中？**
  ```bash
  ps aux | grep funboost
  ```

- [ ] **RabbitMQ 佇列有消費者？**
  ```bash
  rabbitmqadmin list queues name messages consumers
  ```

- [ ] **Redis 有卡住的 key？**
  ```bash
  redis-cli --scan --pattern "inference_task:*"
  ```

- [ ] **資料庫狀態正確？**
  ```sql
  SELECT ope_no, create_time FROM dcop_event_bt 
  WHERE study_uid = '<STUDY_UID>' 
  ORDER BY create_time DESC LIMIT 5;
  ```

- [ ] **GPU 記憶體是否充足？**
  ```bash
  nvidia-smi
  ```

- [ ] **磁碟空間是否充足？**
  ```bash
  df -h
  ```

## 🔄 預防措施

### 1. 定期清理 Redis 快取

建立 cron job 定期清理過期的快取 key：

```bash
# 編輯 crontab
crontab -e

# 每小時清理 TTL 小於 1800 秒的 key
0 * * * * /path/to/scripts/cleanup_redis_cache.sh
```

### 2. 監控 Funboost Consumer

使用 systemd 的 `Restart=on-failure` 確保自動重啟（已配置）：

```ini
[Service]
Restart=on-failure
RestartSec=10
```

### 3. 設定告警

監控以下指標並設定告警：

- RabbitMQ 佇列長度 > 10
- Funboost consumer 進程消失
- Redis key 數量異常增長
- GPU 記憶體使用率 > 90%

## 📝 常見問題 (FAQ)

### Q1: 為什麼重啟服務後就正常了？

**A**: 重啟會：
1. 清空 funboost consumer 的內部狀態
2. 重新建立 RabbitMQ 連接
3. 重新載入模型到 GPU
4. 但 **不會** 清除 Redis 快取（這是修復的重點）

### Q2: Redis key 會自動過期嗎？

**A**: 會的，但：
- 原本設定 6 小時過期（21600 秒）
- 修復後改為 1 小時過期（3600 秒）
- 如果任務卡住超過 1 小時，仍需手動清理

### Q3: 修復後還會卡住嗎？

**A**: 機率大幅降低，因為：
- ✅ 智能檢測卡住的 key 並強制重試
- ✅ Subprocess 有超時控制（30 分鐘）
- ✅ 任務完成後自動清理快取
- ✅ 快取時間縮短為 1 小時

但仍可能在以下情況卡住：
- GPU 記憶體不足導致 OOM kill
- 網路問題導致無法存取檔案系統
- 資料庫連接斷開

### Q4: QPS=1 會不會太慢？

**A**: 這是根據 GPU 能力設定的：
- 單 GPU 同時處理多個深度學習任務會 OOM
- QPS=1 確保序列處理，避免 GPU 記憶體衝突
- 如果有多張 GPU，可以提高 concurrent_num 而非 qps

## 🔗 相關資源

- [Funboost 文檔](https://funboost.readthedocs.io/)
- [RabbitMQ 管理](https://www.rabbitmq.com/management.html)
- [Redis 命令參考](https://redis.io/commands/)

## 📞 需要幫助？

如果問題仍未解決：

1. 執行完整診斷並保存輸出：
   ```bash
   python scripts/diagnose_inference_stuck.py > diagnostic_output.txt 2>&1
   ```

2. 收集日誌：
   ```bash
   sudo journalctl -u brain-parcellation.service -n 500 > service_logs.txt
   ```

3. 檢查 GPU 狀態：
   ```bash
   nvidia-smi > gpu_status.txt
   ```

4. 將這些檔案提供給技術支援團隊

