# ✅ 部署檢查清單 - Redis 雙狀態快取系統

## 📋 部署前準備

### 1. 備份現有系統
```bash
# 備份程式碼
cp -r code_ai/task/task_pipeline.py code_ai/task/task_pipeline.py.backup
cp -r backend/app/sync/service.py backend/app/sync/service.py.backup

# 備份 Redis 資料（可選）
redis-cli --rdb /backup/redis_dump_$(date +%Y%m%d).rdb

# 記錄當前服務狀態
systemctl status brain-parcellation.service > /backup/service_status_before.txt
```

### 2. 確認環境
- [ ] Python 環境正常
- [ ] Redis 服務運行中
- [ ] RabbitMQ 服務運行中
- [ ] GPU 可用 (nvidia-smi)
- [ ] 磁碟空間充足 (> 10GB)

---

## 🚀 部署步驟

### 步驟 1: 停止服務
```bash
sudo systemctl stop brain-parcellation.service
sleep 5

# 確認服務已停止
ps aux | grep funboost
# 應該沒有輸出
```

### 步驟 2: 部署新程式碼
```bash
# 1. 已修改的檔案
git pull origin main
# 或手動複製修改後的檔案
```

### 步驟 3: 清理舊的 Redis 快取
```bash
# ⚠️ 警告：這會清除所有推理任務的快取
redis-cli --scan --pattern "inference_task:*" | xargs redis-cli DEL

# 確認已清除
redis-cli --scan --pattern "inference_task:*" | wc -l
# 應該輸出 0
```

### 步驟 4: 啟動服務
```bash
sudo systemctl start brain-parcellation.service
sleep 10

# 檢查服務狀態
sudo systemctl status brain-parcellation.service
# 應該顯示 "active (running)"
```

### 步驟 5: 驗證部署
```bash
# 1. 檢查 Funboost consumer
ps aux | grep funboost
# 應該看到 funboost 進程

# 2. 檢查 RabbitMQ 連接
rabbitmqadmin list queues name consumers
# task_pipeline_inference_queue 應該有 consumers > 0

# 3. 查看日誌（實時）
sudo journalctl -u brain-parcellation.service -f
# 應該看到正常的日誌輸出
```

---

## 🧪 功能測試

### 測試 1: 正常推理流程
```bash
# 1. 觸發一個新的 Study 進入 300.50 狀態
# （透過您的業務系統）

# 2. 檢查 Redis 快取
redis-cli GET "inference_task:<STUDY_UID>,<STUDY_ID>"
# 應該輸出 "queued"

redis-cli TTL "inference_task:<STUDY_UID>,<STUDY_ID>"
# 應該輸出接近 3600 的數字

# 3. 等待推理完成（10-20 分鐘）

# 4. 再次檢查 Redis
redis-cli GET "inference_task:<STUDY_UID>,<STUDY_ID>"
# 應該輸出 "completed"

redis-cli TTL "inference_task:<STUDY_UID>,<STUDY_ID>"
# 應該輸出接近 7200 的數字
```

### 測試 2: 重複請求保護
```bash
# 1. 使用剛才完成的 Study，再次觸發 300.50

# 2. 檢查日誌
sudo journalctl -u brain-parcellation.service -n 50 | grep "Skipping completed"
# 應該看到 "Skipping completed inference task" 訊息

# 3. 確認沒有推送新任務
rabbitmqadmin get queue=task_pipeline_inference_queue count=1
# 應該沒有新的任務
```

### 測試 3: 診斷工具
```bash
# 1. 執行快速修復
./scripts/quick_fix_stuck.sh
# 應該顯示所有組件正常

# 2. 執行完整診斷
python scripts/diagnose_inference_stuck.py \
    --study-uid <STUDY_UID> \
    --study-id <STUDY_ID>
# 應該顯示詳細的狀態資訊
```

---

## ✅ 部署後檢查清單

### 系統健康檢查
- [ ] ✅ Brain Parcellation 服務運行中
- [ ] ✅ Funboost consumer 進程存在
- [ ] ✅ Redis 連接正常
- [ ] ✅ RabbitMQ 連接正常
- [ ] ✅ GPU 可用且記憶體充足
- [ ] ✅ 日誌無錯誤訊息

### 功能驗證
- [ ] ✅ 新任務可以正常推理
- [ ] ✅ Redis 快取狀態正確（queued → completed）
- [ ] ✅ 完成的任務 2 小時內不會重複執行
- [ ] ✅ 診斷工具運行正常
- [ ] ✅ 快速修復腳本可用

### 監控設定
- [ ] ✅ 日誌監控已啟動
- [ ] ✅ Redis key 數量在正常範圍（< 10）
- [ ] ✅ RabbitMQ 佇列長度正常（< 5）
- [ ] ✅ 定期清理腳本已設定（可選）

---

## 🔍 部署驗證指令

### 一鍵驗證腳本
```bash
#!/bin/bash
echo "========================================="
echo "🔍 部署驗證開始"
echo "========================================="

# 1. 檢查服務狀態
echo -e "\n1️⃣ 檢查服務狀態..."
systemctl is-active brain-parcellation.service && echo "✅ 服務運行中" || echo "❌ 服務未運行"

# 2. 檢查 Funboost consumer
echo -e "\n2️⃣ 檢查 Funboost consumer..."
ps aux | grep -v grep | grep funboost > /dev/null && echo "✅ Consumer 運行中" || echo "❌ Consumer 未運行"

# 3. 檢查 Redis
echo -e "\n3️⃣ 檢查 Redis 連接..."
redis-cli PING > /dev/null 2>&1 && echo "✅ Redis 正常" || echo "❌ Redis 異常"

# 4. 檢查 RabbitMQ
echo -e "\n4️⃣ 檢查 RabbitMQ 佇列..."
CONSUMERS=$(rabbitmqadmin list queues name consumers 2>/dev/null | grep task_pipeline_inference_queue | awk '{print $4}')
if [ "$CONSUMERS" -gt 0 ]; then
    echo "✅ RabbitMQ 正常，消費者數量: $CONSUMERS"
else
    echo "⚠️  RabbitMQ 異常或無消費者"
fi

# 5. 檢查 GPU
echo -e "\n5️⃣ 檢查 GPU 狀態..."
nvidia-smi > /dev/null 2>&1 && echo "✅ GPU 正常" || echo "⚠️  GPU 異常"

# 6. 檢查 Redis key 數量
echo -e "\n6️⃣ 檢查 Redis key 數量..."
KEY_COUNT=$(redis-cli --scan --pattern "inference_task:*" 2>/dev/null | wc -l)
echo "   推理任務 key 數量: $KEY_COUNT"
if [ "$KEY_COUNT" -lt 20 ]; then
    echo "✅ Key 數量正常"
else
    echo "⚠️  Key 數量過多，可能需要清理"
fi

# 7. 檢查最近日誌
echo -e "\n7️⃣ 檢查最近日誌（最後 5 行）..."
sudo journalctl -u brain-parcellation.service -n 5 --no-pager

echo -e "\n========================================="
echo "🎉 驗證完成"
echo "========================================="
```

儲存為 `scripts/verify_deployment.sh` 並執行：
```bash
chmod +x scripts/verify_deployment.sh
./scripts/verify_deployment.sh
```

---

## 🚨 常見問題處理

### 問題 1: 服務無法啟動
```bash
# 查看詳細錯誤
sudo journalctl -u brain-parcellation.service -n 100

# 檢查環境變數
cat .env | grep -E "REDIS|RABBITMQ"

# 手動測試啟動
cd /var/www/brain-parcellation
source .env
funboost start -m funboost consume_all_queues --project_root_path=$(pwd) --import_modules_str code_ai.task
```

### 問題 2: Redis 連接失敗
```bash
# 檢查 Redis 服務
sudo systemctl status redis

# 測試連接
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD PING

# 檢查防火牆
sudo firewall-cmd --list-ports
```

### 問題 3: RabbitMQ 無消費者
```bash
# 檢查 RabbitMQ 服務
sudo systemctl status rabbitmq-server

# 查看連接
rabbitmqadmin list connections

# 重啟 RabbitMQ
sudo systemctl restart rabbitmq-server
```

---

## 📊 效能基準

部署後的預期效能：

| 指標 | 目標值 | 檢查方法 |
|------|--------|----------|
| 服務啟動時間 | < 30 秒 | `systemctl status` |
| 推理任務完成時間 | 10-20 分鐘 | 查看日誌 |
| Redis key 數量 | < 10 | `redis-cli --scan` |
| RabbitMQ 佇列長度 | < 5 | `rabbitmqadmin list queues` |
| GPU 使用率 | 70-90% (推理時) | `nvidia-smi` |
| CPU 使用率 | < 50% | `top` |

---

## 📝 回滾計畫

如果部署失敗，執行回滾：

```bash
# 1. 停止服務
sudo systemctl stop brain-parcellation.service

# 2. 恢復舊程式碼
cp code_ai/task/task_pipeline.py.backup code_ai/task/task_pipeline.py
cp backend/app/sync/service.py.backup backend/app/sync/service.py

# 3. 清理 Redis
redis-cli --scan --pattern "inference_task:*" | xargs redis-cli DEL

# 4. 重啟服務
sudo systemctl start brain-parcellation.service

# 5. 驗證
./scripts/verify_deployment.sh
```

---

## 📞 支援聯絡

如果遇到無法解決的問題：

1. 執行完整診斷：
   ```bash
   python scripts/diagnose_inference_stuck.py > diagnostic_full.txt 2>&1
   ```

2. 收集日誌：
   ```bash
   sudo journalctl -u brain-parcellation.service -n 500 > service_logs.txt
   ```

3. 系統資訊：
   ```bash
   nvidia-smi > gpu_info.txt
   df -h > disk_info.txt
   free -h > memory_info.txt
   ```

4. 提供這些檔案給技術支援

---

**檢查清單版本**: 2.0.0  
**最後更新**: 2025-01-15  
**部署環境**: 生產環境




