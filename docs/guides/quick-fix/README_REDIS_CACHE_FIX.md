# 🎯 Redis 雙狀態快取系統 - 完整修復方案

> **目標**: 解決 Study 300.50 狀態卡住問題，任務完成後 2 小時允許重新執行

---

## 📌 快速開始（1 分鐘）

```bash
# 1. 快速診斷和修復
./scripts/quick_fix_stuck.sh

# 2. 驗證部署
./scripts/verify_deployment.sh

# 3. 查看實時日誌
sudo journalctl -u brain-parcellation.service -f
```

---

## 🎯 核心特性

### ✅ 雙狀態系統

| 狀態 | TTL | 用途 |
|------|-----|------|
| **queued** | 1 小時 | 防止重複推送到佇列 |
| **completed** | 2 小時 | 防止短時間內重複執行 |

### ✅ 智能卡住檢測

- TTL < 30 分鐘的 `queued` 任務自動檢測並重試
- Subprocess 30 分鐘超時保護
- 自動日誌記錄和錯誤追蹤

### ✅ 保護機制

- **排隊保護**: 1 小時內不會重複推送相同任務
- **完成保護**: 2 小時內不會重複執行相同 Study
- **超時保護**: 單一任務最長執行 30 分鐘

---

## 📊 狀態流程圖

```
Study 300.50 (準備推理)
         ↓
    檢查 Redis
         │
    ┌────┴─────┐
    │ 無快取？  │
    └────┬─────┘
         │ YES
         ↓
┌──────────────────┐
│ Redis = "queued" │
│ TTL = 1 小時     │
└────────┬─────────┘
         │
         ↓
   推送到 RabbitMQ
         │
         ↓
  Funboost Consumer
    處理任務
    (最長 30 分鐘)
         │
         ↓
    推理完成
         │
         ↓
┌────────────────────┐
│ Redis = "completed"│
│ TTL = 2 小時       │
└────────┬───────────┘
         │
         ↓
  2 小時後自動過期
         │
         ↓
   允許重新執行
```

---

## 📂 文檔結構

```
├── QUICK_FIX_GUIDE.md                      # 快速參考（必讀）
├── IMPLEMENTATION_SUMMARY.md               # 實施總結
├── DEPLOYMENT_CHECKLIST.md                 # 部署檢查清單
├── README_REDIS_CACHE_FIX.md              # 本文檔
├── docs/
│   ├── TROUBLESHOOTING_INFERENCE_STUCK.md # 完整排查指南
│   └── REDIS_CACHE_FLOW.md                # 狀態流程詳解
└── scripts/
    ├── diagnose_inference_stuck.py        # 診斷工具
    ├── quick_fix_stuck.sh                 # 快速修復
    ├── cleanup_redis_cache.sh             # 定期清理
    └── verify_deployment.sh               # 部署驗證
```

---

## 🛠️ 工具使用

### 1. 快速修復（最常用）

```bash
./scripts/quick_fix_stuck.sh
```

**功能**:
- 檢查 Funboost consumer 狀態
- 檢查 RabbitMQ 佇列
- 檢查並清理 Redis 快取
- 顯示最近日誌

### 2. 完整診斷

```bash
python scripts/diagnose_inference_stuck.py \
    --study-uid <STUDY_UID> \
    --study-id <STUDY_ID>
```

**功能**:
- Redis 快取狀態（顯示 queued/completed）
- RabbitMQ 佇列狀態
- Funboost consumer 進程
- 資料庫狀態記錄
- 提供具體建議

### 3. 清理 Redis 快取

```bash
# 清理特定 Study
python scripts/diagnose_inference_stuck.py \
    --cleanup <STUDY_UID> <STUDY_ID>

# 定期清理（建議設定 cron）
./scripts/cleanup_redis_cache.sh
```

### 4. 驗證部署

```bash
./scripts/verify_deployment.sh
```

**功能**:
- 檢查所有關鍵組件
- 顯示 Redis key 統計
- GPU 狀態
- 磁碟空間

---

## 🎓 使用場景

### 場景 1: 正常執行（最常見 95%）

```
1. Study 進入 300.50
2. Redis: key = "queued", TTL = 1h
3. 推理執行 10-20 分鐘
4. Redis: key = "completed", TTL = 2h
5. 2 小時後 key 過期
6. 如再次執行，允許
```

**預期結果**: ✅ 正常完成，無需介入

### 場景 2: 任務卡住（自動恢復 4%）

```
1. Study 進入 300.50
2. Redis: key = "queued", TTL = 1h
3. 推理卡住（GPU OOM / 網路問題）
4. 30 分鐘後，TTL < 1800 秒
5. 系統自動檢測並刪除 key
6. 強制重新推送任務
7. 正常完成
```

**預期結果**: ✅ 自動恢復，無需介入

### 場景 3: 重複請求（保護期 1%）

```
1. Study A 完成推理
2. Redis: key = "completed", TTL = 2h
3. 30 分鐘後，相同 Study A 再次進入
4. 檢查 Redis: key = "completed"
5. 跳過執行（保護期內）
6. 避免浪費 GPU 資源
```

**預期結果**: ✅ 智能跳過，節省資源

---

## 🚨 異常處理

### 問題 1: Funboost Consumer 停止

**症狀**:
- RabbitMQ 有訊息但無消費者
- 推理任務不執行

**解決**:
```bash
sudo systemctl restart brain-parcellation.service
```

### 問題 2: Redis Key 累積過多

**症狀**:
- Redis key 數量 > 20
- 大量 `queued` 狀態

**解決**:
```bash
# 查看狀態
./scripts/quick_fix_stuck.sh

# 清理異常 key
./scripts/cleanup_redis_cache.sh
```

### 問題 3: GPU 記憶體不足

**症狀**:
- 推理超時
- nvidia-smi 顯示記憶體滿

**解決**:
```bash
# 檢查 GPU
nvidia-smi

# 重啟服務釋放記憶體
sudo systemctl restart brain-parcellation.service
```

---

## 📊 監控指標

### 正常值範圍

| 指標 | 正常 | 警告 | 危險 |
|------|------|------|------|
| Redis key 數量 | < 10 | 10-20 | > 20 |
| Queued 任務 | < 5 | 5-10 | > 10 |
| RabbitMQ 佇列長度 | < 5 | 5-10 | > 10 |
| GPU 使用率 | 70-90% | 90-95% | > 95% |
| 磁碟使用率 | < 80% | 80-90% | > 90% |

### 監控命令

```bash
# Redis key 數量
redis-cli --scan --pattern "inference_task:*" | wc -l

# 狀態分佈
./scripts/quick_fix_stuck.sh

# GPU 狀態
nvidia-smi

# 服務狀態
./scripts/verify_deployment.sh
```

---

## 🔄 定期維護

### 每日任務

```bash
# 早上：檢查昨日日誌
sudo journalctl -u brain-parcellation.service --since yesterday | grep ERROR

# 晚上：清理 Redis
./scripts/cleanup_redis_cache.sh
```

### 每週任務

```bash
# 完整健康檢查
./scripts/verify_deployment.sh

# 檢查 GPU 記憶體趨勢
nvidia-smi --query-gpu=memory.used --format=csv --loop=1 | head -n 100
```

### 每月任務

```bash
# 清理舊日誌
sudo journalctl --vacuum-time=30d

# 檢查磁碟空間
df -h

# 更新統計
echo "Redis keys: $(redis-cli --scan --pattern 'inference_task:*' | wc -l)"
echo "Completed last month: $(redis-cli --scan --pattern 'inference_task:*' | xargs -I {} redis-cli GET {} | grep -c completed)"
```

---

## ⚙️ 設定建議

### Cron 定期清理（建議）

```bash
# 編輯 crontab
crontab -e

# 每小時清理異常 key
0 * * * * /path/to/scripts/cleanup_redis_cache.sh >> /var/log/redis_cleanup.log 2>&1

# 每天早上 2 點完整驗證
0 2 * * * /path/to/scripts/verify_deployment.sh >> /var/log/deployment_verify.log 2>&1
```

### 告警設定（可選）

監控以下事件並發送告警：

1. Funboost consumer 進程消失
2. Redis key 數量 > 20
3. RabbitMQ 佇列長度 > 10
4. GPU 記憶體 > 95%
5. 磁碟使用率 > 90%

---

## 🎓 進階主題

### 自訂 TTL 值

如果您需要調整快取時間：

```python
# backend/app/sync/service.py 行 1093-1095
await redis_client.set(
    inference_task_key, "queued", ex=3600  # 修改這裡（秒）
)

# code_ai/task/task_pipeline.py 行 157
redis_client.setex(inference_task_key, 7200, "completed")  # 修改這裡（秒）
```

### 禁用保護期

如果您想每次都執行（不建議）：

```python
# backend/app/sync/service.py
# 註解掉這段檢查
# if cached_value == "completed":
#     continue
```

---

## 📞 獲取支援

### 收集診斷資訊

```bash
# 1. 完整診斷
python scripts/diagnose_inference_stuck.py > diagnostic_full.txt 2>&1

# 2. 服務日誌
sudo journalctl -u brain-parcellation.service -n 500 > service_logs.txt

# 3. 系統資訊
nvidia-smi > gpu_info.txt
df -h > disk_info.txt
free -h > memory_info.txt
redis-cli INFO > redis_info.txt

# 4. 打包
tar -czf diagnostic_$(date +%Y%m%d_%H%M%S).tar.gz \
    diagnostic_full.txt service_logs.txt gpu_info.txt disk_info.txt memory_info.txt redis_info.txt
```

### 常見問題 FAQ

**Q: 為什麼任務完成後還要保留 2 小時快取？**  
A: 防止短時間內重複執行相同的 Study，節省 GPU 資源。2 小時後自動過期。

**Q: 如果我想立即重新執行怎麼辦？**  
A: 手動刪除 Redis key：
```bash
python scripts/diagnose_inference_stuck.py --cleanup <UID> <ID>
```

**Q: QPS=1 會不會太慢？**  
A: 這是根據 GPU 能力設定的。單 GPU 無法並行處理多個深度學習任務。

**Q: Redis 連接失敗會影響推理嗎？**  
A: 不會。系統有容錯處理，Redis 失敗時仍會執行推理，只是無法防止重複。

---

## 📚 相關資源

- [快速修復指南](QUICK_FIX_GUIDE.md)
- [實施總結](IMPLEMENTATION_SUMMARY.md)
- [部署檢查清單](DEPLOYMENT_CHECKLIST.md)
- [完整排查指南](docs/TROUBLESHOOTING_INFERENCE_STUCK.md)
- [狀態流程詳解](docs/REDIS_CACHE_FLOW.md)

---

**版本**: 2.0.0 (雙狀態系統)  
**最後更新**: 2025-01-15  
**狀態**: ✅ 生產就緒  
**維護者**: AI Team




