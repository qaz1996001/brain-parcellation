# 🚀 Study 300.50 卡住 - 快速修復指南

## ⚡ 立即執行（1 分鐘內解決）

```bash
# 1. 執行快速修復腳本
./scripts/quick_fix_stuck.sh

# 2. 如果仍卡住，手動重啟服務
sudo systemctl restart brain-parcellation.service

# 3. 查看日誌確認
sudo journalctl -u brain-parcellation.service -f
```

## 🔍 針對特定 Study 的診斷

```bash
# 完整診斷
python scripts/diagnose_inference_stuck.py \
    --study-uid <STUDY_UID> \
    --study-id <STUDY_ID>

# 清理 Redis 快取（如果診斷發現問題）
python scripts/diagnose_inference_stuck.py \
    --cleanup <STUDY_UID> <STUDY_ID>
```

## 🛠️ 已修復的問題

### ✅ 1. Redis 快取鎖機制
- **問題**: 快取 6 小時，即使失敗也不清除
- **修復**: 縮短為 1 小時，自動檢測並清除異常 key

### ✅ 2. Subprocess 無超時
- **問題**: 命令卡住會阻塞整個 consumer
- **修復**: 30 分鐘超時，自動終止並記錄錯誤

### ✅ 3. 任務完成不清理快取
- **問題**: 快取累積導致記憶體浪費
- **修復**: 任務完成立即清理 Redis key

## 📊 修復效果

| 項目 | 修復前 | 修復後 |
|------|--------|--------|
| 排隊快取時間 | 6 小時 | 1 小時 (queued) |
| 完成快取時間 | ❌ 無 | ✅ 2 小時 (completed) |
| 卡住檢測 | ❌ 無 | ✅ 自動 TTL < 30min 檢查 |
| Subprocess 超時 | ❌ 無限等待 | ✅ 30 分鐘 |
| 快取狀態管理 | ❌ 單一狀態 | ✅ queued/completed |
| GPU QPS | 1 (不變) | 1 (保持) |

## 📝 預防性維護

### 設定定期清理 (推薦)

```bash
# 編輯 crontab
crontab -e

# 每小時執行清理
0 * * * * /path/to/Task04_git/scripts/cleanup_redis_cache.sh >> /var/log/redis_cleanup.log 2>&1
```

## 🔗 詳細文檔

完整排查指南：[docs/TROUBLESHOOTING_INFERENCE_STUCK.md](docs/TROUBLESHOOTING_INFERENCE_STUCK.md)

## ⚠️ 重要提醒

1. **QPS=1 保持不變**：因為 GPU 記憶體限制，不建議提高
2. **Redis 快取策略**：
   - **排隊狀態 (queued)**: 1 小時，防止重複推送到佇列
   - **完成狀態 (completed)**: 2 小時，防止短時間內重複執行相同任務
3. **超時設定**：30 分鐘是合理值，請勿隨意調整
4. **日誌監控**：建議持續監控 `sudo journalctl -u brain-parcellation.service -f`

## 🆘 如果還是卡住

1. 檢查 GPU 記憶體：`nvidia-smi`
2. 檢查磁碟空間：`df -h`
3. 檢查資料庫連接：`psql -U postgres -c "SELECT 1;"`
4. 聯絡技術支援並提供診斷輸出

---

**最後更新**: 2025-01-15  
**版本**: 1.0.0

