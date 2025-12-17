# 故障排除文檔

此目錄包含系統故障排查和問題解決指南。

## 目錄結構

```
troubleshooting/
├── README.md                    # 本文件
└── TROUBLESHOOTING_INFERENCE_STUCK.md  # Study 推理卡住問題排查
```

## TROUBLESHOOTING_INFERENCE_STUCK.md

Study 300.50 推理卡住問題的完整排查指南，包含：

### 內容章節
1. **問題描述**: Study 狀態進入 300.50 後無法繼續推理
2. **根本原因分析**: 
   - Redis 快取鎖機制問題
   - Subprocess 無超時控制
   - Funboost Consumer 崩潰無自動恢復
3. **已實施的修復**:
   - 智能 Redis 快取管理
   - Subprocess 超時控制
   - 任務完成後清理快取
4. **快速修復指令**: 三種修復方案
5. **診斷清單**: 系統檢查項目
6. **預防措施**: 定期清理和監控設定
7. **常見問題 FAQ**: 問題解答

## 使用建議

### 遇到問題時
1. 先查看「快速修復指令」章節
2. 使用診斷清單逐一檢查
3. 參考常見問題 FAQ

### 預防性維護
1. 設定定期清理（見「預防措施」）
2. 監控關鍵指標
3. 定期執行診斷工具

## 相關文檔

- 快速修復指南: `../guides/quick-fix/QUICK_FIX_GUIDE.md`
- Redis 快取修復: `../guides/quick-fix/README_REDIS_CACHE_FIX.md`
- 部署檢查: `../deployment/DEPLOYMENT_CHECKLIST.md`

