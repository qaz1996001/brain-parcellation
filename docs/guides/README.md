# 指南文檔

此目錄包含各種使用和操作指南。

## 目錄結構

```
guides/
├── README.md                    # 本文件
├── migration/                   # 遷移指南
│   └── MIGRATION_GUIDE.md      # Funboost → PgQueuer 遷移指南
└── quick-fix/                  # 快速修復指南
    ├── QUICK_FIX_GUIDE.md      # Study 300.50 卡住快速修復
    └── README_REDIS_CACHE_FIX.md  # Redis 快取修復完整方案
```

## 遷移指南 (`migration/`)

### MIGRATION_GUIDE.md
Funboost → PgQueuer 遷移指南，包含：
- 遷移概覽
- 遷移步驟（5 個階段）
- API 對應表
- 注意事項
- 測試建議
- 回滾計劃
- 預期效益

**適用場景**: 系統架構遷移、任務隊列升級

## 快速修復指南 (`quick-fix/`)

### QUICK_FIX_GUIDE.md
Study 300.50 卡住的快速修復指南，包含：
- 立即執行指令（1 分鐘內解決）
- 針對特定 Study 的診斷
- 已修復的問題說明
- 修復效果對比
- 預防性維護建議

**適用場景**: 緊急故障修復、日常維護

### README_REDIS_CACHE_FIX.md
Redis 雙狀態快取系統的完整修復方案，包含：
- 快速開始（1 分鐘）
- 核心特性說明
- 狀態流程圖
- 文檔結構
- 工具使用說明
- 使用場景詳解
- 異常處理
- 監控指標
- 定期維護
- 設定建議
- 進階主題
- 獲取支援

**適用場景**: Redis 快取問題、任務重複執行問題

## 使用建議

### 系統遷移
1. 閱讀 `migration/MIGRATION_GUIDE.md` 了解遷移流程
2. 按照遷移步驟逐步執行
3. 參考 API 對應表進行代碼轉換

### 快速修復
1. 遇到問題時先查看 `quick-fix/QUICK_FIX_GUIDE.md`
2. 需要詳細方案時參考 `quick-fix/README_REDIS_CACHE_FIX.md`
3. 執行診斷工具確認問題

## 相關文檔

- 故障排除: `../troubleshooting/TROUBLESHOOTING_INFERENCE_STUCK.md`
- 部署檢查: `../deployment/DEPLOYMENT_CHECKLIST.md`
- 變更記錄: `../changelog/CHANGES.md`

