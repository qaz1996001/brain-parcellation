# Bug 修復文檔

此目錄包含所有重要的 Bug 修復文檔，記錄問題分析、根本原因和解決方案。

## 目錄結構

```
bugs/
├── README.md                    # 本文件
└── dwi/                        # DWI Bug 修復相關
    ├── DWI_BUG_FIX_SUMMARY.md              # Bug 修復總結
    ├── DWI_DATABASE_RECORDS_ANALYSIS.md     # 資料庫記錄分析
    ├── DWI_FINAL_FIX_GUIDE.md              # 最終修復指南
    ├── DWI0_DWI1000_FIX_GUIDE.md           # DWI0/DWI1000 修復指南
    └── DWI0_DWI1000_ROOT_CAUSE_ANALYSIS.md # 根本原因分析
```

## DWI Bug 修復

### 問題概述
DWI (Diffusion Weighted Imaging) 序列在轉檔過程中出現問題，導致已完成的 Study 無法進入推理佇列。

### 文檔說明

1. **DWI_BUG_FIX_SUMMARY.md**
   - Bug 修復的完整總結
   - 問題描述和根本原因
   - 修復方案和測試驗證
   - 影響範圍和部署建議

2. **DWI0_DWI1000_ROOT_CAUSE_ANALYSIS.md**
   - 詳細的根本原因分析
   - 正則表達式語法錯誤說明
   - PostgreSQL 陣列順序問題
   - 解決方案對比

3. **DWI_FINAL_FIX_GUIDE.md**
   - 最終修復指南
   - 實施步驟
   - 測試驗證方法

4. **DWI0_DWI1000_FIX_GUIDE.md**
   - DWI0/DWI1000 特定修復指南
   - 快速修復步驟

5. **DWI_DATABASE_RECORDS_ANALYSIS.md**
   - 資料庫記錄分析
   - 實際數據案例

## 使用建議

### 快速了解問題
1. 閱讀 `DWI_BUG_FIX_SUMMARY.md` 了解整體情況
2. 查看 `DWI0_DWI1000_ROOT_CAUSE_ANALYSIS.md` 了解技術細節

### 實施修復
1. 參考 `DWI_FINAL_FIX_GUIDE.md` 進行修復
2. 使用 `DWI0_DWI1000_FIX_GUIDE.md` 進行快速修復

### 問題排查
1. 查看 `DWI_DATABASE_RECORDS_ANALYSIS.md` 分析數據
2. 對比實際情況與文檔中的案例

## 相關文檔

- 快速修復指南: `../guides/quick-fix/QUICK_FIX_GUIDE.md`
- 故障排除: `../troubleshooting/TROUBLESHOOTING_INFERENCE_STUCK.md`
- 部署檢查: `../deployment/DEPLOYMENT_CHECKLIST.md`

