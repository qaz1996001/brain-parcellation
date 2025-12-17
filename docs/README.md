# 文檔目錄

此目錄包含專案的所有文檔說明，按類型和模組組織。

## 📁 目錄結構

```
docs/
├── README.md                    # 本文件（目錄說明）
├── backend/                     # 後端模組文檔
│   ├── refactoring/            # 重構相關文檔
│   ├── bugs/                   # Bug 修復文檔
│   └── app/                    # 應用程式模組文檔
│       ├── series/             # Series 模組
│       ├── study/              # Study 模組
│       └── sync/               # Sync 模組
├── code_ai/                     # Code AI 模組文檔
│   ├── pipeline/               # Pipeline 處理流程
│   ├── task/                   # 任務管理
│   ├── utils/                  # 工具函數
│   ├── SynthSeg/               # SynthSeg 模組
│   ├── dicom2nii/              # DICOM 轉換
│   └── scheduler/              # 任務調度器
├── bugs/                        # Bug 修復文檔
│   └── dwi/                    # DWI Bug 修復相關
│       ├── DWI_BUG_FIX_SUMMARY.md
│       ├── DWI_DATABASE_RECORDS_ANALYSIS.md
│       ├── DWI_FINAL_FIX_GUIDE.md
│       ├── DWI0_DWI1000_FIX_GUIDE.md
│       └── DWI0_DWI1000_ROOT_CAUSE_ANALYSIS.md
├── modules/                     # 模組文檔
│   ├── rerun/                  # ReRun 模組
│   │   ├── RERUN_MODULE_DOCUMENTATION.md
│   │   └── RERUN_MODULE_UPDATES_SUMMARY.md
│   └── NIFTI_TOOL_LOGGING_UPDATE.md
├── deployment/                  # 部署相關文檔
│   ├── DEPLOYMENT_CHECKLIST.md
│   ├── DOCUMENTATION_CHECKLIST.md
│   └── IMPLEMENTATION_SUMMARY.md
├── troubleshooting/              # 故障排除文檔
│   └── TROUBLESHOOTING_INFERENCE_STUCK.md
├── changelog/                   # 變更記錄
│   └── CHANGES.md
└── guides/                      # 指南類文檔
    ├── migration/               # 遷移指南
    │   └── MIGRATION_GUIDE.md
    └── quick-fix/               # 快速修復指南
        ├── QUICK_FIX_GUIDE.md
        └── README_REDIS_CACHE_FIX.md
```

## 📚 文檔分類說明

### Bug 修復文檔 (`bugs/`)
記錄重要的 Bug 修復過程和解決方案：
- **dwi/**: DWI (Diffusion Weighted Imaging) 相關的 Bug 修復文檔
  - Bug 總結和分析
  - 根本原因分析
  - 修復指南和驗證

### 模組文檔 (`modules/`)
各功能模組的詳細文檔：
- **rerun/**: ReRun 模組的完整文檔和更新摘要
- **NIFTI_TOOL**: NIFTI 轉換工具的日誌更新說明

### 部署文檔 (`deployment/`)
部署和實施相關的文檔：
- 部署檢查清單
- 文檔化檢查清單
- 實施總結

### 故障排除 (`troubleshooting/`)
系統故障排查和問題解決指南：
- Study 推理卡住問題排查
- 常見問題和解決方案

### 變更記錄 (`changelog/`)
專案變更歷史記錄：
- 版本更新說明
- 功能變更和修復記錄

### 指南文檔 (`guides/`)
各種使用和操作指南：
- **migration/**: 系統遷移指南（如 Funboost → PgQueuer）
- **quick-fix/**: 快速修復指南和 Redis 快取修復說明

## 🔍 快速查找

### 按主題查找

| 主題 | 路徑 |
|------|------|
| DWI Bug 修復 | `docs/bugs/dwi/` |
| ReRun 模組 | `docs/modules/rerun/` |
| 部署指南 | `docs/deployment/` |
| 故障排除 | `docs/troubleshooting/` |
| 快速修復 | `docs/guides/quick-fix/` |
| 遷移指南 | `docs/guides/migration/` |
| 變更記錄 | `docs/changelog/` |

### 按模組查找

| 模組 | 路徑 |
|------|------|
| Backend | `docs/backend/` |
| Code AI | `docs/code_ai/` |
| ReRun | `docs/modules/rerun/` |
| NIFTI Tool | `docs/modules/` |

## 📖 使用建議

### 新成員入門
1. 閱讀 `docs/backend/README.md` 了解後端架構
2. 閱讀 `docs/code_ai/README.md` 了解 AI 模組
3. 查看 `docs/changelog/CHANGES.md` 了解最新變更

### 開發者參考
1. **Bug 修復**: 查看 `docs/bugs/` 目錄
2. **模組開發**: 查看 `docs/modules/` 和 `docs/backend/app/`
3. **部署**: 參考 `docs/deployment/` 目錄

### 故障排查
1. **快速修復**: `docs/guides/quick-fix/QUICK_FIX_GUIDE.md`
2. **詳細排查**: `docs/troubleshooting/TROUBLESHOOTING_INFERENCE_STUCK.md`
3. **特定 Bug**: `docs/bugs/dwi/` 目錄

## 📝 文檔維護

### 新增文檔
請按照以下規則將新文檔放在適當目錄：
- Bug 修復 → `docs/bugs/`
- 模組文檔 → `docs/modules/` 或 `docs/backend/app/` 或 `docs/code_ai/`
- 部署相關 → `docs/deployment/`
- 故障排除 → `docs/troubleshooting/`
- 變更記錄 → `docs/changelog/`
- 指南類 → `docs/guides/`

### 更新文檔
- 修改文檔時請保持目錄結構的一致性
- 重大變更請更新 `docs/changelog/CHANGES.md`
- 模組更新請同步更新對應的模組文檔

## 🔗 相關資源

- 後端文檔: `docs/backend/`
- Code AI 文檔: `docs/code_ai/`
- 快速修復: `docs/guides/quick-fix/`
- 故障排除: `docs/troubleshooting/`

## 📊 文檔統計

- **總文檔數**: 30+ 個
- **分類目錄**: 7 個主要分類
- **模組覆蓋**: Backend、Code AI、ReRun、NIFTI Tool
- **文檔類型**: Bug 修復、模組文檔、部署指南、故障排除、變更記錄

---

**最後更新**: 2025-12-17  
**維護者**: Development Team

