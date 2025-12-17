# 模組文檔

此目錄包含各功能模組的詳細文檔和更新說明。

## 目錄結構

```
modules/
├── README.md                    # 本文件
├── rerun/                      # ReRun 模組
│   ├── RERUN_MODULE_DOCUMENTATION.md
│   └── RERUN_MODULE_UPDATES_SUMMARY.md
└── NIFTI_TOOL_LOGGING_UPDATE.md # NIFTI Tool 日誌更新
```

## ReRun 模組 (`rerun/`)

### RERUN_MODULE_DOCUMENTATION.md
ReRun 模組的完整文檔，包含：
- 模組概述和結構
- 主要功能說明
- API 端點文檔
- 資料流程圖
- 使用示例
- 環境變數依賴
- SQL 語句說明

### RERUN_MODULE_UPDATES_SUMMARY.md
ReRun 模組文件更新摘要，包含：
- 更新概述
- 更新的檔案清單
- 文件風格遵循說明
- 品質保證檢查
- 文件統計

## NIFTI Tool

### NIFTI_TOOL_LOGGING_UPDATE.md
NIFTI Tool 日誌更新說明，包含：
- 專用日誌記錄實作
- Multi-output Series 處理邏輯增強
- 錯誤修復說明
- 日誌解讀指南

## 使用建議

### 開發者
1. 閱讀模組文檔了解模組功能
2. 參考更新摘要了解最新變更
3. 查看使用示例進行集成

### 維護者
1. 參考文檔風格進行更新
2. 保持文檔與代碼同步
3. 重大變更時更新文檔

## 相關文檔

- 後端模組: `../backend/app/`
- Code AI 模組: `../code_ai/`
- 部署文檔: `../deployment/`

