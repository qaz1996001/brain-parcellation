# Backend 文檔目錄

此目錄包含所有後端相關的文檔說明，按類型和模組組織。

## 目錄結構

```
docs/backend/
├── README.md                    # 本文件（目錄說明）
├── refactoring/                 # 重構相關文檔
│   ├── REFACTOR_SUMMARY.md      # 重構總結
│   ├── REFACTOR_PLAN_POSTGRESQL.md  # PostgreSQL 重構計劃
│   ├── REFACTOR_COMPLETE.md     # 重構完成報告
│   └── test_refactor.md         # 重構測試文檔
├── bugs/                        # Bug 修復文檔
│   └── FIX_INFERENCE_BUG.md     # 推理 Bug 修復說明
└── app/                         # 應用程式模組文檔
    ├── BACKEND_APP_DOCUMENTATION.md  # 後端應用程式總體文檔
    ├── series/                  # Series 模組文檔
    │   └── README_DOCUMENTATION.md
    ├── study/                   # Study 模組文檔
    │   ├── README_DOCUMENTATION.md
    │   └── DOCUMENTATION_GUIDE.md
    └── sync/                    # Sync 模組文檔
        ├── FINAL_DOCUMENTATION_SUMMARY.md
        ├── SERVICE_DOCUMENTATION.md
        ├── ARCHITECTURE_OVERVIEW.md
        └── DOCUMENTATION_UPDATE.md
```

## 文檔分類說明

### 重構文檔 (refactoring/)
包含所有與系統重構相關的文檔，記錄重構計劃、執行過程和完成狀態。

### Bug 修復文檔 (bugs/)
記錄重要的 Bug 修復過程和解決方案。

### 應用程式模組文檔 (app/)
按模組組織的應用程式文檔：
- **series/**: Series 管理模組文檔
- **study/**: Study 管理模組文檔
- **sync/**: DICOM 同步服務模組文檔

## 使用說明

1. **查找文檔**: 根據文檔類型或模組名稱，在對應目錄中查找
2. **新增文檔**: 請按照現有目錄結構，將新文檔放在適當的分類目錄下
3. **更新文檔**: 修改文檔時請保持目錄結構的一致性

## 相關文檔

- 前端文檔: `docs/frontend/` (如存在)
- 部署文檔: `docs/deployment/` (如存在)
- 故障排除: `docs/TROUBLESHOOTING_INFERENCE_STUCK.md`

