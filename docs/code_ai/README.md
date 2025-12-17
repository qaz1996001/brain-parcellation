# Code AI 文檔目錄

此目錄包含所有 `code_ai` 模組相關的文檔說明，按模組組織。

## 目錄結構

```
docs/code_ai/
├── README.md                    # 本文件（目錄說明）
├── pipeline/                    # Pipeline 處理流程文檔
│   ├── synthseg/               # SynthSeg 相關流程
│   ├── dicom_to_nii/          # DICOM 轉 NIFTI 流程
│   └── inference/             # 推理流程
├── task/                        # 任務管理文檔
│   ├── dicom2nii/             # DICOM 轉 NIFTI 任務
│   └── pipeline/              # Pipeline 任務
├── utils/                       # 工具函數文檔
│   ├── synthseg/              # SynthSeg 工具
│   └── parcellation/          # 分割工具
├── SynthSeg/                    # SynthSeg 模組文檔
├── dicom2nii/                  # DICOM 轉 NIFTI 模組文檔
└── scheduler/                  # 任務調度器文檔
```

## 模組說明

### Pipeline (`pipeline/`)
包含所有數據處理流程的文檔：
- **SynthSeg**: 腦部分割流程
- **DICOM to NIFTI**: DICOM 檔案轉換流程
- **Inference**: AI 模型推理流程
- **CMB**: 微出血檢測流程
- **Aneurysm**: 動脈瘤檢測流程
- **WMH**: 白質高信號檢測流程

### Task (`task/`)
任務隊列和任務管理相關文檔：
- **dicom2nii**: DICOM 轉 NIFTI 任務定義
- **pipeline**: Pipeline 執行任務定義

### Utils (`utils/`)
工具函數和輔助模組文檔：
- **synthseg**: SynthSeg 相關工具函數
- **parcellation**: 腦部分割工具函數

### SynthSeg (`SynthSeg/`)
SynthSeg 模組的核心文檔，包括：
- 模型使用說明
- API 參考
- 配置選項

### DICOM2NII (`dicom2nii/`)
DICOM 轉 NIFTI 轉換模組文檔：
- 轉換流程說明
- 參數配置
- 錯誤處理

### Scheduler (`scheduler/`)
任務調度器文檔：
- 調度策略
- 任務優先級管理
- 資源分配

## 使用說明

1. **查找文檔**: 根據模組名稱，在對應目錄中查找
2. **新增文檔**: 請按照現有目錄結構，將新文檔放在適當的模組目錄下
3. **更新文檔**: 修改文檔時請保持目錄結構的一致性

## 相關文檔

- 後端文檔: `docs/backend/`
- 部署文檔: `docs/deployment/` (如存在)
- 故障排除: `docs/TROUBLESHOOTING_INFERENCE_STUCK.md`

## 模組對應關係

| 源目錄 | 文檔目錄 | 說明 |
|--------|---------|------|
| `code_ai/pipeline/` | `docs/code_ai/pipeline/` | Pipeline 處理流程 |
| `code_ai/task/` | `docs/code_ai/task/` | 任務管理 |
| `code_ai/utils/` | `docs/code_ai/utils/` | 工具函數 |
| `code_ai/SynthSeg/` | `docs/code_ai/SynthSeg/` | SynthSeg 模組 |
| `code_ai/dicom2nii/` | `docs/code_ai/dicom2nii/` | DICOM 轉換 |
| `code_ai/scheduler/` | `docs/code_ai/scheduler/` | 任務調度 |

