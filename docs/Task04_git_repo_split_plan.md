# Task04_git_repo_split_plan.md - 完整的重構計劃

## 目標架構

```
D:\00_Chen\Task04_git_rdx\
├── rdxai/                    # 核心基礎庫（公用 + inference 框架）
├── brain-aneurysm/           # 動脈瘤檢測專案
├── brain-cmb/                # 腦微出血檢測專案
├── brain-parcellation/       # 腦分割專案
└── dicom2nii/                # DICOM 轉換工具
```

## 拆分策略

### 第一步：建立 rdxai 核心庫

rdxai 應包含的內容:

```
rdxai/
├── pyproject.toml
├── README.md
├── rdxai/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── inference_base.py         # 推理基類
│   │   ├── pipeline_base.py          # 管道基類
│   │   └── model_config_base.py      # 配置基類
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dicom_utils.py            # DICOM 處理共用函數
│   │   ├── nifti_utils.py            # NIfTI 轉換共用函數
│   │   └── helpers.py                # 其他工具函數
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── response_models.py        # 共用的 Pydantic 模型
│   │   └── config_models.py          # 配置模型
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── nnunet_handler.py         # nnU-Net 推理通用邏輯
│   │   └── model_loader.py           # 模型加載共用邏輯
│   │
│   └── exceptions/
│       ├── __init__.py
│       └── custom_exceptions.py      # 自定義異常類
│
├── tests/
│   ├── test_utils.py
│   └── test_base_classes.py
```

### 第二步：拆分各個專案倉庫

#### brain-aneurysm 結構
```
brain-aneurysm/
├── pyproject.toml
├── src/
│   └── aneurysm/
│       ├── __init__.py
│       ├── config.py
│       ├── models/
│       ├── inference.py
│       └── api/
├── tests/
└── .gitignore
```

其他專案類似

## 常見問題與解決

### Q: 如何處理循環依賴？
**答**: rdxai 不應依賴任何其他倉庫

### Q: 如何共享模型權重？
**答**: 方案 A：放在 rdxai 的 models 目錄
       方案 B：用獨立的模型倉庫
       方案 C：從遠端 URL 下載（推薦用於大型文件）

## 遷移時間表

- **第 1 週**：分析結構、創建 rdxai 框架
- **第 2 週**：提取共用代碼、編寫基類
- **第 3 週**：遷移 brain-aneurysm
- **第 4 週**：遷移其他倉庫、集成測試
- **第 5 週**：文檔、CI/CD 配置
