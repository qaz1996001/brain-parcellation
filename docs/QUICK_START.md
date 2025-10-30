# QUICK_START.md - Task04_git 多倉庫拆分快速開始指南

## 📚 文檔導航

本次提供的文件和腳本:
1. **Task04_git_repo_split_plan.md** - 完整的重構計劃和策略
2. **migration_checklist.md** - 詳細的遷移檢查清單
3. **init_multi_repos.ps1** - 自動化初始化腳本 (PowerShell)
4. **analyze_repo.py** - 代碼分析工具 (Python)
5. **inference_base_template.py** - InferenceBase 類別範本
6. **rdxai_init_template.py** - rdxai/__init__.py 範本
7. **本文檔** - 快速開始指南

---

## 🚀 快速開始（5 分鐘）

### 步驟 1: 備份 (2 分鐘)
```powershell
Copy-Item -Path D:\00_Chen\Task04_git `
          -Destination D:\00_Chen\Task04_git.backup `
          -Recurse
```

### 步驟 2: 分析 (2 分鐘)
```bash
cd D:\00_Chen\Task04_git_rdx
python .\docs\analyze_repo.py "D:\00_Chen\Task04_git"
```
📊 生成: `repo_analysis_report.json`

### 步驟 3: 自動化 (1 分鐘)
```powershell
cd D:\00_Chen\Task04_git_rdx
& .\docs\init_multi_repos.ps1
```
📁 創建: 完整結構

---

## 🏗️ 新架構

```
Task04_git_rdx/
│
├─ rdxai/              ← 核心庫 (所有專案的基礎)
│  ├─ base/           (基類)
│  ├─ utils/          (工具函數)
│  ├─ models/         (數據模型)
│  ├─ inference/      (推理框架)
│  └─ exceptions/     (異常類)
│
├─ brain-aneurysm/    ← 專案 1 (依賴 rdxai)
├─ brain-cmb/         ← 專案 2 (依賴 rdxai)
├─ brain-parcellation/← 專案 3 (依賴 rdxai)
└─ dicom2nii/         ← 專案 4 (依賴 rdxai)
```

---

## 📦 環境設置

### 安裝所有倉庫
```bash
cd D:\00_Chen\Task04_git_rdx

# 核心庫
pip install -e rdxai[dev]

# 各專案
pip install -e brain-aneurysm[dev]
pip install -e brain-cmb[dev]
pip install -e brain-parcellation[dev]
pip install -e dicom2nii[dev]
```

### 驗證安裝
```bash
python -c "from rdxai import *; print('✓')"
python -c "from brain_aneurysm import *; print('✓')"
```

---

## ✏️ 導入修改速查表

### DICOM 工具
```python
# ❌ 原
from utils.dicom_utils import load_dicom

# ✅ 新
from rdxai.utils.dicom_utils import load_dicom
```

### 基類和管道
```python
# ❌ 原
from base_classes import InferenceBase

# ✅ 新
from rdxai.base import InferenceBase, InferencePipeline
```

---

## 🧪 測試三部曲

### 1️⃣ 導入測試
```bash
python -c "from rdxai import *; print('✓ rdxai')"
python -c "from brain_aneurysm import *; print('✓ aneurysm')"
```

### 2️⃣ 單元測試
```bash
cd rdxai
pytest tests/ -v --cov=rdxai

cd ../brain-aneurysm
pytest tests/ -v
```

### 3️⃣ 集成測試
```bash
pytest tests/integration/ -v
```

✅ 全部通過 = 遷移成功！

---

## 🔍 常用命令

### 測試
```bash
cd rdxai && pytest tests/ -v
cd ../brain-aneurysm && pytest tests/ -v
```

### 查看目錄結構
```bash
tree D:\00_Chen\Task04_git_rdx /L 2
```

### 檢查依賴
```bash
pip check
pip list | findstr "rdxai"
```

---

## 🆘 常見問題

### Q: 導入失敗？
**A**: 檢查是否用了新導入路徑
```python
# 查看導入路徑速查表
```

### Q: 測試失敗？
**A**: 確保 rdxai 已安裝
```bash
pip install -e rdxai[dev]
```

### Q: 循環依賴？
**A**: 規則: brain-aneurysm → rdxai ✓ (正確)
     rdxai → brain-aneurysm ✗ (錯誤)

---

## ⏱️ 時間表

| 階段 | 任務 | 時間 |
|------|------|------|
| 規劃 | 分析 + 自動化初始化 | 30 分 |
| rdxai | 基類 + 工具函數 + 測試 | 2-3 天 |
| 遷移 | 各專案遷移 + 導入修改 | 4 天 |
| 最終 | 集成測試 + 文檔 + CI/CD | 1 天 |
| **總計** | | **7-9 天** |

---

完整文檔見其他 .md 文件。
