# QUICK_REFERENCE.md - 快速參考卡

## 🎯 5 分鐘快速開始

### 1️⃣ 備份 (2 分鐘)
```powershell
Copy-Item -Path D:\00_Chen\Task04_git `
          -Destination D:\00_Chen\Task04_git.backup `
          -Recurse
```

### 2️⃣ 分析 (2 分鐘)
```bash
python analyze_repo.py "D:\00_Chen\Task04_git"
```

### 3️⃣ 自動化 (1 分鐘)
```powershell
.\init_multi_repos.ps1
```

---

## 🏗️ 新架構

```
Task04_git_rdx/
├─ rdxai/              # 核心庫 (共用函數)
├─ brain-aneurysm/    # 專案 1 (依賴 rdxai)
├─ brain-cmb/         # 專案 2 (依賴 rdxai)
├─ brain-parcellation/# 專案 3 (依賴 rdxai)
└─ dicom2nii/         # 專案 4 (依賴 rdxai)
```

---

## ✏️ 導入修改

```python
# ❌ 原
from utils.dicom_utils import load_dicom

# ✅ 新
from rdxai.utils.dicom_utils import load_dicom
```

---

## 🔍 常用命令

```bash
# 安裝
pip install -e rdxai[dev]

# 測試
pytest tests/ -v

# 查看結構
tree D:\00_Chen\Task04_git_rdx /L 2

# 檢查依賴
pip check
```

---

## ⏱️ 時間表

- 規劃: 30 分
- rdxai: 2-3 天
- 遷移: 4 天
- 測試: 1 天
- **總計: 7-9 天**

---

## 🆘 常見問題

| 問題 | 解決方案 |
|------|--------|
| 導入失敗 | 檢查新導入路徑 |
| 測試失敗 | pip install -e rdxai[dev] |
| 循環依賴 | rdxai ← brain-aneurysm ✓ |

---

**更多信息查看其他 .md 文件**
