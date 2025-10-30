# migration_checklist.md - 詳細遷移檢查清單

## 📋 遷移前檢查

### 分析現有 Task04_git
- [ ] 列出所有 Python 模組和其依賴關係
- [ ] 識別重複代碼
- [ ] 列出所有導入
- [ ] 檢查共用模型和配置
- [ ] 檢查現有的 inference 實現

## 🔍 提取共用代碼到 rdxai

### 優先順序 1: 基礎工具函數
- [ ] DICOM 相關函數
- [ ] NIfTI 相關函數
- [ ] 圖像處理
- [ ] 驗證工具

### 優先順序 2: 配置和模型類
- [ ] 提取 ThreeStageModelConfig
- [ ] 提取 NnUNetModelConfig
- [ ] 提取其他共用配置類
- [ ] 提取 Pydantic 模型

### 優先順序 3: 推理基類
- [ ] 創建 InferenceBase 基類
- [ ] 創建 InferencePipeline 基類
- [ ] 創建 ModelLoader 工具類
- [ ] 創建 nnU-Net 推理處理器

## 🧩 遷移各個專案倉庫

### brain-aneurysm 遷移清單
- [ ] 複製 src/aneurysm/ 到 brain-aneurysm/src/aneurysm/
- [ ] 複製測試文件到 brain-aneurysm/tests/
- [ ] 更新導入路徑
- [ ] 更新 pyproject.toml 中的依賴
- [ ] 運行測試驗證

### brain-cmb、brain-parcellation 遷移類似

## 🧪 測試驗證步驟

### 單元測試
```bash
cd rdxai
pip install -e .[dev]
pytest tests/ -v --cov=rdxai
```

### 集成測試
```bash
python -c "from rdxai import *; print('✓ rdxai')"
python -c "from brain_aneurysm import *; print('✓ aneurysm')"
```

## 📝 遷移後檢查清單

### 代碼質量
- [ ] 運行 linter (flake8)
- [ ] 類型檢查 (mypy)
- [ ] 文檔完成

### Git 管理
- [ ] 初始提交
- [ ] 標籤設定
- [ ] 遠端配置

### 文檔和協作
- [ ] 編寫遷移指南
- [ ] 文檔如何依賴 rdxai
- [ ] CI/CD 配置示例
- [ ] 常見問題解答

## 📊 進度追蹤

| 項目 | rdxai | brain-aneurysm | brain-cmb | 狀態 |
|------|-------|----------------|-----------|------|
| 目錄結構 | ✓ | ✓ | ✓ | ✓ |
| 代碼複製 | 🔄 | 🔄 | 🔄 | |
| 導入修復 | 🔄 | 🔄 | 🔄 | |
| 依賴設置 | ✓ | 🔄 | 🔄 | |
| 測試通過 | ⏳ | ⏳ | ⏳ | |
