# DICOM Sync Module Documentation Update

## 📋 概述

已按照 pandas DataFrame 的文檔標準，為 `@sync` 模組的所有核心檔案添加了詳細的文檔字符串和程式註解。遵循 **Linus Torvalds 風格的程式碼審查標準**，強調 "Good Taste" 設計原則。

## 🎯 更新檔案

### 1. `model.py` - 資料庫模型層
- **DCOPConfModel**: DICOM 工具操作設定模型
  - 詳細的 Attributes 文檔
  - 複合主鍵設計說明
  - 使用場景示例
  
- **DCOPEventModel**: DICOM 同步事件紀錄模型
  - 完整的狀態轉遷說明
  - 所有欄位的語義註解
  - 兩個重要方法的詳細文檔：
    - `create_event()`: 使用狀態碼建立事件
    - `create_event_ope_no()`: 使用 ope_no 建立事件
  - Good Taste 設計說明
  
- **StudyPrevLinkModel**: Study 時序鏈結模型
  - 時序關係維護的設計思想
  - 單向鏈結的優勢

### 2. `schemas.py` - Pydantic 驗證層
- **validate_orthanc_id()**: Orthanc ID 驗證函數
  - 正則表達式解釋
  - 錯誤處理說明
  
- **OrthancID / OpeNo**: 自訂類型別名
  - 驗證規則文檔
  - 使用場景
  
- **PostStudyRequest**: Study 同步請求載體
  - 向後相容性設計說明
  - 所有驗證器的邏輯文檔
  - `resolved_ids()` 方法的用途
  
- **DCOPEventRequest**: 標準事件 Schema
  - webhook 與資料庫互轉的角色
  - 所有欄位說明
  
- **DCOPEventNIFTITOOLRequest**: NIFTI_TOOL 報告 Schema
  - 外部工具集成的設計
  - 最小化載體原則
  
- **DCOPStatus**: 狀態碼列舉
  - 完整的狀態轉遷圖
  - 所有 34 個狀態的詳細說明
  - 重試路徑設計
  
- **StydySeriesOpeNoStatus**: 聚合查詢模型
  - 查詢場景說明
  - 時序狀態歷史的表示

### 3. `routers.py` - FastAPI 路由層
模組級文檔：
- API 設計原則（無狀態、異步優先、分頁支持）
- 端點概覽（按功能分類）
- Good Taste 設計原則應用說明

所有 11 個路由端點的詳細文檔：

#### 健康檢查
- `GET /sync/study`: 服務狀態檢查

#### Study 管理
- `POST /sync/study`: 排程新 Study 同步（支援時序鏈結）

#### 事件查詢
- `GET /sync/ope_no`: 事件日誌查詢（支援搜尋/分頁）
- `POST /sync/ope_no`: 批次寫入事件

#### 狀態轉遷
- `POST /sync/study/transfer`: 檢查傳輸是否完成
- `POST /sync/study/convert`: 檢查轉檔是否完成

#### 工具集成
- `POST /sync/nifti_tool`: 接收 NIFTI_TOOL 回報

#### 快取管理
- `GET /sync/cache`: 列出推論任務快取
- `DELETE /sync/cache`: 清除指定快取

#### 狀態查詢
- `GET /sync/query/study_series_ope_no_status`: Series 操作狀態查詢
- `GET /sync/query/stydy_ope_no_status`: Study 操作狀態查詢
- `GET /sync/query/check_study_series_conversion_complete`: 已完成轉檔 Study 列表

## 📚 文檔特點

### 1. **Pandas DataFrame 風格**
- 模組級 docstring 提供完整概述
- 類級 docstring 包含用途、屬性、方法列表
- 方法級 docstring 包含 Parameters、Returns、Raises、Examples、Notes
- 使用 NumPy/SciPy 標準文檔格式

### 2. **Good Taste 設計說明**
每個主要元素都包含設計原則說明：
- 消除特殊情況（如 `resolved_ids()` 統一介面）
- 資料結構驅動（如使用 DCOPStatus 列舉）
- 扁平化邏輯（Early return、無深嵌套）
- 向後相容性維護

### 3. **清晰的狀態轉遷圖**
- ASCII 藝術狀態轉遷圖
- 每個端點的流程說明
- 檢查點（checkpoint）API 的角色

### 4. **實踐示例**
- 每個主要功能都包含真實使用示例
- cURL/HTTP 請求示例
- Python 程式碼示例
- 常見錯誤場景

### 5. **架構級別的註解**
- 模組之間的依賴關係
- 資料流向
- 非同步工作流程

## 🔍 驗證檢查

```bash
# 代碼質量檢查
ruff check backend/app/sync/

# 結果：✓ 無 linting 錯誤
```

## 📖 如何使用

### 查看完整文檔
```python
import inspect
from backend.app.sync.model import DCOPEventModel

# 列印完整文檔
print(inspect.getdoc(DCOPEventModel))

# IDE 智能提示
# 在 IDE 中懸停查看詳細文檔
```

### IDE 支持
- VS Code: 安裝 Python 擴展，Ctrl+K Ctrl+I 查看文檔
- PyCharm: Ctrl+Q 查看快速文檔
- Cursor: 同 VS Code

## 💡 設計精要

### 核心原則
1. **無特殊情況**: `PostStudyRequest.resolved_ids()` 統一新舊 API
2. **狀態驅動**: 使用 DCOPStatus 列舉而非魔術字符串
3. **最小載體**: NIFTI_TOOL webhook 只傳輸必要資訊
4. **鬆散耦合**: 使用 ope_no 進行反向查詢而非直接 API 呼叫
5. **審計追蹤**: 完整的時間戳記和事件日誌

### 狀態機設計
```
Study Transfer Phase
  ↓
Series Transfer Phase
  ↓
[檢查] → NIFTI Conversion Phase
  ↓
[檢查] → Inference Phase
  ↓
Results Reporting Phase
```

## 🚀 後續工作建議

1. **service.py 文檔化**: 業務邏輯層的詳細說明
2. **API 測試文檔**: 為每個端點寫 test cases
3. **架構文檔**: 繪製完整的系統架構圖
4. **效能最佳化**: 根據文檔識別的 checkpoint API 優化
5. **監控指標**: 基於狀態轉遷圖定義關鍵指標

## 📞 維護建議

- 每次新增狀態碼時更新 DCOPStatus
- 每次新增外部工具時更新 tool_id 的說明
- 保持文檔與程式碼同步（代碼審查時檢查文檔）
- 定期檢查 Examples 是否仍然有效

---

**文檔風格**: Pandas DataFrame 標準 + Linus Torvalds "Good Taste" 評論
**最後更新**: 2025-12-17
**覆蓋範圍**: model.py, schemas.py, routers.py（~1000 行代碼）

