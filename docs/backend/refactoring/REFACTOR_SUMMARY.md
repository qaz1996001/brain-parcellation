# Backend 重構規劃總結

## 執行摘要

根據 **Linus 程式碼審查標準**，本規劃旨在將 backend 中可以用 PostgreSQL 完成的邏輯移到資料庫層，使 backend 僅作為 web 與 DB 的連接層。

## 核心發現

### 可以移到 PostgreSQL 的邏輯

1. **查詢和聚合邏輯**（高優先級）
   - `_get_recent_study_uids` - 簡單聚合查詢
   - `query_studies_pending_completion` - 複雜 CTE 查詢
   - `nifti_tool_get_series_info` - 複雜的狀態檢查
   - `identify_completed_studies` - 狀態比較邏輯

2. **數據驗證邏輯**（中優先級）
   - `link_prev_study` 驗證 - 可以移到 trigger
   - `create_event` 配置查詢 - 可以移到 function

### 必須保留在 Backend 的邏輯

1. HTTP 請求處理（FastAPI）
2. 外部 API 調用（httpx）
3. Redis 快取操作
4. 任務隊列操作（funboost）
5. 事件分組和 URL 映射
6. 文件路徑處理（依賴環境變數）
7. 錯誤處理和 HTTP 響應轉換

## 實施計劃

### 階段 1: 查詢邏輯遷移（立即實施）

**目標 Functions**:
- `fn_get_recent_study_uids(limit, lookback_hours)`
- `fn_query_studies_pending_completion(study_uid, target_status)`
- `fn_get_series_for_conversion(study_uid, target_ope_no)`
- `fn_identify_completed_studies(study_uids[], target_status)`

**預期效益**:
- 減少網絡傳輸
- 提升查詢效能
- 簡化 backend 程式碼

### 階段 2: 數據驗證遷移（短期實施）

**目標 Components**:
- `trg_validate_study_prev_link` - Trigger 驗證
- `fn_get_ope_config(tool_id, status_code/ope_no)` - 配置查詢

**預期效益**:
- 數據一致性保證
- 減少應用層驗證邏輯

### 階段 3: Backend 簡化（長期實施）

**目標**:
- 移除 Python 中的數據處理邏輯
- 保持 HTTP 處理、驗證、外部服務調用
- 簡化錯誤處理流程

## 風險評估

| 項目 | 風險等級 | 說明 |
|------|---------|------|
| 簡單聚合查詢 | 低 | 只讀操作，容易測試 |
| 複雜 CTE 查詢 | 中 | 需要仔細測試邏輯一致性 |
| 模型類重構 | 高 | 可能影響現有功能 |
| Trigger 自動化 | 高 | 需要謹慎設計 |

## 測試策略

1. **PostgreSQL Functions 測試**
   - 使用 SQL 測試腳本
   - 測試邊界情況
   - 效能測試（EXPLAIN ANALYZE）

2. **Backend 測試**
   - 模擬資料庫調用
   - 整合測試
   - 回歸測試

## 命名規範

根據 `resource/sql/90_DB_Object_Naming_Rule_v6.txt`:
- **Functions**: `fn_Function1_(sub_function)`
- **Triggers**: `trg_trigger_name`
- **Tables**: `table_name_bt`

## 下一步行動

1. ✅ 創建詳細規劃文檔（已完成）
2. ⏳ 創建 SQL migration 文件
3. ⏳ 實施階段 1 的 functions
4. ⏳ 更新 Backend 程式碼
5. ⏳ 編寫測試
6. ⏳ 效能測試和驗證

## 參考文檔

- 詳細規劃: `REFACTOR_PLAN_POSTGRESQL.md`
- 現有 SQL functions: `resource/sql/stored_procedure.sql`
- Backend service: `backend/app/sync/service.py`
- 命名規範: `resource/sql/90_DB_Object_Naming_Rule_v6.txt`


