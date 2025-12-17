# Backend 重構規劃：將邏輯移至 PostgreSQL

## 概述

根據 **Linus 程式碼審查標準**的原則，本規劃旨在將 backend 中可以用 PostgreSQL 完成的邏輯移到資料庫層，使 backend 僅作為 web 與 DB 的連接層，保持簡潔性。

## 核心原則

### 1. 簡潔性（Simplicity）
- Backend 應該只包含必要的 HTTP 處理和資料庫調用
- 複雜的 SQL 查詢應該移到 PostgreSQL functions
- 避免在 Python 中進行複雜的數據處理

### 2. 單一職責（Single Responsibility）
- **Backend**: HTTP 請求/響應處理、驗證、外部服務調用
- **PostgreSQL**: 數據查詢、聚合、狀態檢查、數據驗證

### 3. 效能考量
- 在資料庫層進行聚合可以減少網絡傳輸
- 使用 PostgreSQL 的索引和查詢優化器
- 避免在應用層進行大量數據處理

### 4. 可維護性
- 將業務邏輯集中在資料庫層，更容易維護和測試
- Backend 程式碼更簡潔，更容易理解

## 現有 PostgreSQL Functions

根據 `resource/sql/stored_procedure.sql`，現有以下 functions：

1. `get_stydy_series_ope_no_status(p_ope_no)` - 查詢 series 狀態
2. `get_series_below_threshold_operations(p_ope_no)` - 查詢低於閾值的操作
3. `get_stydy_ope_no_status(p_ope_no)` - 查詢 study 狀態
4. `get_all_studies_status()` - 獲取所有 study 狀態（返回 JSON）

## 需要遷移到 PostgreSQL 的邏輯

### 階段 1: 查詢邏輯遷移（高優先級、低風險）

#### 1.1 `_get_recent_study_uids`
**位置**: `backend/app/sync/service.py:337-397`

**當前邏輯**:
- 使用 GROUP BY 和 MAX 聚合查詢最近更新的 study_uids
- 支援 limit 和 lookback_hours 參數

**建議的 PostgreSQL Function**:
```sql
CREATE OR REPLACE FUNCTION fn_get_recent_study_uids(
    p_limit INTEGER DEFAULT 100,
    p_lookback_hours INTEGER DEFAULT NULL
)
RETURNS TABLE(study_uid VARCHAR) AS $$
BEGIN
    IF p_lookback_hours IS NOT NULL THEN
        RETURN QUERY
        SELECT sub.study_uid
        FROM (
            SELECT
                study_uid,
                MAX(update_time) AS last_update
            FROM dcop_event_bt
            WHERE study_uid IS NOT NULL
            GROUP BY study_uid
        ) AS sub
        WHERE sub.last_update >= (NOW() - (p_lookback_hours || ' hours')::INTERVAL)
        ORDER BY sub.last_update DESC
        LIMIT p_limit;
    ELSE
        RETURN QUERY
        SELECT sub.study_uid
        FROM (
            SELECT
                study_uid,
                MAX(update_time) AS last_update
            FROM dcop_event_bt
            WHERE study_uid IS NOT NULL
            GROUP BY study_uid
        ) AS sub
        ORDER BY sub.last_update DESC
        LIMIT p_limit;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

**Backend 簡化後**:
```python
async def _get_recent_study_uids(
    self,
    session: AsyncSession,
    *,
    limit: Optional[int] = None,
    lookback_hours: Optional[int] = None,
) -> List[str]:
    sql = text("SELECT study_uid FROM fn_get_recent_study_uids(:limit, :lookback_hours)")
    results = await session.execute(sql, {"limit": limit or 100, "lookback_hours": lookback_hours})
    return [row.study_uid for row in results.all()]
```

#### 1.2 `query_studies_pending_completion`
**位置**: `backend/app/sync/service.py:919-1077`

**當前邏輯**:
- 複雜的 CTE 查詢，按 series_uid + rename_dicom_path 分組
- 檢查所有 series 是否達到目標狀態
- 支援按 study_uid 過濾

**建議的 PostgreSQL Function**:
```sql
CREATE OR REPLACE FUNCTION fn_query_studies_pending_completion(
    p_study_uid VARCHAR DEFAULT NULL,
    p_target_status NUMERIC DEFAULT 100.100
)
RETURNS TABLE(
    study_uid VARCHAR,
    series_uid VARCHAR,
    study_id VARCHAR,
    rename_dicom_path VARCHAR,
    ope_no NUMERIC[],
    result_data JSONB[],
    params_data JSONB[],
    create_time TIMESTAMP,
    update_time TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    WITH series_ope_no_status AS (
        SELECT
            dcop_event_bt.study_uid,
            dcop_event_bt.series_uid,
            MAX(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
            COALESCE(
                dcop_event_bt.params_data->>'rename_dicom_path',
                dcop_event_bt.result_data->>'rename_dicom_path'
            ) as rename_dicom_path,
            array_agg(DISTINCT dcop_event_bt.ope_no) as ope_no,
            array_agg(dcop_event_bt.result_data) as result_data,
            array_agg(dcop_event_bt.params_data) as params_data,
            MAX(dcop_event_bt.create_time) as create_time,
            MAX(dcop_event_bt.update_time) as update_time
        FROM dcop_event_bt
        WHERE dcop_event_bt.series_uid IS NOT NULL
          AND (p_study_uid IS NULL OR dcop_event_bt.study_uid = p_study_uid)
        GROUP BY 
            dcop_event_bt.study_uid,
            dcop_event_bt.series_uid,
            COALESCE(
                dcop_event_bt.params_data->>'rename_dicom_path',
                dcop_event_bt.result_data->>'rename_dicom_path'
            )
    ),
    max_study_ope AS (
        SELECT deb.study_id, max(deb.ope_no::numeric) as ope_no 
        FROM dcop_event_bt deb 
        GROUP BY study_id
    )
    SELECT 
        sos.study_uid,
        sos.series_uid,
        sos.study_id,
        sos.rename_dicom_path,
        sos.ope_no,
        sos.result_data,
        sos.params_data,
        sos.create_time,
        sos.update_time
    FROM series_ope_no_status as sos
    INNER JOIN max_study_ope as debb ON sos.study_id = debb.study_id
    WHERE 
        p_target_status > ALL (sos.ope_no::NUMERIC[])
        AND debb.ope_no::NUMERIC <= ANY (sos.ope_no::NUMERIC[])
        AND EXISTS (
            SELECT 1
            FROM unnest(sos.result_data) AS pd
            WHERE pd IS NOT NULL
        );
END;
$$ LANGUAGE plpgsql;
```

#### 1.3 `nifti_tool_get_series_info`
**位置**: `backend/app/sync/service.py:517-786`

**當前邏輯**:
- 複雜的 CTE 查詢，包含多層聚合和過濾
- 檢查 series 狀態是否低於目標 ope_no
- 處理 rename_dicom_path 的多種情況

**建議的 PostgreSQL Function**:
```sql
CREATE OR REPLACE FUNCTION fn_get_series_for_conversion(
    p_study_uid VARCHAR,
    p_target_ope_no NUMERIC
)
RETURNS TABLE(
    study_uid VARCHAR,
    series_uid VARCHAR,
    study_id VARCHAR,
    rename_dicom_path VARCHAR,
    ope_no NUMERIC[],
    result_data JSONB,
    params_data JSONB,
    create_time TIMESTAMP,
    update_time TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    WITH series_rename_status AS (
        SELECT
            dcop_event_bt.study_uid,
            dcop_event_bt.series_uid,
            COALESCE(
                dcop_event_bt.params_data->>'rename_dicom_path',
                dcop_event_bt.result_data->>'rename_dicom_path'
            ) as rename_dicom_path,
            MAX(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
            array_agg(DISTINCT dcop_event_bt.ope_no) as ope_no_array,
            array_agg(dcop_event_bt.result_data ORDER BY dcop_event_bt.create_time DESC) as result_data_array,
            array_agg(dcop_event_bt.params_data ORDER BY dcop_event_bt.create_time DESC) as params_data_array,
            MAX(dcop_event_bt.create_time) as create_time,
            MAX(dcop_event_bt.update_time) as update_time
        FROM dcop_event_bt
        WHERE dcop_event_bt.study_uid = p_study_uid
          AND dcop_event_bt.series_uid IS NOT NULL
          AND dcop_event_bt.result_data IS NOT NULL
        GROUP BY 
            dcop_event_bt.study_uid, 
            dcop_event_bt.series_uid,
            COALESCE(
                dcop_event_bt.params_data->>'rename_dicom_path',
                dcop_event_bt.result_data->>'rename_dicom_path'
            )
    )
    SELECT DISTINCT ON (srs.series_uid, COALESCE(srs.rename_dicom_path, ''))
        srs.study_uid,
        srs.series_uid,
        srs.study_id,
        srs.rename_dicom_path,
        srs.ope_no_array as ope_no,
        srs.result_data_array[1] as result_data,
        srs.params_data_array[1] as params_data,
        srs.create_time,
        srs.update_time
    FROM series_rename_status as srs
    WHERE p_target_ope_no > ALL (srs.ope_no_array::NUMERIC[])
      AND EXISTS (
          SELECT 1
          FROM unnest(srs.result_data_array) AS pd
          WHERE pd IS NOT NULL
      )
    ORDER BY srs.series_uid, COALESCE(srs.rename_dicom_path, ''), srs.create_time DESC;
END;
$$ LANGUAGE plpgsql;
```

#### 1.4 `identify_completed_studies`
**位置**: `backend/app/sync/service.py:1230-1325`

**當前邏輯**:
- 檢查 study 下所有 series 是否都達到目標狀態
- 支援相同 series_uid 但不同 rename_dicom_path 的情況

**建議的 PostgreSQL Function**:
```sql
CREATE OR REPLACE FUNCTION fn_identify_completed_studies(
    p_study_uids VARCHAR[],
    p_target_status NUMERIC DEFAULT 100.100
)
RETURNS TABLE(
    study_uid VARCHAR,
    study_id VARCHAR,
    all_series_completed BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH series_status AS (
        SELECT
            dcop_event_bt.study_uid,
            dcop_event_bt.series_uid,
            COALESCE(
                dcop_event_bt.params_data->>'rename_dicom_path',
                dcop_event_bt.result_data->>'rename_dicom_path'
            ) as rename_dicom_path,
            array_agg(DISTINCT dcop_event_bt.ope_no::NUMERIC) as ope_no_array
        FROM dcop_event_bt
        WHERE dcop_event_bt.study_uid = ANY(p_study_uids)
          AND dcop_event_bt.series_uid IS NOT NULL
        GROUP BY 
            dcop_event_bt.study_uid,
            dcop_event_bt.series_uid,
            COALESCE(
                dcop_event_bt.params_data->>'rename_dicom_path',
                dcop_event_bt.result_data->>'rename_dicom_path'
            )
    ),
    study_summary AS (
        SELECT
            study_uid,
            COUNT(DISTINCT (series_uid, COALESCE(rename_dicom_path, ''))) as total_series,
            COUNT(DISTINCT CASE 
                WHEN p_target_status > ALL(ope_no_array::NUMERIC[]) 
                THEN (series_uid, COALESCE(rename_dicom_path, '')) 
            END) as completed_series
        FROM series_status
        GROUP BY study_uid
    )
    SELECT 
        ss.study_uid,
        MAX(dcop_event_bt.study_id)::VARCHAR as study_id,
        (ss.total_series = ss.completed_series AND ss.total_series > 0) as all_series_completed
    FROM study_summary ss
    JOIN dcop_event_bt ON dcop_event_bt.study_uid = ss.study_uid
    WHERE ss.total_series = ss.completed_series
      AND ss.total_series > 0
    GROUP BY ss.study_uid, ss.total_series, ss.completed_series;
END;
$$ LANGUAGE plpgsql;
```

### 階段 2: 數據驗證遷移（中優先級、中風險）

#### 2.1 `link_prev_study` 驗證邏輯
**位置**: `backend/app/sync/service.py:243-277`

**建議的 PostgreSQL Trigger**:
```sql
CREATE OR REPLACE FUNCTION trg_validate_study_prev_link()
RETURNS TRIGGER AS $$
BEGIN
    -- 驗證邏輯
    IF NEW.study_uid IS NULL THEN
        RAISE EXCEPTION 'study_uid cannot be NULL';
    END IF;
    
    IF NEW.prev_study_uid IS NULL THEN
        RAISE EXCEPTION 'prev_study_uid cannot be NULL';
    END IF;
    
    IF NEW.study_uid = NEW.prev_study_uid THEN
        RAISE EXCEPTION 'study_uid cannot equal prev_study_uid';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_study_prev_link_validation
    BEFORE INSERT OR UPDATE ON study_prev_link
    FOR EACH ROW
    EXECUTE FUNCTION trg_validate_study_prev_link();
```

#### 2.2 `create_event` 配置查詢
**位置**: `backend/app/sync/model.py:74-140`

**建議的 PostgreSQL Function**:
```sql
CREATE OR REPLACE FUNCTION fn_get_ope_config(
    p_tool_id VARCHAR,
    p_status_code VARCHAR DEFAULT NULL,
    p_ope_no VARCHAR DEFAULT NULL
)
RETURNS TABLE(
    ope_no VARCHAR,
    ope_name VARCHAR,
    status_code VARCHAR
) AS $$
BEGIN
    IF p_status_code IS NOT NULL THEN
        RETURN QUERY
        SELECT 
            dcop_conf_bt.ope_no,
            dcop_conf_bt.ope_name,
            dcop_conf_bt.status_code
        FROM dcop_conf_bt
        WHERE dcop_conf_bt.tool_id = p_tool_id
          AND dcop_conf_bt.status_code = p_status_code
          AND dcop_conf_bt.active = 1
        LIMIT 1;
    ELSIF p_ope_no IS NOT NULL THEN
        RETURN QUERY
        SELECT 
            dcop_conf_bt.ope_no,
            dcop_conf_bt.ope_name,
            dcop_conf_bt.status_code
        FROM dcop_conf_bt
        WHERE dcop_conf_bt.tool_id = p_tool_id
          AND dcop_conf_bt.ope_no = p_ope_no
          AND dcop_conf_bt.active = 1
        LIMIT 1;
    ELSE
        RAISE EXCEPTION 'Either p_status_code or p_ope_no must be provided';
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## 必須保留在 Backend 的邏輯

以下邏輯**不應該**移到 PostgreSQL，因為它們涉及外部依賴或應用層職責：

1. **HTTP 請求處理** - FastAPI routers 和請求驗證
2. **外部 API 調用** - httpx 客戶端調用
3. **Redis 快取操作** - 任務狀態快取
4. **任務隊列操作** - funboost 任務推送
5. **事件分組和 URL 映射** - `post_ope_no_task` 中的 URL 映射邏輯
6. **文件路徑處理** - 依賴環境變數的路徑構建
7. **錯誤處理和 HTTP 響應轉換** - 將資料庫錯誤轉換為適當的 HTTP 響應

## 實施優先順序

### 立即實施（高影響、低風險）
1. ✅ `fn_get_recent_study_uids` - 簡單聚合，風險低
2. ✅ `fn_query_studies_pending_completion` - 已經有類似的查詢邏輯

### 短期實施（中影響、中風險）
1. ⏳ `fn_get_series_for_conversion` - 需要仔細測試
2. ⏳ `fn_identify_completed_studies` - 需要確保邏輯一致性

### 長期實施（低影響、高風險）
1. 🔄 將 `create_event` 邏輯移到 PostgreSQL - 需要重構模型類
2. 🔄 使用 triggers 自動化某些操作 - 需要謹慎設計

## 測試策略

### PostgreSQL Functions 測試
1. 使用 `pgTAP` 或手動 SQL 測試腳本
2. 測試邊界情況和錯誤處理
3. 效能測試（EXPLAIN ANALYZE）

### Backend 測試
1. 模擬資料庫調用進行單元測試
2. 整合測試確保函數調用正確
3. 回歸測試確保功能不變

## 命名規範

根據 `resource/sql/90_DB_Object_Naming_Rule_v6.txt`：

- **Functions**: `fn_Function1_(sub_function)`
- **Triggers**: `trg_trigger_name`
- **Tables**: `table_name_bt` (bt = basic table)

## 風險評估

### 低風險
- 簡單的聚合查詢
- 只讀操作

### 中風險
- 複雜的 CTE 查詢
- 涉及多表聯接的邏輯

### 高風險
- 修改現有模型類
- 使用 triggers 自動化操作
- 可能影響現有功能的變更

## 後續步驟

1. **創建 SQL migration 文件** - 包含所有新的 functions 和 triggers
2. **更新 Backend 程式碼** - 將查詢邏輯改為調用 PostgreSQL functions
3. **編寫測試** - 確保功能正確性
4. **效能測試** - 驗證效能提升
5. **文檔更新** - 更新 API 文檔和開發指南

## 參考資料

- Linus 程式碼審查標準（簡潔性、單一職責原則）
- PostgreSQL 官方文檔：Functions、Triggers、CTEs
- 現有 SQL functions: `resource/sql/stored_procedure.sql`
- 命名規範: `resource/sql/90_DB_Object_Naming_Rule_v6.txt`


