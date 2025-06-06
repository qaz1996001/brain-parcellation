-- ================================================================
-- PostgreSQL 存儲過程定義
-- ================================================================

-- 1. 檢查特定study的series傳輸完成狀態
CREATE OR REPLACE FUNCTION check_study_series_status(p_study_uid VARCHAR)
RETURNS TABLE (
    study_uid VARCHAR,
    total_series BIGINT,
    completed_series BIGINT,
    transferring_series BIGINT,
    new_series BIGINT,
    transfer_status VARCHAR,
    completion_percentage NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p_study_uid,
        COUNT(DISTINCT series_uid) as total_series,
        COUNT(DISTINCT CASE WHEN code_name = 'SERIES_TRANSFER_COMPLETE' THEN series_uid END) as completed_series,
        COUNT(DISTINCT CASE WHEN code_name = 'SERIES_TRANSFERRING' THEN series_uid END) as transferring_series,
        COUNT(DISTINCT CASE WHEN code_name = 'SERIES_NEW' THEN series_uid END) as new_series,
        CASE
            WHEN COUNT(DISTINCT CASE WHEN code_name = 'SERIES_TRANSFER_COMPLETE' THEN series_uid END) =
                 COUNT(DISTINCT CASE WHEN code_name IN ('SERIES_NEW', 'SERIES_TRANSFERRING', 'SERIES_TRANSFER_COMPLETE') THEN series_uid END)
            THEN '全部完成'::VARCHAR
            ELSE '傳輸中'::VARCHAR
        END as transfer_status,
        ROUND(
            COUNT(DISTINCT CASE WHEN code_name = 'SERIES_TRANSFER_COMPLETE' THEN series_uid END) * 100.0 /
            NULLIF(COUNT(DISTINCT series_uid), 0), 2
        ) as completion_percentage
    FROM dcop_event_bt
    WHERE dcop_event_bt.study_uid = p_study_uid
      AND code_name IN ('SERIES_NEW', 'SERIES_TRANSFERRING', 'SERIES_TRANSFER_COMPLETE');
END;
$$ LANGUAGE plpgsql;

-- 2. 檢查study下每個series的最新狀態
CREATE OR REPLACE FUNCTION get_series_latest_status(p_study_uid VARCHAR)
RETURNS TABLE (
    study_uid VARCHAR,
    series_uid VARCHAR,
    latest_status VARCHAR,
    last_update_time TIMESTAMP,
    status_description VARCHAR,
    minutes_since_update NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH series_status AS (
        SELECT
            dcop_event_bt.study_uid,
            dcop_event_bt.series_uid,
            dcop_event_bt.code_name,
            dcop_event_bt.create_time,
            ROW_NUMBER() OVER (PARTITION BY dcop_event_bt.study_uid, dcop_event_bt.series_uid ORDER BY dcop_event_bt.create_time DESC) as rn
        FROM dcop_event_bt
        WHERE dcop_event_bt.study_uid = p_study_uid
          AND code_name IN ('SERIES_NEW', 'SERIES_TRANSFERRING', 'SERIES_TRANSFER_COMPLETE')
    )
    SELECT
        s.study_uid,
        s.series_uid,
        s.code_name as latest_status,
        s.create_time as last_update_time,
        CASE
            WHEN s.code_name = 'SERIES_TRANSFER_COMPLETE' THEN '已完成'::VARCHAR
            WHEN s.code_name = 'SERIES_TRANSFERRING' THEN '傳輸中'::VARCHAR
            WHEN s.code_name = 'SERIES_NEW' THEN '新建'::VARCHAR
            ELSE '未知'::VARCHAR
        END as status_description,
        ROUND(EXTRACT(EPOCH FROM (NOW() - s.create_time))/60, 2) as minutes_since_update
    FROM series_status s
    WHERE s.rn = 1
    ORDER BY s.create_time;
END;
$$ LANGUAGE plpgsql;

-- 3. 檢查所有study的傳輸完成情況
CREATE OR REPLACE FUNCTION get_all_studies_status()
RETURNS TABLE (
    json_data JSON
) AS $$
BEGIN
    RETURN QUERY
    WITH series_analysis AS (
  SELECT
    study_uid,
    study_id,
    -- 使用聚合函數處理 result_data
    json_agg(result_data) as result,
    COUNT(DISTINCT series_uid) as total_series,
    COUNT(DISTINCT CASE WHEN ope_no = '100.095' THEN series_uid END) as completed_series,
    array_agg(DISTINCT series_uid) as all_series_array,
    array_agg(DISTINCT series_uid) FILTER (WHERE ope_no = '100.095') as completed_series_array
  FROM dcop_event_bt
  WHERE tool_id = 'DICOM_TOOL'
    AND result_data is not null
--     AND study_id IS NOT NULL
    AND ope_no::FLOAT < 100.100
  GROUP BY study_uid, study_id
)
SELECT
    json_build_object(
        'study_uid',study_uid,
        'study_id',study_id,
        'result',result,
        'total_series', total_series,
        'completed_series', completed_series,
        'completed_series_array', COALESCE(completed_series_array, ARRAY[]::text[]),
        'uncompleted_series_array',
        CASE
            WHEN completed_series_array IS NULL THEN all_series_array
            ELSE (SELECT array_agg(uid) FROM unnest(all_series_array) AS uid
                                        WHERE uid != ALL(completed_series_array))
            END
    ) as json_data
from series_analysis;
END;
$$ LANGUAGE plpgsql;

-- 4. 檢查未完成傳輸的series
CREATE OR REPLACE FUNCTION get_incomplete_series()
RETURNS TABLE (
    study_uid VARCHAR,
    series_uid VARCHAR,
    current_status VARCHAR,
    last_update TIMESTAMP,
    minutes_since_last_update NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH series_latest_status AS (
        SELECT
            dcop_event_bt.study_uid,
            dcop_event_bt.series_uid,
            dcop_event_bt.code_name,
            dcop_event_bt.create_time,
            ROW_NUMBER() OVER (PARTITION BY dcop_event_bt.study_uid, dcop_event_bt.series_uid ORDER BY dcop_event_bt.create_time DESC) as rn
        FROM dcop_event_bt
        WHERE code_name IN ('SERIES_NEW', 'SERIES_TRANSFERRING', 'SERIES_TRANSFER_COMPLETE')
    )
    SELECT
        s.study_uid,
        s.series_uid,
        s.code_name as current_status,
        s.create_time as last_update,
        ROUND(EXTRACT(EPOCH FROM (NOW() - s.create_time))/60, 2) as minutes_since_last_update
    FROM series_latest_status s
    WHERE s.rn = 1
      AND s.code_name != 'SERIES_TRANSFER_COMPLETE'
    ORDER BY s.study_uid, s.create_time;
END;
$$ LANGUAGE plpgsql;

-- 5. 檢查指定日期範圍的傳輸狀態
CREATE OR REPLACE FUNCTION get_transfer_status_by_date(
    p_start_date DATE DEFAULT CURRENT_DATE,
    p_end_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    transfer_date DATE,
    study_uid VARCHAR,
    total_series BIGINT,
    completed_series BIGINT,
    in_progress_series BIGINT,
    completion_percentage NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        DATE(dcop_event_bt.create_time) as transfer_date,
        dcop_event_bt.study_uid,
        COUNT(DISTINCT series_uid) as total_series,
        COUNT(DISTINCT CASE WHEN code_name = 'SERIES_TRANSFER_COMPLETE' THEN series_uid END) as completed_series,
        COUNT(DISTINCT CASE WHEN code_name = 'SERIES_TRANSFERRING' THEN series_uid END) as in_progress_series,
        ROUND(
            COUNT(DISTINCT CASE WHEN code_name = 'SERIES_TRANSFER_COMPLETE' THEN series_uid END) * 100.0 /
            NULLIF(COUNT(DISTINCT series_uid), 0), 2
        ) as completion_percentage
    FROM dcop_event_bt
    WHERE DATE(dcop_event_bt.create_time) BETWEEN p_start_date AND p_end_date
      AND code_name IN ('SERIES_NEW', 'SERIES_TRANSFERRING', 'SERIES_TRANSFER_COMPLETE')
    GROUP BY DATE(dcop_event_bt.create_time), dcop_event_bt.study_uid
    ORDER BY transfer_date DESC, dcop_event_bt.study_uid;
END;
$$ LANGUAGE plpgsql;