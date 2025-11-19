-- ================================================================
-- PostgreSQL 存儲過程定義
-- ================================================================

create function get_stydy_series_ope_no_status(p_ope_no character varying)
    returns TABLE(study_uid character varying, series_uid character varying, study_id character varying, ope_no character varying[], result_data json[], params_data json[])
    language plpgsql
as
$$
BEGIN
    RETURN QUERY
    WITH  series_ope_no_status AS (
    SELECT
        dcop_event_bt.study_uid,
        dcop_event_bt.series_uid,
        MIN(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
--         array_agg(DISTINCT dcop_event_bt.study_id) as study_id,
        array_agg(DISTINCT dcop_event_bt.ope_no ) as ope_no,
        array_agg(dcop_event_bt.result_data) as result_data,
        array_agg(dcop_event_bt.params_data) as params_data
    FROM dcop_event_bt
    where dcop_event_bt.series_uid is not null
    group by dcop_event_bt.study_uid,dcop_event_bt.series_uid)
    SELECT sons.study_uid,sons.series_uid,sons.study_id,sons.ope_no,sons.result_data,sons.params_data
    FROM series_ope_no_status as sons
    WHERE
        p_ope_no::NUMERIC > ALL (sons.ope_no::NUMERIC[])
      AND
        EXISTS (
                SELECT 1
                FROM unnest(sons.result_data) AS pd
                WHERE pd IS NOT NULL);
END;
$$;

alter function get_stydy_series_ope_no_status(varchar) owner to postgres_n;



create function get_series_below_threshold_operations(p_ope_no character varying)
    returns TABLE(study_uid character varying, series_uid character varying, study_id character varying, ope_no character varying[], result_data json[], params_data json[])
    language plpgsql
as
$$
BEGIN
    RETURN QUERY
    WITH  series_ope_no_status AS (
    SELECT
        dcop_event_bt.study_uid,
        dcop_event_bt.series_uid,
        MAX(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
        array_agg(DISTINCT dcop_event_bt.ope_no ) as ope_no,
        array_agg(dcop_event_bt.result_data) as result_data,
        array_agg(dcop_event_bt.params_data) as params_data
    FROM dcop_event_bt
    where dcop_event_bt.series_uid is not null
    and p_ope_no::NUMERIC > dcop_event_bt.ope_no::NUMERIC
    group by dcop_event_bt.study_uid,dcop_event_bt.series_uid)
    SELECT sons.study_uid,sons.series_uid,sons.study_id,sons.ope_no,sons.result_data,sons.params_data
    FROM series_ope_no_status as sons
    WHERE
        p_ope_no::NUMERIC > ALL (sons.ope_no::NUMERIC[])
      AND
        EXISTS (
                SELECT 1
                FROM unnest(sons.result_data) AS pd
                WHERE pd IS NOT NULL);
END;
$$;
alter function get_series_below_threshold_operations(varchar) owner to postgres_n;


create function get_stydy_ope_no_status(p_ope_no character varying)
    returns TABLE(study_uid character varying, study_id character varying, ope_no character varying[], result_data json[], params_data json[])
    language plpgsql
as
$$
BEGIN
    RETURN QUERY
    WITH  series_ope_no_status AS (
    SELECT
        dcop_event_bt.study_uid,
        MAX(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
--         array_agg(DISTINCT dcop_event_bt.study_id) as study_id,
        array_agg(DISTINCT dcop_event_bt.ope_no ) as ope_no,
        array_agg(dcop_event_bt.result_data) as result_data,
        array_agg(dcop_event_bt.params_data) as params_data
    FROM dcop_event_bt
    group by dcop_event_bt.study_uid)
    SELECT sons.study_uid,sons.study_id,sons.ope_no,sons.result_data,sons.params_data
    FROM series_ope_no_status as sons
    WHERE
        p_ope_no::NUMERIC > ALL (sons.ope_no::NUMERIC[])
      AND
        EXISTS (
                SELECT 1
                FROM unnest(sons.result_data) AS pd
                WHERE pd IS NOT NULL);
END;
$$;

alter function get_stydy_ope_no_status(varchar) owner to postgres_n;

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