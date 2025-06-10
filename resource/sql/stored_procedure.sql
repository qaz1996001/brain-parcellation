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
