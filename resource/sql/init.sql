create type dcop_status as enum ('STUDY_NEW', 'STUDY_TRANSFERRING', 'STUDY_TRANSFER_COMPLETE', 'STUDY_CONVERTING', 'STUDY_CONVERSION_COMPLETE', 'STUDY_INFERENCE_READY', 'STUDY_INFERENCE_QUEUED', 'STUDY_INFERENCE_RUNNING', 'STUDY_INFERENCE_FAILED', 'STUDY_INFERENCE_COMPLETE', 'STUDY_RESULTS_SENT', 'SERIES_NEW', 'SERIES_TRANSFERRING', 'SERIES_TRANSFER_COMPLETE', 'SERIES_CONVERTING', 'SERIES_CONVERSION_COMPLETE', 'SERIES_INFERENCE_FAILED', 'SERIES_INFERENCE_READY', 'SERIES_INFERENCE_QUEUED', 'SERIES_INFERENCE_RUNNING', 'SERIES_INFERENCE_COMPLETE', 'SERIES_RESULTS_SENT');

alter type dcop_status owner to postgres_n;

create table dcop_conf_bt
(
    tool_id     varchar(32) not null,
    ope_no      varchar(7)  not null,
    ope_name    varchar(36),
    status_code varchar(32),
    description varchar,
    active      integer,
    rec_time    timestamp,
    create_time timestamp,
    update_time timestamp,
    constraint pk_dcop_conf_bt
        primary key (tool_id, ope_no)
);

alter table dcop_conf_bt
    owner to postgres_n;

create index ix_dcop_conf_bt_tool_id
    on dcop_conf_bt (tool_id);

create index ix_dcop_conf_bt_status_code
    on dcop_conf_bt (status_code);

create table dcop_event_bt
(
    vsprimarykey varchar(128) not null
        constraint pk_dcop_event_bt
            primary key,
    tool_id      varchar(32)  not null,
    study_uid    varchar(128) not null,
    series_uid   varchar(128),
    study_id     varchar(128),
    event_cate   integer,
    code_name    varchar(32)  not null,
    code_desc    varchar(64),
    params_data  json,
    result_data  json,
    ope_no       varchar(7)   not null,
    ope_name     varchar(36),
    claim_time   timestamp    not null,
    rec_time     timestamp    not null,
    create_time  timestamp    not null,
    update_time  timestamp
);

alter table dcop_event_bt
    owner to postgres_n;

create index ix_dcop_event_bt_tool_id
    on dcop_event_bt (tool_id);

create index ix_dcop_event_bt_series_uid
    on dcop_event_bt (series_uid);

create index ix_dcop_event_bt_vsprimarykey
    on dcop_event_bt (vsprimarykey);

create index ix_dcop_event_bt_study_id
    on dcop_event_bt (study_id);

create index ix_dcop_event_bt_ope_no
    on dcop_event_bt (ope_no);

create index ix_dcop_event_bt_study_uid
    on dcop_event_bt (study_uid);

create table dcop_event_bth
(
    vsprimarykey varchar(128),
    tool_id      varchar(32),
    study_uid    varchar(128),
    series_uid   varchar(128),
    study_id     varchar(128),
    event_cate   integer,
    code_name    varchar(32),
    code_desc    varchar(64),
    params_data  json,
    result_data  json,
    ope_no       varchar(7),
    ope_name     varchar(36),
    claim_time   timestamp,
    rec_time     timestamp,
    create_time  timestamp,
    update_time  timestamp
);

alter table dcop_event_bth
    owner to postgres_n;

create table study_prev_link
(
    id             serial
        constraint pk_study_prev_link
            primary key,
    study_uid      varchar(128) not null,
    prev_study_uid varchar(128) not null,
    created_at     timestamp    not null
);

alter table study_prev_link
    owner to postgres_n;

create unique index ix_study_prev_link_study_uid
    on study_prev_link (study_uid);

create function update_update_time_on_user_task() returns trigger
    language plpgsql
as
$$
BEGIN
    NEW.update_time = now();
    RETURN NEW;
END;
$$;

alter function update_update_time_on_user_task() owner to postgres_n;

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

create function get_all_studies_status()
    returns TABLE(json_data json)
    language plpgsql
as
$$
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
$$;

alter function get_all_studies_status() owner to postgres_n;

create function get_stydy_series_ope_no_status(p_ope_no character varying)
    returns TABLE(study_uid character varying, series_uid character varying, study_id character varying, ope_no character varying[], result_data json[], params_data json[], create_time timestamp without time zone, update_time timestamp without time zone)
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
--         array_agg(DISTINCT dcop_event_bt.study_id) as study_id,
        array_agg(DISTINCT dcop_event_bt.ope_no ) as ope_no,
        array_agg(dcop_event_bt.result_data) as result_data,
        array_agg(dcop_event_bt.params_data) as params_data,
        MAX(dcop_event_bt.create_time) as create_time,
        MAX(dcop_event_bt.update_time) as update_time
    FROM dcop_event_bt
    where dcop_event_bt.series_uid is not null
    group by dcop_event_bt.study_uid,dcop_event_bt.series_uid)
    SELECT sons.study_uid,sons.series_uid,sons.study_id,sons.ope_no,sons.result_data,sons.params_data,sons.create_time,sons.update_time
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

create function get_stydy_ope_no_status(p_ope_no character varying)
    returns TABLE(study_uid character varying, study_id character varying, ope_no character varying[], result_data json[], params_data json[], create_time timestamp without time zone, update_time timestamp without time zone)
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
        array_agg(dcop_event_bt.params_data) as params_data,
        MAX(dcop_event_bt.create_time) as create_time,
        MAX(dcop_event_bt.update_time) as update_time
    FROM dcop_event_bt
    group by dcop_event_bt.study_uid)
    SELECT sons.study_uid,sons.study_id,sons.ope_no,sons.result_data,sons.params_data,sons.create_time,sons.update_time
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

-- SERIES_INFERENCE_TOOL configuration
INSERT INTO dcop_conf_bt (tool_id, ope_no, ope_name, status_code, active, create_time, update_time)
VALUES
    ('SERIES_INFERENCE_TOOL', '300.055', 'Series inference ready', 'SERIES_INFERENCE_READY', 1, NOW(), NOW()),
    ('SERIES_INFERENCE_TOOL', '300.105', 'Series inference queued', 'SERIES_INFERENCE_QUEUED', 1, NOW(), NOW()),
    ('SERIES_INFERENCE_TOOL', '300.155', 'Series inference running', 'SERIES_INFERENCE_RUNNING', 1, NOW(), NOW()),
    ('SERIES_INFERENCE_TOOL', '300.295', 'Series inference complete', 'SERIES_INFERENCE_COMPLETE', 1, NOW(), NOW())
ON CONFLICT (tool_id, ope_no) DO NOTHING;

