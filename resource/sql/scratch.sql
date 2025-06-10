CREATE TABLE dcop_event_bt (
    VsPrimaryKey VARCHAR(128) NOT NULL,
    tool_id VARCHAR(32) NOT NULL,
    study_uid VARCHAR(128) NOT NULL,
    series_uid VARCHAR(128),
    kind INTEGER, -- 事件類型
    code_name VARCHAR(32) NOT NULL,
    code_desc VARCHAR(64),
    event_cate INTEGER, -- 事件類別
    field_value DOUBLE PRECISION,
    field_data VARCHAR(128),
    ope_no VARCHAR(7) NOT NULL,
    ope_name VARCHAR(36),
    claim_time TIMESTAMP NOT NULL,
    rec_time TIMESTAMP NOT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (VsPrimaryKey)
);

-- 歷史表結構相同
CREATE TABLE dcop_event_bth AS TABLE dcop_event_bt WITH NO DATA;

-- CREATE TABLE dcop_collect_bt (
--     VsPrimaryKey VARCHAR(128) NOT NULL,
--     tool_id      VARCHAR(32),
--     study_uid    VARCHAR(128) NOT NULL,
--     series_uid   VARCHAR(128),
--     patient_id   VARCHAR(64),
--     modality     VARCHAR(16),
--     status       VARCHAR(32),
--     ope_no       VARCHAR(7) NOT NULL,
--     ope_name     VARCHAR(36),
--     file_path    VARCHAR(256),
--     claim_time   TIMESTAMP,
--     rec_time     TIMESTAMP,
--     create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     PRIMARY KEY (VsPrimaryKey)
-- );
-- dcop_collect_bt 索引
-- CREATE INDEX idx_collect_bt_study_uid ON dcop_collect_bt(study_uid);
-- CREATE INDEX idx_collect_bt_series_uid ON dcop_collect_bt(series_uid);
-- CREATE INDEX idx_collect_bt_create_time ON dcop_collect_bt(create_time);
-- CREATE INDEX idx_collect_bt_ope_no ON dcop_collect_bt(ope_no);
-- -- 歷史表結構相同
-- CREATE TABLE dcop_collect_bth AS TABLE dcop_collect_bt WITH NO DATA;

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


-- 插入狀態配置

insert into dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time)
values  ('DICOM_TOOL', '100.020', '檢測到新的study', 'STUDY_NEW', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('DICOM_TOOL', '100.025', '檢測到新的series', 'SERIES_NEW', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('DICOM_TOOL', '100.050', 'study傳輸DICOM中', 'STUDY_TRANSFERRING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('DICOM_TOOL', '100.055', 'series傳輸DICOM中', 'SERIES_TRANSFERRING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('DICOM_TOOL', '100.095', 'series傳輸DICOM完成', 'SERIES_TRANSFER_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('DICOM_TOOL', '100.100', 'study傳輸DICOM完成', 'STUDY_TRANSFER_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('NIFTI_TOOL', '200.150', 'study轉換NIFTI', 'STUDY_CONVERTING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('NIFTI_TOOL', '200.155', 'series轉換NIFTI', 'SERIES_CONVERTING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('NIFTI_TOOL', '200.190', 'series轉換NIFTI跳過', 'SERIES_CONVERSION_SKIP', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('NIFTI_TOOL', '200.195', 'series轉換NIFTI完成', 'SERIES_CONVERSION_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('NIFTI_TOOL', '200.200', 'study轉換NIFTI完成', 'STUDY_CONVERSION_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('INFERENCE_TOOL', '300.000', 'study推論失敗', 'STUDY_INFERENCE_FAILED', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('INFERENCE_TOOL', '300.050', 'study準備推論', 'STUDY_INFERENCE_READY', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('INFERENCE_TOOL', '300.055', 'series準備推論', 'SERIES_INFERENCE_READY', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('INFERENCE_TOOL', '300.100', 'study推論排隊中', 'STUDY_INFERENCE_QUEUED', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('INFERENCE_TOOL', '300.105', 'series推論排隊中', 'SERIES_INFERENCE_QUEUED', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('INFERENCE_TOOL', '300.150', 'study推論執行中', 'STUDY_INFERENCE_RUNNING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('INFERENCE_TOOL', '300.155', 'series推論執行中', 'SERIES_INFERENCE_RUNNING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('INFERENCE_TOOL', '300.295', 'series推論完成', 'SERIES_INFERENCE_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('INFERENCE_TOOL', '300.300', 'study推論完成', 'STUDY_INFERENCE_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('UPLOAD_TOOL', '500.500', 'study結果已發送', 'STUDY_RESULTS_SENT', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('DICOM_TOOL', '100.021', '重新的執行study', 'STUDY_NEW_RE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('DICOM_TOOL', '100.051', 'study重新傳輸DICOM中', 'STUDY_TRANSFERRING_RE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('DICOM_TOOL', '200.151', 'study重新轉換NIFTI', 'STUDY_CONVERTING_RE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828'),
        ('DICOM_TOOL', '300.151', 'study重新推論執行中', 'STUDY_INFERENCE_RUNNING_RE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');


-- dcop_event_bt 索引
CREATE INDEX idx_event_bt_study_uid ON dcop_event_bt(study_uid);
CREATE INDEX idx_event_bt_series_uid ON dcop_event_bt(series_uid);
CREATE INDEX idx_event_bt_create_time ON dcop_event_bt(create_time);
CREATE INDEX idx_event_bt_ope_no ON dcop_event_bt(ope_no);


-- dcop_conf_bt 索引

CREATE INDEX  idx_conf_bt_tool_id ON dcop_conf_bt (tool_id);

CREATE INDEX idx_conf_bt_status_code ON dcop_conf_bt (status_code);


CREATE  FUNCTION update_update_time_on_user_task()
RETURNS TRIGGER AS $$
BEGIN
    NEW.update_time = now();
    RETURN NEW;
END;
$$ language 'plpgsql';


-- CREATE TRIGGER trigger_update_update_time_on_dcop_collect_bt
-- AFTER INSERT ON dcop_collect_bt
-- EXECUTE FUNCTION update_update_time_on_user_task();
--
-- CREATE TRIGGER trigger_update_update_time_on_dcop_collect_bth
-- AFTER INSERT ON dcop_collect_bth
-- EXECUTE FUNCTION update_update_time_on_user_task();


CREATE TRIGGER trigger_update_update_time_on_dcop_event_bt
AFTER INSERT ON dcop_event_bt
EXECUTE FUNCTION update_update_time_on_user_task();

CREATE TRIGGER trigger_update_update_time_on_dcop_event_bth
AFTER INSERT ON dcop_event_bth
EXECUTE FUNCTION update_update_time_on_user_task();


CREATE TYPE dcop_status AS ENUM (
    'STUDY_NEW',
    'STUDY_TRANSFERRING',
    'STUDY_TRANSFER_COMPLETE',
    'STUDY_CONVERTING',
    'STUDY_CONVERSION_COMPLETE',
    'STUDY_INFERENCE_READY',
    'STUDY_INFERENCE_QUEUED',
    'STUDY_INFERENCE_RUNNING',
    'STUDY_INFERENCE_FAILED',
    'STUDY_INFERENCE_COMPLETE',
    'STUDY_RESULTS_SENT',

    'SERIES_NEW',
    'SERIES_TRANSFERRING',
    'SERIES_TRANSFER_COMPLETE',
    'SERIES_CONVERTING',
    'SERIES_CONVERSION_COMPLETE',
    'SERIES_INFERENCE_FAILED',
    'SERIES_INFERENCE_READY',
    'SERIES_INFERENCE_QUEUED',
    'SERIES_INFERENCE_RUNNING',
    'SERIES_INFERENCE_COMPLETE',
    'SERIES_RESULTS_SENT'
);
