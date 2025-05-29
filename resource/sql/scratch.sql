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

CREATE TABLE dcop_collect_bt (
    VsPrimaryKey VARCHAR(128) NOT NULL,
    tool_id      VARCHAR(32),
    study_uid    VARCHAR(128) NOT NULL,
    series_uid   VARCHAR(128),
    patient_id   VARCHAR(64),
    modality     VARCHAR(16),
    status       VARCHAR(32),
    ope_no       VARCHAR(7) NOT NULL,
    ope_name     VARCHAR(36),
    file_path    VARCHAR(256),
    claim_time   TIMESTAMP,
    rec_time     TIMESTAMP,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (VsPrimaryKey)
);
-- 歷史表結構相同
CREATE TABLE dcop_collect_bth AS TABLE dcop_collect_bt WITH NO DATA;

CREATE TABLE dcop_param_tool_bt (
    tool_id                  VARCHAR(32) NOT NULL,
    source_directory         VARCHAR(256),
    destination_directory    VARCHAR(256),
    conversion_format        VARCHAR(32),
    max_parallel_transfers   INTEGER DEFAULT 5,
    max_parallel_conversions INTEGER DEFAULT 3,
    inference_batch_size     INTEGER DEFAULT 10,
    retry_count              INTEGER DEFAULT 3,
    active1a                 INTEGER DEFAULT 1,
    value1a                  VARCHAR(64),
    value1b                  VARCHAR(64),
    rate1a                   FLOAT DEFAULT 1,
    note1                    VARCHAR(255),
    rec_user                 VARCHAR(16),
    rec_time                 TIMESTAMP,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tool_id)
);

-- 插入預設工具參數
INSERT INTO dcop_param_tool_bt
(tool_id, source_directory, destination_directory, conversion_format) VALUES
('DICOM_TOOL', '/mnt/dicom/source', '/mnt/dicom/processed', 'nii');

CREATE TABLE dcop_conf_bt (
    tool_id     VARCHAR(32) NOT NULL,
    ope_no      VARCHAR(7) NOT NULL,
    ope_name    VARCHAR(36),
    status_code VARCHAR(32),
    description TEXT,
    active      INTEGER DEFAULT 1,
    rec_time    TIMESTAMP,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tool_id, ope_no)
);

-- 插入狀態配置
INSERT INTO dcop_conf_bt
(tool_id, ope_no, ope_name, status_code) VALUES
('DICOM_TOOL', '100.020', '檢測到新的study', 'STUDY_NEW'),
('DICOM_TOOL', '100.025', '檢測到新的series', 'SERIES_NEW'),

('DICOM_TOOL', '100.050', 'study傳輸DICOM中', 'STUDY_TRANSFERRING'),
('DICOM_TOOL', '100.055', 'series傳輸DICOM中', 'SERIES_TRANSFERRING'),

('DICOM_TOOL', '100.095', 'series傳輸DICOM完成', 'SERIES_TRANSFER_COMPLETE'),
('DICOM_TOOL', '100.100', 'study傳輸DICOM完成', 'STUDY_TRANSFER_COMPLETE'),


('NIFTI_TOOL', '200.150', 'study轉換NIFTI', 'STUDY_CONVERTING'),
('NIFTI_TOOL', '200.155', 'series轉換NIFTI', 'SERIES_CONVERTING'),

('NIFTI_TOOL', '200.195', 'series轉換NIFTI完成', 'SERIES_CONVERSION_COMPLETE'),
('NIFTI_TOOL', '200.200', 'study轉換NIFTI完成', 'STUDY_CONVERSION_COMPLETE'),


('INFERENCE_TOOL', '300.000', 'study推論失敗', 'STUDY_INFERENCE_FAILED'),
('INFERENCE_TOOL', '300.000', 'series推論失敗', 'STUDY_INFERENCE_FAILED'),

('INFERENCE_TOOL', '300.050', 'study準備推論', 'STUDY_INFERENCE_READY'),
('INFERENCE_TOOL', '300.055', 'series準備推論', 'SERIES_INFERENCE_READY'),


('INFERENCE_TOOL', '300.100', 'study推論排隊中', 'STUDY_INFERENCE_QUEUED'),
('INFERENCE_TOOL', '300.105', 'series推論排隊中', 'SERIES_INFERENCE_QUEUED'),


('INFERENCE_TOOL', '300.150', 'study推論執行中', 'STUDY_INFERENCE_RUNNING'),
('INFERENCE_TOOL', '300.155', 'series推論執行中', 'SERIES_INFERENCE_RUNNING'),

('INFERENCE_TOOL', '300.295', 'series推論完成', 'SERIES_INFERENCE_COMPLETE'),
('INFERENCE_TOOL', '300.300', 'study推論完成', 'STUDY_INFERENCE_COMPLETE'),

('UPLOAD_TOOL', '500.500', 'study結果已發送', 'STUDY_RESULTS_SENT');


-- dcop_event_bt 索引
CREATE INDEX idx_event_bt_study_uid ON dcop_event_bt(study_uid);
CREATE INDEX idx_event_bt_series_uid ON dcop_event_bt(series_uid);
CREATE INDEX idx_event_bt_create_time ON dcop_event_bt(create_time);
CREATE INDEX idx_event_bt_ope_no ON dcop_event_bt(ope_no);

-- dcop_collect_bt 索引
CREATE INDEX idx_collect_bt_study_uid ON dcop_collect_bt(study_uid);
CREATE INDEX idx_collect_bt_series_uid ON dcop_collect_bt(series_uid);
CREATE INDEX idx_collect_bt_create_time ON dcop_collect_bt(create_time);
CREATE INDEX idx_collect_bt_ope_no ON dcop_collect_bt(ope_no);

-- 事件表歸檔觸發器
CREATE OR REPLACE FUNCTION archive_event()
RETURNS TRIGGER AS $$
BEGIN
    -- 將超過一定時間的記錄移到歷史表
    INSERT INTO dcop_event_bth
    SELECT * FROM dcop_event_bt
    WHERE create_time < NOW() - INTERVAL '30 days';

    DELETE FROM dcop_event_bt
    WHERE create_time < NOW() - INTERVAL '30 days';

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_archive_event
AFTER INSERT ON dcop_event_bt
EXECUTE FUNCTION archive_event();

-- 收集表歸檔觸發器
CREATE OR REPLACE FUNCTION archive_collect()
RETURNS TRIGGER AS $$
BEGIN
    -- 將超過一定時間的記錄移到歷史表
    INSERT INTO dcop_collect_bth
    SELECT * FROM dcop_collect_bt
    WHERE create_time < NOW() - INTERVAL '30 days';

    DELETE FROM dcop_collect_bt
    WHERE create_time < NOW() - INTERVAL '30 days';

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_archive_collect
AFTER INSERT ON dcop_collect_bt
EXECUTE FUNCTION archive_collect();

-- 獲取Study最新狀態
CREATE OR REPLACE FUNCTION get_study_latest_status(
    p_study_uid VARCHAR(128)
) RETURNS TABLE (
    status VARCHAR(32),
    ope_no VARCHAR(7),
    ope_name VARCHAR(36)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        status,
        ope_no,
        ope_name
    FROM dcop_collect_bt
    WHERE study_uid = p_study_uid
    ORDER BY rec_time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

CREATE  FUNCTION update_update_time_on_user_task()
RETURNS TRIGGER AS $$
BEGIN
    NEW.update_time = now();
    RETURN NEW;
END;
$$ language 'plpgsql';


CREATE TRIGGER trigger_update_update_time_on_dcop_collect_bt
AFTER INSERT ON dcop_collect_bt
EXECUTE FUNCTION update_update_time_on_user_task();

CREATE TRIGGER trigger_update_update_time_on_dcop_collect_bth
AFTER INSERT ON dcop_collect_bth
EXECUTE FUNCTION update_update_time_on_user_task();

CREATE TRIGGER trigger_update_update_time_on_dcop_conf_bt
AFTER INSERT ON dcop_conf_bt
EXECUTE FUNCTION update_update_time_on_user_task();

CREATE TRIGGER trigger_update_update_time_on_dcop_event_bt
AFTER INSERT ON dcop_event_bt
EXECUTE FUNCTION update_update_time_on_user_task();

CREATE TRIGGER trigger_update_update_time_on_dcop_event_bth
AFTER INSERT ON dcop_event_bth
EXECUTE FUNCTION update_update_time_on_user_task();

CREATE TRIGGER trigger_update_update_time_on_dcop_param_tool_bt
AFTER INSERT ON dcop_param_tool_bt
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
