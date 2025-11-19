create table dcop_conf_bt
(
    tool_id     varchar(32) not null,
    ope_no      varchar(7)  not null,
    ope_name    varchar(36),
    status_code varchar(32),
    description text,
    active      integer   default 1,
    rec_time    timestamp,
    create_time timestamp default CURRENT_TIMESTAMP,
    update_time timestamp default CURRENT_TIMESTAMP,
    primary key (tool_id, ope_no)
);

alter table dcop_conf_bt
    owner to postgres_n;

INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('DICOM_TOOL', '100.020', '檢測到新的study', 'STUDY_NEW', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('DICOM_TOOL', '100.025', '檢測到新的series', 'SERIES_NEW', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('DICOM_TOOL', '100.050', 'study傳輸DICOM中', 'STUDY_TRANSFERRING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('DICOM_TOOL', '100.055', 'series傳輸DICOM中', 'SERIES_TRANSFERRING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('DICOM_TOOL', '100.095', 'series傳輸DICOM完成', 'SERIES_TRANSFER_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('DICOM_TOOL', '100.100', 'study傳輸DICOM完成', 'STUDY_TRANSFER_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('NIFTI_TOOL', '200.150', 'study轉換NIFTI', 'STUDY_CONVERTING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('NIFTI_TOOL', '200.155', 'series轉換NIFTI', 'SERIES_CONVERTING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('NIFTI_TOOL', '200.190', 'series轉換NIFTI跳過', 'SERIES_CONVERSION_SKIP', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');

INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('NIFTI_TOOL', '200.195', 'series轉換NIFTI完成', 'SERIES_CONVERSION_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('NIFTI_TOOL', '200.200', 'study轉換NIFTI完成', 'STUDY_CONVERSION_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('INFERENCE_TOOL', '300.000', 'study推論失敗', 'STUDY_INFERENCE_FAILED', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('INFERENCE_TOOL', '300.050', 'study準備推論', 'STUDY_INFERENCE_READY', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('INFERENCE_TOOL', '300.055', 'series準備推論', 'SERIES_INFERENCE_READY', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('INFERENCE_TOOL', '300.100', 'study推論排隊中', 'STUDY_INFERENCE_QUEUED', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('INFERENCE_TOOL', '300.105', 'series推論排隊中', 'SERIES_INFERENCE_QUEUED', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('INFERENCE_TOOL', '300.150', 'study推論執行中', 'STUDY_INFERENCE_RUNNING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('INFERENCE_TOOL', '300.155', 'series推論執行中', 'SERIES_INFERENCE_RUNNING', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('INFERENCE_TOOL', '300.295', 'series推論完成', 'SERIES_INFERENCE_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('INFERENCE_TOOL', '300.300', 'study推論完成', 'STUDY_INFERENCE_COMPLETE', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
INSERT INTO public.dcop_conf_bt (tool_id, ope_no, ope_name, status_code, description, active, rec_time, create_time, update_time) VALUES ('UPLOAD_TOOL', '500.500', 'study結果已發送', 'STUDY_RESULTS_SENT', null, 1, null, '2025-05-29 12:49:40.626828', '2025-05-29 12:49:40.626828');
