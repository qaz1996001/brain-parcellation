from enum import Enum


class StudyStatus(Enum):
    NEW = "new"                          # 新檢測到的study
    TRANSFERRING = "transferring"        # DICOM傳輸中
    TRANSFER_COMPLETE = "transfer_complete"  # DICOM傳輸完成    AE-> ORTHANC | ORTHANC-> file system
    CONVERTING = "converting"            # 轉換中              dicom series -> nii series
    CONVERSION_COMPLETE = "conversion_complete"  # 轉換完成    one study all series is conversion complete
    INFERENCE_READY = "inference_ready"  # 準備推論
    INFERENCE_QUEUED = "inference_queued"  # 推論排隊中
    INFERENCE_RUNNING = "inference_running"  # 推論執行中
    INFERENCE_FAILED = "inference_failed"  # 推論失敗
    INFERENCE_COMPLETE = "inference_complete"  # 推論完成
    RESULTS_SENT = "results_sent"        # 結果已發送


class SeriesStatus(Enum):
    NEW = "new"                          # 新檢測到的series
    TRANSFERRING = "transferring"        # DICOM傳輸中
    TRANSFER_COMPLETE = "transfer_complete"  # DICOM傳輸完成
    CONVERTING = "converting"            # 轉換中
    CONVERSION_COMPLETE = "conversion_complete"  # 轉換完成


class InferenceTaskStatus(Enum):
    QUEUED = "queued"                    # 等待中
    RUNNING = "running"                  # 執行中
    FAILED = "failed"                    # 執行失敗
    COMPLETE = "complete"                # 執行完成
    SENT = "sent"                        # 結果已發送
