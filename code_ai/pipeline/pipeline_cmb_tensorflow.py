#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-03-28 10:40

Python 3.10.13
tensorflow==2.14.0
numpy==1.26.0
SimpleITK==2.3.1
nibabel==5.1.0
scikit-image==0.22.0
pynvml==12.0.0

@author: sean Ho
"""

import glob
import shutil
import warnings
from typing import Any, Dict, List, Optional, Tuple
from code_ai.utils.inference import InferenceEnum

warnings.filterwarnings("ignore")  # 忽略警告輸出
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import logging
import pynvml  # 導入套件
import tensorflow as tf

autotune = tf.data.experimental.AUTOTUNE
from code_ai import PYTHON3, load_dotenv
from code_ai.pipeline.cmb import CMBServiceTF
from code_ai.pipeline import get_study_id, pipeline_parser
from code_ai.pipeline.dicomseg import dicom_seg_cmb_file
from code_ai.pipeline.upload import upload_dicom_seg, upload_json


load_dotenv()


def pipeline_cmb(
    ID: str,
    swan_file: str,
    t1_file: str,
    path_output: str,
    path_processModel="/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_CMB/",
    path_log="/mnt/d/wsl_ubuntu/pipeline/sean/log/",
    gpu_n=0,
    needFollowup: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # 當使用gpu有錯時才確認
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    # 以log紀錄資訊，先建置log
    localt = time.localtime(time.time())  # 取得 struct_time 格式的時間
    # 以下加上時間註記，建立唯一性
    time_str_short = (
        str(localt.tm_year)
        + str(localt.tm_mon).rjust(2, "0")
        + str(localt.tm_mday).rjust(2, "0")
    )
    log_file = os.path.join(path_log, time_str_short + ".log")
    if not os.path.isfile(log_file):  # 如果log檔不存在
        f = open(log_file, "a+")  # a+	可讀可寫	建立，不覆蓋
        f.write("")  # 寫入檔案，設定為空
        f.close()  # 執行完結束

    FORMAT = "%(asctime)s %(levelname)s %(message)s"  # 日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(
        level=logging.INFO, filename=log_file, filemode="a", format=FORMAT
    )

    logging.info("!!! Pred CMB call.")
    path_processID = os.path.join(path_processModel, ID)  # 前處理dicom路徑(test case)
    os.makedirs(path_processID, exist_ok=True)  # 如果資料夾不存在就建立
    print(ID, " Start...")

    try:
        # %% Deep learning相關
        pynvml.nvmlInit()  # 初始化
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            gpu_n
        )  # 取得GPU i的handle，後續通過handle來處理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通過handle取得GPU i的資訊
        gpumRate = memoryInfo.used / memoryInfo.total

        if gpumRate < 0.6:
            # plt.ion()    # 開啟互動模式，畫圖都是一閃就過
            # 一些記憶體的配置
            # print(keras.__version__)
            # print(tf.__version__)
            gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
            tf.config.experimental.set_visible_devices(
                devices=gpus[gpu_n], device_type="GPU"
            )
            # print(gpus, cpus)
            tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

            gpu_line = (
                "{} {} -i {} --template {} --output {} --all False --CMB TRUE".format(
                    PYTHON3,
                    os.path.join(os.path.dirname(__file__), "main.py"),
                    swan_file,
                    t1_file,
                    path_processID,
                )
            )

            os.system(gpu_line)

            temp_path_str = glob.glob(
                "{}/synthseg_*SWAN_original_CMB*.nii.gz".format(path_processID)
            )[0]
            if not os.path.exists(temp_path_str):
                raise FileNotFoundError(temp_path_str)
            path_output_dir = os.path.join(path_output, ID)
            os.makedirs(path_output_dir, exist_ok=True)
            temp_path_basename = os.path.basename(temp_path_str)
            temp_path_basename = temp_path_basename.replace(
                get_study_id(temp_path_basename), ""
            ).replace("__", "_")
            synthseg_temp_path_basename = os.path.join(
                path_output_dir, temp_path_basename
            )
            shutil.copy(
                temp_path_str, os.path.join(path_output_dir, temp_path_basename)
            )
            output_nii_path_str = os.path.join(path_output_dir, "Pred_CMB.nii.gz")
            output_json_path_str = os.path.join(path_output_dir, "Pred_CMB.json")
            #
            cmb_pipeline = CMBServiceTF()
            cmb_pipeline.cmb_classify(
                swan_path_str=swan_file,
                temp_path_str=temp_path_str,
                output_nii_path_str=output_nii_path_str,
                output_json_path_str=output_json_path_str,
            )
            logging.info("!!! " + str(ID) + " gpu_cmb finish.")
            print(f"{ID} gpu_cmb finish.")

            return (
                synthseg_temp_path_basename,
                output_nii_path_str,
                output_json_path_str,
            )
        else:
            logging.error("!!! " + str(ID) + " Insufficient GPU Memory.")

    except Exception:
        logging.error("!!! " + str(ID) + " gpu have error code.")
        logging.error("Catch an exception.", exc_info=True)

    return None, None, None


def _trigger_followup_pipeline(
    current_study_id: str,
    current_output_path: str,
    current_nii_path: str,
    current_synthseg_path: str,
    needFollowup: List[Dict[str, Any]],
    path_process: str,
) -> None:
    """
    觸發 followup 比對 pipeline

    參數說明：
    - current_study_id: 當前檢查 (baseline)
    - needFollowup: API 傳入的以前檢查列表 (followup candidates)
    - path_process: 處理資料夾的根路徑

    needFollowup 結構（由 API 傳入）：
    [
        {
            "patient_id": str,
            "study_date": str,  # "2025-07-21"
            "orthancStudyUid": str,
            "models": [{"model_type": int, "type": str}, ...],
            ...
        },
        ...
    ]

    轉換邏輯：
    - Baseline = 當前檢查 (--ID)
    - Followup = needFollowup 列表中的以前檢查
    - 對每個 followup 執行 pipeline_followup
    """
    from code_ai.pipeline.followup import pipeline_followup

    # Baseline (當前檢查) 的資訊
    baseline_id = current_study_id
    baseline_inputs = [current_nii_path, current_synthseg_path]
    baseline_output_dir = os.path.join(current_output_path, baseline_id)

    print(f"[Followup] Processing {len(needFollowup)} followup candidates...")

    # 對每個 followup 候選進行比對
    for followup_item in needFollowup:
        try:
            # 從 API 資料提取 patient_id 和 study_date
            patient_id = followup_item.get("patient_id", "")
            study_date_raw = followup_item.get("study_date", "")  # "2025-07-21"

            if not patient_id or not study_date_raw:
                print(f"[Followup] Skipping item: missing patient_id or study_date")
                continue

            # 轉換日期格式: "2025-07-21" -> "20250721"
            study_date = study_date_raw.replace("-", "")
            print(f"[Followup] Looking for: {patient_id}_{study_date}_MR_* in {current_output_path}")

            # 在 path_output 中匹配 followup 目錄
            # Pattern: {patient_id}_{study_date}_MR_*
            followup_study_id = _find_study_id_by_pattern(
                current_output_path, patient_id, study_date
            )

            if not followup_study_id:
                print(
                    f"[Followup] No matching directory found for "
                    f"patient_id={patient_id}, study_date={study_date}"
                )
                continue

            # 建構 followup 路徑
            followup_output_dir = os.path.join(current_output_path, followup_study_id)
            followup_pred_path = os.path.join(followup_output_dir, "Pred_CMB.nii.gz")
            followup_synthseg_path = _find_synthseg_file(followup_output_dir)

            if not followup_synthseg_path:
                print(f"[Followup] No synthseg file found in {followup_output_dir}")
                continue

            print(f"[Followup] Found: followup_study_id={followup_study_id}")
            followup_inputs = [followup_pred_path, followup_synthseg_path]
            followup_json = os.path.join(followup_output_dir, "Pred_CMB_platform_json.json")

            print(f"[Followup] Triggering: baseline={baseline_id}, followup={followup_study_id}")

            # 呼叫 pipeline_followup
            pipeline_followup(
                baseline_ID=baseline_id,
                baseline_Inputs=baseline_inputs,
                baseline_DicomDir="",  # 若需要可從環境變數或參數取得
                baseline_DicomSegDir="",
                baseline_json=os.path.join(baseline_output_dir, "Pred_CMB_platform_json.json"),
                followup_ID=followup_study_id,
                followup_Inputs=followup_inputs,
                followup_DicomSegDir="",
                followup_json=followup_json,
                path_output=current_output_path,
                model="CMB",
                path_process=path_process,
            )

            print(f"[Followup] Completed: baseline={baseline_id}, followup={followup_study_id}")
            logging.info(f"[Followup] Completed: baseline={baseline_id}, followup={followup_study_id}")

        except Exception as e:
            print(f"[Followup] Failed for item {followup_item}: {e}")
            logging.error(f"[Followup] Failed for item {followup_item}: {e}")
            logging.error("Catch an exception.", exc_info=True)


def _find_study_id_by_pattern(output_path: str, patient_id: str, study_date: str) -> Optional[str]:
    """
    在 output_path 中找到匹配 {patient_id}_{study_date}_MR_* 的目錄

    Returns:
        完整的 study_id (如 "15397285_20250721_MR_21407150021") 或 None
    """
    pattern = f"{patient_id}_{study_date}_MR_*"
    matches = glob.glob(os.path.join(output_path, pattern))

    if not matches:
        return None

    # 如果有多個匹配，取第一個（或可以加入更精確的邏輯）
    matched_dir = matches[0]
    return os.path.basename(matched_dir)


def _find_synthseg_file(output_dir: str) -> Optional[str]:
    """
    在 output_dir 中找到 synthseg 檔案

    Pattern: synthseg_*SWAN*CMB*.nii.gz
    """
    pattern = os.path.join(output_dir, "synthseg_*SWAN*CMB*.nii.gz")
    matches = glob.glob(pattern)

    if matches:
        return matches[0]

    # Fallback: 找任何 synthseg 開頭的檔案
    pattern = os.path.join(output_dir, "synthseg_*.nii.gz")
    matches = glob.glob(pattern)

    return matches[0] if matches else None


# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == "__main__":
    # /mnt/e/pipeline/sean/rename_nifti/15397285_20260129_MR_21412080021/Pred_CMB_platform_json.json
    # /mnt/e/pipeline/sean/rename_nifti/15397285_20260129_MR_21412080021
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    tf.config.experimental.set_visible_devices(devices=gpus, device_type="GPU")
    # print(gpus, cpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # python /mnt/d/wsl_ubuntu/pipeline/sean/code/pipeline_cmb_tensorflow.py --ID 00971591_20160503_MR_250425032 --Inputs /mnt/d/wsl_ubuntu/pipeline/sean/example_input/00971591_20160503_MR_250425032/SWAN.nii.gz /mnt/d/wsl_ubuntu/pipeline/sean/example_input/00971591_20160503_MR_250425032/T1FLAIR_AXI.nii.gz
    parser = pipeline_parser()
    parser.add_argument('--needFollowup', type=str, default=None,
                        help='Followup config as JSON string (priority: CLI > env var NEED_FOLLOWUP_JSON)')
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    InputsDicomDir = args.InputsDicomDir  # 將列表合併為字符串，保留順序
    # 下面設定各個路徑
    path_output = str(args.Output_folder)

    path_code = os.getenv("PATH_CODE")
    path_process = os.getenv("PATH_PROCESS")
    path_processModel = os.path.join(path_process, "Deep_CMB")
    path_json = os.getenv("PATH_JSON")
    path_log = os.getenv("PATH_LOG")

    gpu_n = int(os.getenv("GPU_N", 0))
    swan_path_str = Inputs[0]
    t1_path_str = Inputs[1]

    # 讀取 needFollowup: CLI > env var (backward compatible)
    import json
    needFollowup_json = args.needFollowup or os.getenv("NEED_FOLLOWUP_JSON")
    needFollowup = json.loads(needFollowup_json) if needFollowup_json else None

    # 建置資料夾
    os.makedirs(
        path_processModel, exist_ok=True
    )  # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_output, exist_ok=True)

    # 直接當作function的輸入
    cmb_path_str, output_nii_path_str, output_json_path_str = pipeline_cmb(
        ID=ID,
        swan_file=swan_path_str,
        t1_file= t1_path_str,
        path_output= path_output,
        path_processModel= path_processModel,
        path_log= path_log,
        gpu_n= gpu_n,
        needFollowup=needFollowup,
    )

    if output_nii_path_str is not None:
        stdout, stderr = dicom_seg_cmb_file(
            ID, InputsDicomDir, output_nii_path_str, path_output
        )

        # ===== Followup 觸發邏輯（在 dicom_seg 之後執行）=====
        print(f"[Followup] needFollowup = {needFollowup}")
        if needFollowup:
            _trigger_followup_pipeline(
                current_study_id=ID,
                current_output_path=path_output,
                current_nii_path=output_nii_path_str,
                current_synthseg_path=cmb_path_str,
                needFollowup=needFollowup,
                path_process=path_process,
            )
    #     upload_dicom_seg(
    #         path_output,
    #         output_nii_path_str,
    #     )
    #     upload_json(ID, InferenceEnum.CMB)
    # dicom_seg
    # if cmb_path_str is not None:
    #     stdout, stderr = dicom_seg_cmb_file(ID,InputsDicomDir,cmb_path_str,path_output )
    #     upload_dicom_seg(path_output,cmb_path_str,)
