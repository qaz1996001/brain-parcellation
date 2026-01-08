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

Refactored following Knuth & Linus principles (CLAUDE.md):
- Pure Functions: Accept paths as parameters, not from environment
- Single Responsibility: Each function does ONE thing well
- Deterministic, testable: No side effects from environment
"""
import glob

import shutil
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from code_ai.pipeline.upload.inference_complete import upload_inference_complete, ModelName

import pydicom

# warnings.filterwarnings("ignore")  # 忽略警告输出
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import logging
import pynvml  # 导包
import tensorflow as tf
autotune = tf.data.experimental.AUTOTUNE

from code_ai import PYTHON3, load_dotenv
from code_ai.config import CodeAIConfig, load_code_ai_config_from_env
from code_ai.pipeline.cmb import CMBServiceTF
from code_ai.pipeline import get_study_id, pipeline_parser, dicom_seg_cmb_file

load_dotenv()

# Module-level logger
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Pure Function Results
# =============================================================================

@dataclass
class TimingResult:
    """計時結果"""
    step_name: str
    elapsed_seconds: float
    success: bool
    message: str = ""

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60


@dataclass
class GPUCheckResult:
    """GPU 檢查結果"""
    available: bool
    memory_usage_rate: float
    message: str


@dataclass
class InferenceResult:
    """推論結果"""
    success: bool
    output_path: Optional[str]
    message: str


@dataclass
class CMBPipelineResult:
    """CMB Pipeline 完整結果"""
    success: bool
    synthseg_path: Optional[str]
    output_nii_path: Optional[str]
    output_json_path: Optional[str]
    timings: Dict[str, TimingResult]
    total_elapsed_seconds: float
    upload_result: Optional[str] = None  # "success", "failed", or None

    @property
    def total_elapsed_minutes(self) -> float:
        return self.total_elapsed_seconds / 60


# =============================================================================
# Pure Functions - Single Responsibility
# =============================================================================

def get_study_instance_uid_from_dicom(dicom_dir: str) -> Optional[str]:
    """
    從 DICOM 目錄讀取 StudyInstanceUID。

    從目錄中找到第一個 DICOM 檔案並讀取其 StudyInstanceUID。

    Args:
        dicom_dir: DICOM 檔案目錄路徑

    Returns:
        StudyInstanceUID 字串，若無法讀取則返回 None
    """
    if not dicom_dir or not os.path.isdir(dicom_dir):
        logger.warning(f"DICOM 目錄不存在或無效: {dicom_dir}")
        return None

    try:
        # 嘗試找到第一個 DICOM 檔案
        for root, dirs, files in os.walk(dicom_dir):
            for filename in files:
                # 跳過非 DICOM 檔案 (常見的非 DICOM 副檔名)
                if filename.lower().endswith(('.json', '.xml', '.txt', '.nii', '.gz', '.jpg', '.png')):
                    continue

                filepath = os.path.join(root, filename)
                try:
                    # 嘗試讀取 DICOM 檔案
                    dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                    if hasattr(dcm, 'StudyInstanceUID'):
                        study_uid = str(dcm.StudyInstanceUID)
                        logger.info(f"從 DICOM 讀取 StudyInstanceUID: {study_uid}")
                        return study_uid
                except Exception:
                    # 不是有效的 DICOM 檔案，繼續嘗試下一個
                    continue

        logger.warning(f"在目錄中找不到有效的 DICOM 檔案: {dicom_dir}")
        return None

    except Exception as e:
        logger.error(f"讀取 DICOM StudyInstanceUID 失敗: {e}")
        return None


def setup_logger(path_log: str, log_level: int = logging.INFO) -> str:
    """
    設定 logger 並建立 log 檔案。

    Pure function: 接收明確路徑參數，不從環境變數讀取。

    Args:
        path_log: Log 檔案目錄路徑
        log_level: Logging 等級

    Returns:
        log_file: 建立的 log 檔案路徑
    """
    localt = time.localtime(time.time())
    time_str_short = f"{localt.tm_year}{str(localt.tm_mon).rjust(2, '0')}{str(localt.tm_mday).rjust(2, '0')}"
    log_file = os.path.join(path_log, f'{time_str_short}.log')

    if not os.path.isfile(log_file):
        with open(log_file, "a+") as f:
            f.write("")

    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=log_level, filename=log_file, filemode='a', format=FORMAT)

    return log_file


def check_gpu_memory(gpu_n: int, threshold: float = 0.6) -> GPUCheckResult:
    """
    檢查 GPU 記憶體使用率。

    Pure function: 只檢查 GPU 狀態，不修改任何設定。

    Args:
        gpu_n: GPU 設備編號
        threshold: 記憶體使用率閾值 (預設 0.6 = 60%)

    Returns:
        GPUCheckResult: GPU 檢查結果
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage_rate = memory_info.used / memory_info.total

        available = usage_rate < threshold
        message = f"GPU {gpu_n} 記憶體使用率: {usage_rate:.2%}"

        if not available:
            message += f" (超過閾值 {threshold:.0%}，GPU 記憶體不足)"

        # raise RuntimeError(f"[check_gpu_memory] Debug")
        return GPUCheckResult(
            available=available,
            memory_usage_rate=usage_rate,
            message=message
        )
    except Exception as e:
        return GPUCheckResult(
            available=False,
            memory_usage_rate=1.0,
            message=f"GPU 檢查失敗: {str(e)}"
        )


def configure_tensorflow_gpu(gpu_n: int) -> bool:
    """
    設定 TensorFlow GPU 配置。

    Args:
        gpu_n: GPU 設備編號

    Returns:
        bool: 設定是否成功
    """
    try:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if not gpus or gpu_n >= len(gpus):
            logger.error(f"GPU {gpu_n} 不存在，可用 GPU 數量: {len(gpus)}")
            return False

        tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_n], True)
        return True
    except Exception as e:
        logger.error(f"TensorFlow GPU 設定失敗: {str(e)}")
        return False


def run_synthseg_inference(
    swan_file: str,
    t1_file: str,
    output_dir: str
) -> InferenceResult:
    """
    執行 SynthSeg 推論 (呼叫 main.py)。

    Pure function: 接收明確路徑，執行推論。

    Args:
        swan_file: SWAN 輸入檔案路徑
        t1_file: T1 模板檔案路徑
        output_dir: 輸出目錄

    Returns:
        InferenceResult: 推論結果
    """
    try:
        gpu_line = '{} {} -i {} --template {} --output {} --all False --CMB TRUE'.format(
            PYTHON3,
            os.path.join(os.path.dirname(__file__), 'main.py'),
            swan_file,
            t1_file,
            output_dir
        )

        logger.info(f"執行 SynthSeg 推論: {gpu_line}")
        os.system(gpu_line)

        # 檢查輸出檔案
        output_pattern = f'{output_dir}/synthseg_*SWAN_original_CMB*.nii.gz'
        output_files = glob.glob(output_pattern)

        if not output_files:
            return InferenceResult(
                success=False,
                output_path=None,
                message=f"找不到推論輸出檔案: {output_pattern}"
            )

        output_path = output_files[0]
        if not os.path.exists(output_path):
            return InferenceResult(
                success=False,
                output_path=None,
                message=f"輸出檔案不存在: {output_path}"
            )

        return InferenceResult(
            success=True,
            output_path=output_path,
            message="SynthSeg 推論完成"
        )

    except Exception as e:
        return InferenceResult(
            success=False,
            output_path=None,
            message=f"SynthSeg 推論失敗: {str(e)}"
        )


def process_output_files(
    ID: str,
    temp_path_str: str,
    path_output: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    處理輸出檔案：複製並重命名。

    Pure function: 接收明確路徑，處理檔案。

    Args:
        ID: Study ID
        temp_path_str: 暫存檔案路徑
        path_output: 輸出目錄

    Returns:
        Tuple of (synthseg_path, output_nii_path, output_json_path)
    """
    try:
        path_output_dir = os.path.join(path_output, ID)
        os.makedirs(path_output_dir, exist_ok=True)

        # 處理檔名
        temp_path_basename = os.path.basename(temp_path_str)
        temp_path_basename = temp_path_basename.replace(
            get_study_id(temp_path_basename), ''
        ).replace('__', '_')

        synthseg_output_path = os.path.join(path_output_dir, temp_path_basename)
        shutil.copy(temp_path_str, synthseg_output_path)

        output_nii_path = os.path.join(path_output_dir, 'Pred_CMB.nii.gz')
        output_json_path = os.path.join(path_output_dir, 'Pred_CMB.json')

        return synthseg_output_path, output_nii_path, output_json_path

    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"處理輸出檔案失敗: {str(e)}")
        return None, None, None


def run_cmb_classification(
    swan_path_str: str,
    temp_path_str: str,
    output_nii_path_str: str,
    output_json_path_str: str
) -> bool:
    """
    執行 CMB 分類。

    Pure function: 接收明確路徑，執行分類。

    Args:
        swan_path_str: SWAN 輸入檔案路徑
        temp_path_str: SynthSeg 暫存輸出路徑
        output_nii_path_str: NIfTI 輸出路徑
        output_json_path_str: JSON 輸出路徑

    Returns:
        bool: 分類是否成功
    """
    try:
        cmb_pipeline = CMBServiceTF()
        cmb_pipeline.cmb_classify(
            swan_path_str=swan_path_str,
            temp_path_str=temp_path_str,
            output_nii_path_str=output_nii_path_str,
            output_json_path_str=output_json_path_str
        )
        return True
    except Exception as e:
        logger.error(f"CMB 分類失敗: {str(e)}")
        return False


def timed_execution(func, step_name: str, *args, **kwargs) -> Tuple[any, TimingResult]:
    """
    計時執行函數的通用包裝器。

    Args:
        func: 要執行的函數
        step_name: 步驟名稱（用於 logging）
        *args, **kwargs: 傳遞給函數的參數

    Returns:
        Tuple of (function_result, TimingResult)
    """
    start_time = time.time()
    logger.info(f"[開始] {step_name}")

    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        timing = TimingResult(
            step_name=step_name,
            elapsed_seconds=elapsed,
            success=True,
            message=f"{step_name} 完成"
        )
        logger.info(f"[完成] {step_name} | 耗時: {elapsed:.2f} 秒")
        return result, timing

    except Exception as e:
        elapsed = time.time() - start_time
        timing = TimingResult(
            step_name=step_name,
            elapsed_seconds=elapsed,
            success=False,
            message=f"{step_name} 失敗: {str(e)}"
        )
        logger.error(f"[失敗] {step_name} | 耗時: {elapsed:.2f} 秒 | 錯誤: {str(e)}")
        return None, timing


# =============================================================================
# Main Pipeline Function (Orchestrator)
# =============================================================================

def pipeline_cmb(
    ID: str,
    swan_file: str,
    t1_file: str,
    path_output: str,
    path_processModel: str = '/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_CMB/',
    path_log: str = '/mnt/d/wsl_ubuntu/pipeline/sean/log/',
    gpu_n: int = 0,
    config: Optional[CodeAIConfig] = None,
    ai_app_inference_complete: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    model_name: ModelName = "cmb_model",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    CMB (Cerebral Microbleed) detection pipeline.

    重構後的 pipeline 遵循 Knuth & Linus 原則：
    - 拆分為多個純函數，各司其職
    - 每個步驟都有計時
    - 接收明確參數，不依賴環境變數
    - 成功/失敗都會呼叫 upload_inference_complete 通知平台

    Supports dual-mode configuration:
    - Old pattern: pipeline_cmb(ID, swan, t1, output, ...) - uses explicit params
    - New pattern: pipeline_cmb(ID, swan, t1, output, config=config) - uses config

    Args:
        ID: Study identifier
        swan_file: Path to SWAN (SWI) input file
        t1_file: Path to T1 input file
        path_output: Output directory
        path_processModel: Processing model directory
        path_log: Log directory
        gpu_n: GPU device number
        config: Optional CodeAIConfig for pure function pattern
        ai_app_inference_complete: API URL for inference complete notification
        study_instance_uid: DICOM Study Instance UID (用於失敗時通知)
        model_name: 模型名稱 ("cmb_model" 或 "aneurysm_model")

    Returns:
        Tuple of (synthseg_path, output_nii_path, output_json_path) or (None, None, None)
    """
    pipeline_start_time = time.time()
    timings: Dict[str, TimingResult] = {}

    # 定義失敗處理的輔助函數
    def handle_failure(error_msg: str) -> Tuple[None, None, None]:
        """處理失敗情況：記錄錯誤並上傳失敗通知"""
        logging.error(error_msg)
        if ai_app_inference_complete and study_instance_uid:
            upload_inference_complete(
                url=ai_app_inference_complete,
                success=False,
                study_instance_uid=study_instance_uid,
                model_name=model_name,
            )
            logging.info(f"已發送推論失敗通知: study_uid={study_instance_uid}")
        return None, None, None

    # Dual-mode: use config if provided, otherwise use explicit parameters
    if config is not None:
        path_log = str(config.paths.path_log)
        path_json = str(config.paths.path_json)
        path_code = str(config.paths.path_code)
        gpu_n = config.model.gpu_n

    # 當使用 GPU 有錯時才確認
    tf_logger = tf.get_logger()
    tf_logger.setLevel(logging.ERROR)

    # Step 1: 設定 Logger
    log_file, timing = timed_execution(
        setup_logger, "設定 Logger",
        path_log
    )
    timings["setup_logger"] = timing

    logging.info(f'=== CMB Pipeline 開始 ID: {ID} ===')

    # 建立處理目錄
    path_processID = os.path.join(path_processModel, ID)
    os.makedirs(path_processID, exist_ok=True)

    try:
        # Step 2: 檢查 GPU 記憶體
        gpu_check, timing = timed_execution(
            check_gpu_memory, "檢查 GPU 記憶體",
            gpu_n
        )
        timings["check_gpu"] = timing

        if not gpu_check or not gpu_check.available:
            return handle_failure(f'!!! {ID} GPU 記憶體不足: {gpu_check.message if gpu_check else "檢查失敗"}')

        logging.info(gpu_check.message)

        # Step 3: 設定 TensorFlow GPU
        gpu_configured, timing = timed_execution(
            configure_tensorflow_gpu, "設定 TensorFlow GPU",
            gpu_n
        )
        timings["configure_gpu"] = timing

        if not gpu_configured:
            return handle_failure(f'!!! {ID} TensorFlow GPU 設定失敗')

        # Step 4: 執行 SynthSeg 推論
        inference_result, timing = timed_execution(
            run_synthseg_inference, "SynthSeg 推論",
            swan_file, t1_file, path_processID
        )
        timings["synthseg_inference"] = timing

        if not inference_result or not inference_result.success:
            return handle_failure(f'!!! {ID} SynthSeg 推論失敗: {inference_result.message if inference_result else "未知錯誤"}')

        temp_path_str = inference_result.output_path

        # Step 5: 處理輸出檔案
        output_paths, timing = timed_execution(
            process_output_files, "處理輸出檔案",
            ID, temp_path_str, path_output
        )
        timings["process_output"] = timing

        if not output_paths or output_paths[0] is None:
            return handle_failure(f'!!! {ID} 處理輸出檔案失敗')

        synthseg_path, output_nii_path, output_json_path = output_paths

        # Step 6: 執行 CMB 分類
        classification_success, timing = timed_execution(
            run_cmb_classification, "CMB 分類",
            swan_file, temp_path_str, output_nii_path, output_json_path
        )
        timings["cmb_classification"] = timing

        if not classification_success:
            return handle_failure(f'!!! {ID} CMB 分類失敗')

        # 計算總耗時
        total_elapsed = time.time() - pipeline_start_time

        # 輸出計時摘要
        logging.info(f'=== CMB Pipeline 完成 ID: {ID} ===')
        logging.info('--- 各步驟耗時摘要 ---')
        for step_name, timing_result in timings.items():
            status = "✓" if timing_result.success else "✗"
            logging.info(f'  {status} {timing_result.step_name}: {timing_result.elapsed_seconds:.2f} 秒')
        logging.info(f'--- 總耗時: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分鐘) ---')

        return synthseg_path, output_nii_path, output_json_path

    except Exception as e:
        logging.error("Catch an exception.", exc_info=True)
        return handle_failure(f'!!! {ID} Pipeline 執行錯誤: {str(e)}')


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    # 計時開始
    main_start_time = time.time()

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus, device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    config :CodeAIConfig = load_code_ai_config_from_env()

    parser = pipeline_parser()
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs
    InputsDicomDir = args.InputsDicomDir
    path_output = str(args.Output_folder)

    path_code         = os.getenv("PATH_CODE", config.paths.path_code)
    path_process      = os.getenv("PATH_PROCESS", config.paths.path_process)
    path_processModel = os.path.join(path_process, 'Deep_CMB')
    path_json         = os.getenv("PATH_JSON", config.paths.path_json)
    path_log          = os.getenv("PATH_LOG", config.paths.path_log)
    ai_app_inference_complete = os.getenv("AI_APP_INFERENCE_COMPLETE")
    # 從 InputsDicomDir 讀取 study_instance_uid
    study_instance_uid = get_study_instance_uid_from_dicom(InputsDicomDir)
    if not study_instance_uid:
        logging.warning(f"無法從 DICOM 目錄讀取 StudyInstanceUID，使用 ID 作為備用: {ID}")
        study_instance_uid = ID

    gpu_n = int(os.getenv("GPU_N", 0))
    swan_path_str = Inputs[0]
    t1_path_str = Inputs[1]

    # 建置資料夾
    os.makedirs(path_processModel, exist_ok=True)
    os.makedirs(path_json, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)
    os.makedirs(path_output, exist_ok=True)

    ## 設定 main 的 logger
    log_file = setup_logger(path_log)
    logging.info(f'=== CMB Pipeline CLI 開始執行 ID: {ID} ===')

    ## 執行 pipeline (已整合 upload_inference_complete)
    cmb_path_str, output_nii_path_str, output_json_path_str = pipeline_cmb(
        ID=ID,
        swan_file=swan_path_str,
        t1_file=t1_path_str,
        path_output=path_output,
        path_log=path_log,
        path_processModel=path_processModel,
        gpu_n=gpu_n,
        ai_app_inference_complete=ai_app_inference_complete,
        study_instance_uid=study_instance_uid,
        model_name="cmb_model",
    )

    # 後處理：DICOM-SEG 轉換 (若推論成功)
    if output_nii_path_str is not None:
        dicom_seg_result, timing = timed_execution(dicom_seg_cmb_file,
                                                   'dicom_seg_cmb_file',
                                                   ID, InputsDicomDir, output_nii_path_str, path_output
                        )

        # Step: 發送推論成功通知 (在 DICOM-SEG 轉換完成之後)
        if ai_app_inference_complete:

            rdx_json_path = output_json_path_str.replace('.json', '_rdx_cmb_pred_json.json')
            upload_result, timing = timed_execution(upload_inference_complete,
                                                    'upload_inference_complete',
                                                    ai_app_inference_complete,
                                                    True,
                                                    rdx_json_path,
                                                    "cmb_model"
                                                    )
            # upload_result = upload_inference_complete(
            #     url=ai_app_inference_complete,
            #     success=True,
            #     output_json_path=rdx_json_path,
            #     model_name="cmb_model",
            # )
            if upload_result:
                logging.info(f"已發送推論成功通知: inference_id={upload_result.inferenceId}")
            else:
                logging.warning("發送推論成功通知失敗，但推論結果已保存")

    # 計時結束
    main_elapsed = time.time() - main_start_time
    logging.info(f'=== CMB Pipeline CLI 執行完成 ID: {ID} | 總耗時: {main_elapsed:.2f} 秒 ({main_elapsed/60:.2f} 分鐘) ===')
