"""
Unified inference task pipeline.

This module provides a single entry point for both Study Level and Series Level inference.
The dispatch is based on the presence of 'series_uids' in func_params.

Design Philosophy (Linus Style):
- Data structure drives behavior (series_uids presence determines level)
- Single queue for natural GPU mutual exclusion (qps=1)
- Minimal code change: original logic wrapped in _task_study_pipeline_inference
- No external coordination needed (Redis lock, DB semaphore, etc.)

Usage:
    Study Level (existing behavior):
        task_pipeline_inference.push({
            'nifti_study_path': '/path/to/study',
            'dicom_study_path': '/path/to/dicom',
            'study_uid': '...',
            ...
        })

    Series Level - Direct Mode (NIFTI already exists):
        task_pipeline_inference.push({
            'series_uids': ['series_a', 'series_b'],
            'nifti_series_paths': ['/path/a.nii.gz', '/path/b.nii.gz'],
            'model_id': 'aneurysm_v1',
            'inference_id': 'uuid-xxx',
            'study_uid': '...',
            ...
        })

    Series Level - Conversion Mode (raw DICOM → NIFTI → inference):
        task_pipeline_inference.push({
            'series_uids': ['series_a', 'series_b'],
            'raw_dicom_series_paths': ['/raw/series_a', '/raw/series_b'],
            'needs_conversion': True,
            'model_id': 'aneurysm_v1',
            'path_rename_dicom': '/path/to/rename_dicom',  # optional, fallback to env
            'path_rename_nifti': '/path/to/rename_nifti',  # optional, fallback to env
            'study_uid': '...',
            ...
        })
"""

import json
import os
import pathlib
import subprocess
from typing import Dict, List, Optional, Any
from uuid import uuid4

from funboost import Booster, fct
from funboost.core.serialization import Serialization
import nb_log

logger = nb_log.LogManager("task_pipeline_inference_queue").get_logger_and_add_handlers(
    log_filename="task_pipeline_inference_queue.log"
)

from backend.app.sync.schemas import DCOPStatus, DCOPEventRequest
from backend.app.sync.urls import SYNC_PROT_OPE_NO
from code_ai.task.params import BoosterParamsMyAI, BoosterParamsMyRABBITMQ
from code_ai.utils.inference import build_inference_cmd
from code_ai.utils.inference.schema import InferenceCmd, InferenceCmdItem


def _extract_path_from_params(
    func_params: Dict[str, Any], param_name: str, env_var_name: str
) -> str:
    """Extract path from task parameters with fallback to environment variable.

    This utility function implements the parameter injection attern for task path configuration.
    It first checks func_params for the path, then falls back to environment variables if not found.
    This enables dual deployment where backends can pass paths as parameters to shared workers.

    Args:
        func_params: Task parameters dictionary from dispatcher
        param_name: Parameter key to extract (e.g., 'path_process')
        env_var_name: Environment variable name for fallback (e.g., 'PATH_PROCESS')

    Returns:
        str: The path value from parameters or environment

    Raises:
        ValueError: If path not found in parameters or environment
    """
    path_value = func_params.get(param_name)

    if path_value is None:
        logger.warning(
            f"Path parameter '{param_name}' not found in task parameters, "
            f"falling back to environment variable '{env_var_name}'."
        )
        path_value = os.getenv(env_var_name)

        if path_value is None:
            raise ValueError(
                f"{param_name} must be provided in task parameters or "
                f"{env_var_name} environment variable must be set"
            )

    return path_value


# =============================================================================
# 統一入口 (Unified Entry Point)
# =============================================================================


@Booster(
    BoosterParamsMyAI(
        queue_name="task_pipeline_inference_queue",
        qps=1,
    )
)
def task_pipeline_inference(func_params: Dict[str, Any]):
    """
    統一推論入口：根據 func_params 結構判斷 Study/Series Level。

    Linus: "Data structure is the documentation"
    - 有 series_uids → Series Level（新功能）
    - 沒有 series_uids → Study Level（原有邏輯）

    單一 Queue + qps=1 = 天然 GPU 互斥，無需額外協調機制。
    """
    if "series_uids" in func_params:
        logger.info(
            f"[Series Level] Processing {len(func_params['series_uids'])} series"
        )
        return _task_series_pipeline_inference(func_params)
    else:
        study_id = func_params.get("study_id", "unknown")
        logger.info(f"[Study Level] Processing study: {study_id}")
        return _task_study_pipeline_inference(func_params)


# =============================================================================
# Study Level（原有邏輯，完全保留）
# =============================================================================


def _task_study_pipeline_inference(func_params: Dict[str, Any]):
    """
    Study Level 推論 - 處理整個 study 的所有 series。

    這是原有的 task_pipeline_inference 邏輯，完全不變。
    """
    from code_ai.task.task_dicom2nii import call_post_httpx

    upload_data_api_url = func_params.get("upload_data_api_url")
    if upload_data_api_url is None:
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        if upload_data_api_url is None:
            raise ValueError(
                "upload_data_api_url must be provided in task parameters or "
                "UPLOAD_DATA_API_URL environment variable must be set"
            )

    path_process = _extract_path_from_params(
        func_params, "path_process", "PATH_PROCESS"
    )
    path_cmd_tools = os.path.join(path_process, "Deep_cmd_tools")
    path_json = _extract_path_from_params(func_params, "path_json", "PATH_JSON")
    path_log = _extract_path_from_params(func_params, "path_log", "PATH_LOG")
    path_root = _extract_path_from_params(func_params, "path_root", "PATH_ROOT")

    os.makedirs(path_json, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)
    os.makedirs(path_cmd_tools, exist_ok=True)

    nifti_study_path = func_params["nifti_study_path"]
    dicom_study_path = func_params["dicom_study_path"]

    inference_item_cmd = build_inference_cmd(
        pathlib.Path(nifti_study_path),
        pathlib.Path(dicom_study_path),
        path_root=path_root,
    )

    if inference_item_cmd.cmd_items:
        cmd_output_path = os.path.join(
            path_cmd_tools, f"{inference_item_cmd.cmd_items[0].study_id}_cmd.json"
        )
    else:
        temp_id = os.path.basename(dicom_study_path)
        cmd_output_path = os.path.join(path_cmd_tools, f"{temp_id}_cmd.json")

    with open(cmd_output_path, "w") as f:
        f.write(json.dumps(inference_item_cmd.model_dump()["cmd_items"]))

    study_uid = func_params.get("study_uid", None)
    study_id = func_params.get("study_id", None)
    api_url = f"{upload_data_api_url}{SYNC_PROT_OPE_NO}"

    if study_uid and study_id:
        dcop_event = DCOPEventRequest(
            study_uid=study_uid,
            series_uid=None,
            study_id=study_id,
            ope_no=DCOPStatus.STUDY_INFERENCE_RUNNING.value,
            tool_id="INFERENCE_TOOL",
            params_data={
                "inference_item_cmd": inference_item_cmd.cmd_items,
                "func_params": func_params,
                "task": fct.function_result_status.get_status_dict(),
            },
        )
        call_post_httpx.push(
            {
                "url": api_url,
                "data": dcop_event.model_dump_json(),
            }
        )

    result_list = []
    for inference_item in inference_item_cmd.cmd_items:
        process = subprocess.Popen(
            args=inference_item.cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        result_list.append((inference_item.cmd_str, stdout.decode(), stderr.decode()))

    result = Serialization.to_json_str(result_list)

    if study_uid and study_id:
        dcop_event = DCOPEventRequest(
            study_uid=study_uid,
            series_uid=None,
            study_id=study_id,
            ope_no=DCOPStatus.STUDY_INFERENCE_COMPLETE.value,
            tool_id="INFERENCE_TOOL",
            params_data={
                "inference_item_cmd": inference_item_cmd.cmd_items,
                "func_params": func_params,
                "task": fct.function_result_status.get_status_dict(),
            },
            result_data={"result": result},
        )
        call_post_httpx.push(
            {
                "url": api_url,
                "data": dcop_event.model_dump_json(),
            }
        )

    return result


# =============================================================================
# Series Level（新增邏輯）
# =============================================================================


def _validate_series_params(func_params: Dict[str, Any]) -> None:
    """
    驗證 Series Level 參數。

    支持兩種模式：
    1. 直接模式: nifti_series_paths 已存在
    2. 轉換模式: needs_conversion=True + raw_dicom_series_paths

    Linus: "Fail fast and fail loud. Don't try to be clever."
    """
    # 基本必要字段
    base_required = ["series_uids", "model_id"]
    missing_base = [f for f in base_required if f not in func_params]
    if missing_base:
        raise ValueError(
            f"Series Level inference missing required fields: {missing_base}. "
            f"Required: {base_required}"
        )

    series_uids = func_params["series_uids"]
    if not isinstance(series_uids, list):
        raise ValueError(f"series_uids must be a list, got {type(series_uids)}")

    needs_conversion = func_params.get("needs_conversion", False)

    if needs_conversion:
        # 轉換模式: 需要 raw_dicom_series_paths
        if "raw_dicom_series_paths" not in func_params:
            raise ValueError(
                "When needs_conversion=True, raw_dicom_series_paths is required"
            )
        raw_paths = func_params["raw_dicom_series_paths"]
        if not isinstance(raw_paths, list):
            raise ValueError(
                f"raw_dicom_series_paths must be a list, got {type(raw_paths)}"
            )
        if len(series_uids) != len(raw_paths):
            raise ValueError(
                f"series_uids ({len(series_uids)} items) and "
                f"raw_dicom_series_paths ({len(raw_paths)} items) must have same length"
            )
        logger.info(
            f"Validated Series Level params (conversion mode): "
            f"{len(series_uids)} series, model_id={func_params['model_id']}"
        )
    else:
        # 直接模式: 需要 nifti_series_paths
        if "nifti_series_paths" not in func_params:
            raise ValueError(
                "nifti_series_paths is required when needs_conversion is False or not set"
            )
        nifti_paths = func_params["nifti_series_paths"]
        if not isinstance(nifti_paths, list):
            raise ValueError(
                f"nifti_series_paths must be a list, got {type(nifti_paths)}"
            )
        if len(series_uids) != len(nifti_paths):
            raise ValueError(
                f"series_uids ({len(series_uids)} items) and "
                f"nifti_series_paths ({len(nifti_paths)} items) must have same length"
            )
        logger.info(
            f"Validated Series Level params (direct mode): "
            f"{len(series_uids)} series, model_id={func_params['model_id']}"
        )


def _convert_single_series_to_nifti(
    raw_dicom_path: str,
    output_dicom_base: str,
    output_nifti_base: str,
    series_uid: str,
    study_id: str,
) -> tuple:
    """
    單個 series 的 DICOM 轉換: raw_dicom → rename_dicom → nifti。

    這是同步執行的轉換函數，直接調用底層轉換邏輯，
    而非使用 funboost 的異步 push（避免額外的 queue 調度開銷）。

    Args:
        raw_dicom_path: 原始 DICOM 目錄路徑
        output_dicom_base: rename_dicom 基礎目錄
        output_nifti_base: nifti 輸出基礎目錄
        series_uid: Series UID
        study_id: Study ID

    Returns:
        tuple: (rename_dicom_path, nifti_path) 或 (None, None) 如果失敗
    """
    from code_ai.task.task_dicom2nii import (
        rename_dicom_file,
        copy_dicom_file,
        ConvertManager,
        _execute_dcm2niix,
    )

    raw_dicom_dir = pathlib.Path(raw_dicom_path)
    output_dicom_base_path = pathlib.Path(output_dicom_base)
    output_nifti_base_path = pathlib.Path(output_nifti_base)

    if not raw_dicom_dir.exists():
        logger.error(f"Raw DICOM directory not found: {raw_dicom_path}")
        return None, None

    # Step 1: raw_dicom → rename_dicom
    # 收集所有 DICOM 文件
    dicom_files = list(raw_dicom_dir.rglob("*.dcm"))
    if not dicom_files:
        dicom_files = [f for f in raw_dicom_dir.rglob("*") if f.is_file()]

    if not dicom_files:
        logger.error(f"No DICOM files found in: {raw_dicom_path}")
        return None, None

    # 處理每個 DICOM 文件
    rename_dicom_path = None
    for dicom_file in dicom_files:
        try:
            rename_result = rename_dicom_file(
                dicom_file,
                ConvertManager.processing_strategy_list,
                ConvertManager.modality_processing_strategy,
                ConvertManager.mr_acquisition_type_processing_strategy,
            )
            copy_result = copy_dicom_file(
                rename_result, dicom_file, output_dicom_base_path
            )
            if copy_result:
                # 從結果中提取 rename_dicom 路徑
                # copy_dicom_file 返回 JSON tuple: [input_path, output_path]
                result_tuple = json.loads(copy_result)
                # result_tuple[1] = output path (renamed DICOM file)
                rename_dicom_path = pathlib.Path(result_tuple[1]).parent
        except Exception as e:
            logger.warning(f"Failed to process DICOM file {dicom_file}: {e}")
            continue

    if rename_dicom_path is None:
        logger.error(f"Failed to rename DICOM files for series: {series_uid}")
        return None, None

    logger.info(f"DICOM renamed: {raw_dicom_path} → {rename_dicom_path}")

    # Step 2: rename_dicom → nifti (使用純函數，避免重複造輪子)
    # 計算輸出 NIFTI 路徑
    study_folder = rename_dicom_path.parent
    series_name = rename_dicom_path.name
    nifti_study_path = output_nifti_base_path / study_folder.name
    nifti_series_path = nifti_study_path / series_name
    nifti_file_path = pathlib.Path(f"{nifti_series_path}.nii.gz")

    # 調用 _execute_dcm2niix 純函數（重用 task_dicom2nii.py 的邏輯）
    logger.info(f"Running dcm2niix for: {rename_dicom_path} → {nifti_file_path}")

    result = _execute_dcm2niix(
        series_path=rename_dicom_path,
        output_series_path=nifti_series_path,
        output_series_file_path=nifti_file_path,
        timeout=300,
    )

    # 檢查結果
    if nifti_file_path.exists():
        logger.info(f"NIFTI created: {nifti_file_path}")
        return str(rename_dicom_path), str(nifti_file_path)
    else:
        # 嘗試找到任何生成的 .nii.gz 文件（dcm2niix 可能使用不同的檔名）
        nifti_files = list(nifti_series_path.parent.glob(f"{series_name}*.nii.gz"))
        if nifti_files:
            logger.info(f"NIFTI created: {nifti_files[0]}")
            return str(rename_dicom_path), str(nifti_files[0])

    logger.error(f"NIFTI file not created for series: {series_uid}, result: {result}")
    return str(rename_dicom_path), None


def _batch_convert_series_to_nifti(
    raw_dicom_paths: List[str],
    series_uids: List[str],
    output_dicom_base: str,
    output_nifti_base: str,
    study_id: str,
    upload_data_api_url: str = None,
) -> tuple:
    """
    批量轉換多個 series: raw_dicom → rename_dicom → nifti。

    Args:
        raw_dicom_paths: 原始 DICOM 路徑列表
        series_uids: Series UID 列表
        output_dicom_base: rename_dicom 基礎目錄
        output_nifti_base: nifti 輸出基礎目錄
        study_id: Study ID
        upload_data_api_url: API URL（用於發送轉換事件）

    Returns:
        tuple: (dicom_series_paths, nifti_paths) 兩個列表
    """
    from code_ai.task.task_dicom2nii import call_post_httpx
    from backend.app.sync.schemas import DCOPEventRequest, DCOPStatus
    from backend.app.sync import urls as sync_urls

    dicom_series_paths = []
    nifti_paths = []

    for i, (raw_path, series_uid) in enumerate(zip(raw_dicom_paths, series_uids)):
        logger.info(f"Converting series {i + 1}/{len(series_uids)}: {series_uid}")

        dicom_path, nifti_path = _convert_single_series_to_nifti(
            raw_dicom_path=raw_path,
            output_dicom_base=output_dicom_base,
            output_nifti_base=output_nifti_base,
            series_uid=series_uid,
            study_id=study_id,
        )

        dicom_series_paths.append(dicom_path)
        nifti_paths.append(nifti_path)

        # 發送轉換完成事件（可選）
        if upload_data_api_url and dicom_path:
            try:
                status = (
                    DCOPStatus.SERIES_CONVERSION_COMPLETE.value
                    if nifti_path
                    else DCOPStatus.SERIES_CONVERSION_SKIP.value
                )
                dcop_event = DCOPEventRequest(
                    study_uid=None,  # Series Level 可能沒有 study_uid
                    series_uid=series_uid,
                    study_id=study_id,
                    ope_no=status,
                    tool_id="NIFTI_TOOL",
                    result_data={
                        "raw_dicom_path": raw_path,
                        "rename_dicom_path": dicom_path,
                        "nifti_path": nifti_path,
                    },
                )
                api_url = f"{upload_data_api_url}{sync_urls.SYNC_PROT_OPE_NO}"
                call_post_httpx.push(
                    {"url": api_url, "data": dcop_event.model_dump_json()}
                )
            except Exception as e:
                logger.warning(f"Failed to send conversion event: {e}")

    return dicom_series_paths, nifti_paths


def _resolve_model_id_to_inference_enum(model_id: str):
    """
    將 model_id (UUID 或字串) 映射到 InferenceEnum。

    Args:
        model_id: 模型識別碼，可以是：
            - UUID 字串（例如 '3fa85f64-5717-4562-b3fc-2c963f66afa6'）
            - 模型名稱（例如 'CMB', 'Aneurysm'）

    Returns:
        InferenceEnum: 對應的推論枚舉值

    Raises:
        ValueError: 如果 model_id 無法映射到已知模型
    """
    from code_ai.utils.inference import InferenceEnum

    # model_id UUID 到 InferenceEnum 的映射表
    # 這些 UUID 應該與資料庫中的模型配置對應
    MODEL_UUID_MAPPING = {
        # CMB (Cerebral Microbleed) - Swagger 示例 UUID
        "3fa85f64-5717-4562-b3fc-2c963f66afa6": InferenceEnum.CMB,
        # 可在此添加更多 UUID 映射
    }

    # 首先嘗試直接從 UUID 映射
    if model_id in MODEL_UUID_MAPPING:
        return MODEL_UUID_MAPPING[model_id]

    # 其次嘗試將 model_id 當作 InferenceEnum 名稱
    try:
        return InferenceEnum(model_id)
    except ValueError:
        pass

    # 最後嘗試不區分大小寫匹配
    model_id_upper = model_id.upper()
    for enum_member in InferenceEnum:
        if enum_member.value.upper() == model_id_upper:
            return enum_member

    raise ValueError(
        f"Unknown model_id: {model_id}. Valid models: {[e.value for e in InferenceEnum]}"
    )


def _is_batch_inputs_model(model_id: str) -> bool:
    """
    檢查模型是否需要批量輸入（多個 NIFTI 檔案作為一次推論的輸入）。

    例如 CMB 模型需要 SWAN + T1BRAVO 兩個輸入檔案。

    Args:
        model_id: 模型識別碼

    Returns:
        bool: True 如果模型需要批量輸入
    """
    from code_ai.pipeline import pipelines

    try:
        inference_enum = _resolve_model_id_to_inference_enum(model_id)
        if inference_enum in pipelines:
            return getattr(pipelines[inference_enum], "batch_inputs", False)
    except (ValueError, KeyError):
        pass

    return False


def _build_series_inference_cmd(
    nifti_paths: list,
    model_id: str,
    output_dir: str,
    dicom_dir: Optional[str] = None,
    study_id: Optional[str] = None,
    path_root: Optional[str] = None,
) -> InferenceCmd:
    """
    建立 series 推論命令。

    使用 code_ai/pipeline/__init__.py 中的 pipelines 配置來生成正確的推論命令。
    重用 check_study_mapping_inference 的邏輯確保 nifti_paths 順序與 config.yaml 一致。

    支援多輸入模型（如 CMB 需要 SWAN + T1BRAVO）。

    Args:
        nifti_paths: NIFTI 檔案路徑列表（單輸入模型傳 [path]，多輸入傳 [path1, path2, ...]）
        model_id: 模型識別碼（UUID 或模型名稱如 'CMB'）
        output_dir: 輸出目錄
        dicom_dir: DICOM 目錄（用於 DICOM-SEG 生成）
        study_id: Study ID（可選，從 nifti_path 推斷）
        path_root: PATH_ROOT 配置（可選，用於雙部署架構）

    Returns:
        InferenceCmd: 包含 InferenceCmdItem 的命令對象（與 Study Level 一致）

    Raises:
        ValueError: 如果 model_id 無法映射到已知模型
        KeyError: 如果模型未在 pipelines 中配置
    """
    import pathlib

    from code_ai.pipeline import pipelines
    from code_ai.utils.inference import Task, check_study_mapping_inference

    # Normalize input to list
    if isinstance(nifti_paths, str):
        nifti_paths = [nifti_paths]

    if not nifti_paths:
        raise ValueError("nifti_paths cannot be empty")

    # Step 1: 解析 model_id 到 InferenceEnum
    inference_enum = _resolve_model_id_to_inference_enum(model_id)

    # Step 2: 檢查 pipeline 是否已配置
    if inference_enum not in pipelines:
        available = [e.value for e in pipelines.keys()]
        raise KeyError(
            f"Model '{inference_enum.value}' not configured in pipelines. "
            f"Available: {available}"
        )

    pipeline_config = pipelines[inference_enum]

    # Step 3: 推斷 study_path 和 study_id
    first_path = nifti_paths[0]
    study_path = pathlib.Path(first_path).parent

    resolved_study_id: str
    if study_id is None:
        # 從路徑推斷，例如：/path/to/10516407_20231215_MR_21210200091/SWAN.nii.gz
        path_parts = first_path.split(os.sep)
        inferred_id: Optional[str] = None
        for part in reversed(path_parts):
            if "_" in part and not part.endswith(".nii.gz"):
                inferred_id = part
                break
        resolved_study_id = inferred_id if inferred_id else "unknown_study"
    else:
        resolved_study_id = study_id

    # Step 4: 重用 check_study_mapping_inference 獲取正確順序的 nifti_paths
    # 這確保 CMB 等多輸入模型的參數順序與 config.yaml 定義一致
    sorted_nifti_paths = nifti_paths  # 預設使用原順序
    if pipeline_config.batch_inputs and len(nifti_paths) > 1:
        try:
            mapping_result = check_study_mapping_inference(study_path)
            if mapping_result:
                study_mapping = mapping_result.get(study_path.name, {})
                model_paths = study_mapping.get(inference_enum.value, [])
                if model_paths and len(model_paths) == len(nifti_paths):
                    # 使用 check_study_mapping_inference 返回的順序
                    sorted_nifti_paths = model_paths
                    logger.info(
                        f"Using sorted nifti_paths from check_study_mapping_inference: "
                        f"{[os.path.basename(p) for p in sorted_nifti_paths]}"
                    )
        except Exception as e:
            logger.warning(
                f"Failed to get sorted paths from check_study_mapping_inference: {e}, "
                f"using original order"
            )

    # Step 5: 創建 Task 對象
    # Task 需要 input_path_list 和 output_path
    # 對於多輸入模型（batch_inputs=True），傳入排序後的路徑
    #
    # 重要：Output_folder 應該是 study_path.parent（與用戶範例一致）
    # 範例：--Inputs .../study_id/SWAN.nii.gz --Output_folder .../
    # 而不是 path_json 等其他路徑
    correct_output_path = str(study_path.parent)
    task = Task(
        intput_path_list=sorted_nifti_paths,  # Note: alias is "intput_path_list"
        output_path=correct_output_path,
        output_path_list=[],  # 輸出檔案列表將由 pipeline 自動生成
    )

    # Step 6: 使用 PipelineConfig.generate_cmd 生成命令
    cmd_str = pipeline_config.generate_cmd(
        study_id=resolved_study_id,
        task=task,
        input_dicom_dir=dicom_dir,
        path_root=path_root,
    )

    # Step 7: 創建 InferenceCmdItem（與 Study Level 的 build_inference_cmd 一致）
    inference_item = InferenceCmdItem(
        study_id=resolved_study_id,
        name=inference_enum,
        cmd_str=cmd_str,
        input_list=sorted_nifti_paths,
        output_list=task.output_path_list,
        input_dicom_dir=dicom_dir or "",
    )

    # Step 8: 返回 InferenceCmd（與 Study Level 格式一致）
    return InferenceCmd(cmd_items=[inference_item])


def _task_series_pipeline_inference(func_params: Dict[str, Any]):
    """
    Series Level 推論 - 處理指定的 series。

    與 Study Level 的區別：
    - 處理特定的 series，而非整個 study
    - 使用 model_id 指定模型，而非自動選擇
    - 使用 SERIES_INFERENCE_* 狀態碼
    - 使用 SERIES_INFERENCE_TOOL 作為 tool_id

    支持兩種模式：
    1. 直接模式 (needs_conversion=False，預設):
       - 必要參數: series_uids, nifti_series_paths, model_id
       - 假設 NIFTI 檔案已存在

    2. 轉換模式 (needs_conversion=True):
       - 必要參數: series_uids, raw_dicom_series_paths, model_id
       - 可選參數: path_rename_dicom, path_rename_nifti (或從環境變數讀取)
       - 自動執行: raw_dicom → rename_dicom → nifti → 推論

    Args:
        func_params: 任務參數字典，包含：
            - series_uids: List[str] - Series UID 列表
            - model_id: str - 模型識別碼
            - needs_conversion: bool - 是否需要 DICOM 轉換 (預設 False)

            直接模式:
            - nifti_series_paths: List[str] - NIFTI 檔案路徑列表

            轉換模式:
            - raw_dicom_series_paths: List[str] - 原始 DICOM 目錄路徑列表
            - path_rename_dicom: str - rename_dicom 輸出目錄
            - path_rename_nifti: str - NIFTI 輸出目錄

    Returns:
        str: 推論結果 JSON 字串
    """
    from code_ai.task.task_dicom2nii import call_post_httpx

    # Step 1: 驗證參數
    _validate_series_params(func_params)

    # Step 2: 提取配置
    upload_data_api_url = func_params.get("upload_data_api_url")
    if upload_data_api_url is None:
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        if upload_data_api_url is None:
            raise ValueError(
                "upload_data_api_url must be provided in task parameters or "
                "UPLOAD_DATA_API_URL environment variable must be set"
            )

    path_process = _extract_path_from_params(
        func_params, "path_process", "PATH_PROCESS"
    )
    path_json = _extract_path_from_params(func_params, "path_json", "PATH_JSON")
    path_log = _extract_path_from_params(func_params, "path_log", "PATH_LOG")
    path_root = _extract_path_from_params(func_params, "path_root", "PATH_ROOT")

    os.makedirs(path_json, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)

    # Step 2.5: 條件式轉換 (raw_dicom → rename_dicom → nifti)
    needs_conversion = func_params.get("needs_conversion", False)
    series_uids = func_params["series_uids"]

    if needs_conversion:
        # 轉換模式: 執行 DICOM 轉換
        logger.info(f"Conversion mode enabled for {len(series_uids)} series")

        # 提取轉換所需的路徑參數
        raw_dicom_paths = func_params["raw_dicom_series_paths"]
        path_rename_dicom = _extract_path_from_params(
            func_params, "path_rename_dicom", "PATH_RENAME_DICOM"
        )
        path_rename_nifti = _extract_path_from_params(
            func_params, "path_rename_nifti", "PATH_RENAME_NIFTI"
        )
        study_id = func_params.get("study_id", "unknown_study")

        # 執行批量轉換
        dicom_series_paths, nifti_paths = _batch_convert_series_to_nifti(
            raw_dicom_paths=raw_dicom_paths,
            series_uids=series_uids,
            output_dicom_base=path_rename_dicom,
            output_nifti_base=path_rename_nifti,
            study_id=study_id,
            upload_data_api_url=upload_data_api_url,
        )

        # 檢查轉換結果
        failed_conversions = [
            (uid, path) for uid, path in zip(series_uids, nifti_paths) if path is None
        ]
        if failed_conversions:
            logger.warning(
                f"Some series failed to convert: "
                f"{[uid for uid, _ in failed_conversions]}"
            )
    else:
        # 直接模式: 使用已存在的 NIFTI 路徑
        nifti_paths = func_params["nifti_series_paths"]
        dicom_series_paths = func_params.get(
            "dicom_series_paths", [None] * len(series_uids)
        )

    # Step 3: 提取其他 series 參數
    model_id = func_params["model_id"]
    inference_id = func_params.get("inference_id", str(uuid4()))
    study_uid = func_params.get("study_uid")
    study_id = func_params.get("study_id")
    # 注意: dicom_series_paths 和 nifti_paths 已在 Step 2.5 中設置

    api_url = f"{upload_data_api_url}{SYNC_PROT_OPE_NO}"

    # Step 4: 發送 SERIES_INFERENCE_RUNNING 事件
    if study_uid and study_id:
        dcop_event = DCOPEventRequest(
            study_uid=study_uid,
            series_uid=series_uids[0] if len(series_uids) == 1 else None,
            study_id=study_id,
            ope_no=DCOPStatus.SERIES_INFERENCE_RUNNING.value,
            tool_id="SERIES_INFERENCE_TOOL",
            params_data={
                "inference_id": inference_id,
                "series_count": len(series_uids),
                "series_uids": series_uids,
                "model_id": model_id,
                "func_params": func_params,
                "task": fct.function_result_status.get_status_dict(),
            },
        )
        try:
            call_post_httpx.push(
                {
                    "url": api_url,
                    "data": dcop_event.model_dump_json(),
                }
            )
            logger.info(
                f"Posted SERIES_INFERENCE_RUNNING for inference_id={inference_id}"
            )
        except Exception as e:
            logger.error(f"Failed to post RUNNING event: {e}")

    # Step 5: 執行推論
    result_list = []
    all_success = True
    error_message = None
    all_inference_cmd_items = []  # 收集所有 InferenceCmdItem（與 Study Level 一致）

    # 檢查是否為批量輸入模型（如 CMB 需要 SWAN + T1BRAVO）
    is_batch_model = _is_batch_inputs_model(model_id)

    try:
        if is_batch_model:
            # ====== 批量輸入模式 ======
            # 所有 series 的 NIFTI 檔案作為單次推論的多個輸入
            logger.info(
                f"Batch inputs model detected: {model_id}, "
                f"processing {len(series_uids)} series as single inference"
            )

            # 檢查所有 NIFTI 檔案存在
            valid_nifti_paths = []
            missing_series = []
            for series_uid, nifti_path in zip(series_uids, nifti_paths):
                if nifti_path is None or not os.path.exists(nifti_path):
                    missing_series.append(series_uid)
                    logger.error(f"NIFTI file not found: {nifti_path}")
                else:
                    valid_nifti_paths.append(nifti_path)

            if missing_series:
                error_msg = f"Missing NIFTI files for series: {missing_series}"
                logger.error(error_msg)
                for series_uid in missing_series:
                    result_list.append(
                        {
                            "series_uid": series_uid,
                            "status": "failed",
                            "error": error_msg,
                        }
                    )
                all_success = False

            if valid_nifti_paths:
                # 建立輸出目錄（使用所有 series UID 組合）
                combined_uid = "_".join(series_uids[:2])  # 最多用前兩個
                output_dir = os.path.join(path_json, f"{combined_uid}_{model_id}")
                os.makedirs(output_dir, exist_ok=True)

                # 取得第一個 DICOM 路徑（用於 DICOM-SEG）
                dicom_dir = dicom_series_paths[0] if dicom_series_paths else None

                # 建立推論命令（傳入所有 NIFTI 路徑，返回 InferenceCmd）
                inference_cmd = _build_series_inference_cmd(
                    nifti_paths=valid_nifti_paths,
                    model_id=model_id,
                    output_dir=output_dir,
                    dicom_dir=dicom_dir,
                    study_id=study_id,
                    path_root=path_root,
                )
                # 收集 InferenceCmdItem（與 Study Level 一致）
                all_inference_cmd_items.extend(inference_cmd.cmd_items)
                # 提取 cmd_str 用於 subprocess 執行
                cmd_str = inference_cmd.cmd_items[0].cmd_str
                logger.info(f"Executing batch inference: {cmd_str}")

                # 執行推論
                try:
                    process = subprocess.Popen(
                        args=cmd_str,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=path_process,
                    )
                    stdout, stderr = process.communicate(timeout=600)

                    if process.returncode != 0:
                        error_msg = f"Inference failed with code {process.returncode}: {stderr.decode()}"
                        logger.error(error_msg)
                        for series_uid in series_uids:
                            if series_uid not in missing_series:
                                result_list.append(
                                    {
                                        "series_uid": series_uid,
                                        "status": "failed",
                                        "error": error_msg,
                                        "stderr": stderr.decode(),
                                    }
                                )
                        all_success = False
                    else:
                        # 解析結果
                        prediction_file = os.path.join(output_dir, "prediction.json")
                        if os.path.exists(prediction_file):
                            with open(prediction_file, "r") as f:
                                prediction = json.load(f)
                        else:
                            prediction = {"raw_output": stdout.decode()}

                        # 批量模式：所有有效 series 共享同一結果
                        for series_uid in series_uids:
                            if series_uid not in missing_series:
                                result_list.append(
                                    {
                                        "series_uid": series_uid,
                                        "status": "success",
                                        "prediction": prediction,
                                        "output_dir": output_dir,
                                        "batch_inference": True,
                                    }
                                )
                        logger.info(
                            f"Batch inference completed successfully for {len(valid_nifti_paths)} series"
                        )

                except subprocess.TimeoutExpired:
                    error_msg = "Batch inference timeout (>10 min)"
                    logger.error(error_msg)
                    process.kill()
                    for series_uid in series_uids:
                        if series_uid not in missing_series:
                            result_list.append(
                                {
                                    "series_uid": series_uid,
                                    "status": "failed",
                                    "error": error_msg,
                                }
                            )
                    all_success = False

                except Exception as e:
                    error_msg = f"Batch inference error: {str(e)}"
                    logger.error(error_msg)
                    for series_uid in series_uids:
                        if series_uid not in missing_series:
                            result_list.append(
                                {
                                    "series_uid": series_uid,
                                    "status": "failed",
                                    "error": error_msg,
                                }
                            )
                    all_success = False

        else:
            # ====== 單一輸入模式（原有邏輯） ======
            # 每個 series 獨立執行推論
            for i, (series_uid, nifti_path) in enumerate(zip(series_uids, nifti_paths)):
                logger.info(
                    f"Processing series {i + 1}/{len(series_uids)}: {series_uid}"
                )

                # 檢查 NIFTI 檔案存在
                if nifti_path is None or not os.path.exists(nifti_path):
                    error_msg = (
                        f"NIFTI file not found or conversion failed: {nifti_path}"
                    )
                    logger.error(error_msg)
                    result_list.append(
                        {
                            "series_uid": series_uid,
                            "status": "failed",
                            "error": error_msg,
                        }
                    )
                    all_success = False
                    continue

                # 建立輸出目錄
                output_dir = os.path.join(path_json, f"{series_uid}_{model_id}")
                os.makedirs(output_dir, exist_ok=True)

                # 取得 DICOM 路徑（如果有）
                dicom_dir = (
                    dicom_series_paths[i] if i < len(dicom_series_paths) else None
                )

                # 建立推論命令（傳入單一路徑的列表，返回 InferenceCmd）
                inference_cmd = _build_series_inference_cmd(
                    nifti_paths=[nifti_path],
                    model_id=model_id,
                    output_dir=output_dir,
                    dicom_dir=dicom_dir,
                    study_id=study_id,
                    path_root=path_root,
                )
                # 收集 InferenceCmdItem（與 Study Level 一致）
                all_inference_cmd_items.extend(inference_cmd.cmd_items)
                # 提取 cmd_str 用於 subprocess 執行
                cmd_str = inference_cmd.cmd_items[0].cmd_str
                logger.info(f"Executing: {cmd_str}")

                # 執行推論
                try:
                    process = subprocess.Popen(
                        args=cmd_str,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=path_process,
                    )
                    stdout, stderr = process.communicate(timeout=600)  # 10 分鐘超時

                    if process.returncode != 0:
                        error_msg = f"Inference failed with code {process.returncode}: {stderr.decode()}"
                        logger.error(error_msg)
                        result_list.append(
                            {
                                "series_uid": series_uid,
                                "status": "failed",
                                "error": error_msg,
                                "stderr": stderr.decode(),
                            }
                        )
                        all_success = False
                    else:
                        # 解析結果
                        prediction_file = os.path.join(output_dir, "prediction.json")
                        if os.path.exists(prediction_file):
                            with open(prediction_file, "r") as f:
                                prediction = json.load(f)
                        else:
                            prediction = {"raw_output": stdout.decode()}

                        result_list.append(
                            {
                                "series_uid": series_uid,
                                "status": "success",
                                "prediction": prediction,
                                "output_dir": output_dir,
                            }
                        )
                        logger.info(
                            f"Series {series_uid} inference completed successfully"
                        )

                except subprocess.TimeoutExpired:
                    error_msg = f"Inference timeout (>10 min) for series {series_uid}"
                    logger.error(error_msg)
                    process.kill()
                    result_list.append(
                        {
                            "series_uid": series_uid,
                            "status": "failed",
                            "error": error_msg,
                        }
                    )
                    all_success = False

                except Exception as e:
                    error_msg = f"Inference error for series {series_uid}: {str(e)}"
                    logger.error(error_msg)
                    result_list.append(
                        {
                            "series_uid": series_uid,
                            "status": "failed",
                            "error": error_msg,
                        }
                    )
                    all_success = False

    except Exception as e:
        error_message = f"Fatal error during series inference: {str(e)}"
        logger.error(error_message)
        all_success = False

    # Step 6: 發送 SERIES_INFERENCE_COMPLETE/FAILED 事件
    result_json = Serialization.to_json_str(result_list)

    if study_uid and study_id:
        completion_status = (
            DCOPStatus.SERIES_INFERENCE_COMPLETE.value
            if all_success
            else DCOPStatus.SERIES_INFERENCE_FAILED.value
        )

        dcop_event = DCOPEventRequest(
            study_uid=study_uid,
            series_uid=series_uids[0] if len(series_uids) == 1 else None,
            study_id=study_id,
            ope_no=completion_status,
            tool_id="SERIES_INFERENCE_TOOL",
            params_data={
                "inference_item_cmd": all_inference_cmd_items,  # 與 Study Level 一致
                "func_params": func_params,  # 與 Study Level 一致
                "inference_id": inference_id,
                "series_count": len(series_uids),
                "series_uids": series_uids,
                "model_id": model_id,
                "task": fct.function_result_status.get_status_dict(),
            },
            result_data={
                "result": result_json,
                "all_success": all_success,
                "error": error_message,
            },
        )

        try:
            call_post_httpx.push(
                {
                    "url": api_url,
                    "data": dcop_event.model_dump_json(),
                }
            )
            logger.info(f"Posted {completion_status} for inference_id={inference_id}")
        except Exception as e:
            logger.error(f"Failed to post COMPLETE event: {e}")

    return result_json


# =============================================================================
# Subprocess Task（保持不變）
# =============================================================================


@Booster(
    BoosterParamsMyRABBITMQ(
        queue_name="task_subprocess_queue",
        concurrent_num=3,
        qps=1,
    )
)
def task_subprocess_inference(func_params: Dict[str, Any]):
    """Subprocess 推論任務（保持不變）"""
    path_process = _extract_path_from_params(
        func_params, "path_process", "PATH_PROCESS"
    )
    path_cmd_tools = os.path.join(path_process, "Deep_cmd_tools")
    os.makedirs(path_cmd_tools, exist_ok=True)

    cmd_str = func_params["cmd_str"]
    process = subprocess.Popen(
        args=cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    logger.info(stdout.decode())
    logger.warn(stderr.decode())
    return stdout.decode()
