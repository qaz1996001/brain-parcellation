import json
import os
import pathlib
import subprocess
from typing import Dict, List

import httpx
from funboost import Booster,fct
from funboost.core.serialization import Serialization
import nb_log
# 設置日誌記錄器
logger = nb_log.LogManager('task_pipeline_inference_queue').get_logger_and_add_handlers(
    log_filename='task_pipeline_inference_queue.log'
)

from backend.app.sync.schemas import DCOPStatus,DCOPEventRequest
from backend.app.sync.urls import SYNC_PROT_OPE_NO
from code_ai.task.params import BoosterParamsMyAI,BoosterParamsMyRABBITMQ
from code_ai.utils.inference import build_inference_cmd


def _extract_path_from_params(func_params: Dict[str, any], param_name: str, env_var_name: str) -> str:
    """Extract path from task parameters with fallback to environment variable.

    This utility function implements the parameter injection pattern for task path configuration.
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

    Examples:
        >>> path = _extract_path_from_params(func_params, 'path_process', 'PATH_PROCESS')
        >>> # First tries func_params['path_process'], then os.getenv('PATH_PROCESS')
    """
    # Try to get path from task parameters (injected by dispatcher)
    path_value = func_params.get(param_name)

    if path_value is None:
        # Fallback to environment variable for backward compatibility
        logger.warning(
            f"Path parameter '{param_name}' not found in task parameters, "
            f"falling back to environment variable '{env_var_name}'. "
            f"Consider updating dispatcher to inject path parameters."
        )
        path_value = os.getenv(env_var_name)

        if path_value is None:
            raise ValueError(
                f"{param_name} must be provided in task parameters or "
                f"{env_var_name} environment variable must be set"
            )

    return path_value


@Booster(BoosterParamsMyAI(queue_name ='task_pipeline_inference_queue',
                           qps=1,
                           ))
def task_pipeline_inference(func_params  : Dict[str,any]):
    #
    from code_ai.task.task_dicom2nii import call_post_httpx
    # Use upload_data_api_url from task parameters (injected by dispatcher)
    upload_data_api_url = func_params.get('upload_data_api_url')
    if upload_data_api_url is None:
        # Fallback to environment variable for backward compatibility
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        if upload_data_api_url is None:
            raise ValueError("upload_data_api_url must be provided in task parameters or UPLOAD_DATA_API_URL environment variable must be set")

    # Extract path configuration from task parameters with environment fallback
    # This enables dual deployment: Production and Testing backends can pass different paths
    # to the same GPU worker, making this function pure (not dependent on worker's .env)
    path_process   = _extract_path_from_params(func_params, 'path_process', 'PATH_PROCESS')
    path_cmd_tools = os.path.join(path_process, 'Deep_cmd_tools')
    path_json      = _extract_path_from_params(func_params, 'path_json', 'PATH_JSON')
    path_log       = _extract_path_from_params(func_params, 'path_log', 'PATH_LOG')
    path_root      = _extract_path_from_params(func_params, 'path_root', 'PATH_ROOT')
    # 建置資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_cmd_tools, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，

    nifti_study_path = func_params['nifti_study_path']
    dicom_study_path = func_params['dicom_study_path']

    inference_item_cmd = build_inference_cmd(pathlib.Path(nifti_study_path),
                                             pathlib.Path(dicom_study_path),
                                             path_root=path_root)
    if inference_item_cmd.cmd_items:
        cmd_output_path = os.path.join(path_cmd_tools, f'{inference_item_cmd.cmd_items[0].study_id}_cmd.json')
    else:
        temp_id = os.path.basename(dicom_study_path)
        cmd_output_path = os.path.join(path_cmd_tools, f'{temp_id}_cmd.json')
    with open(cmd_output_path, 'w') as f:
        f.write(json.dumps(inference_item_cmd.model_dump()['cmd_items']))

    study_uid = func_params.get('study_uid', None)
    study_id = func_params.get('study_id', None)
    api_url = f"{upload_data_api_url}{SYNC_PROT_OPE_NO}"

    if study_uid and study_id:
        dcop_event = DCOPEventRequest(study_uid=study_uid, series_uid=None, study_id=study_id,
                                      ope_no=DCOPStatus.STUDY_INFERENCE_RUNNING.value,
                                      tool_id='INFERENCE_TOOL',
                                      params_data={'inference_item_cmd': inference_item_cmd.cmd_items,
                                                   'func_params': func_params,
                                                   'task': fct.function_result_status.get_status_dict()
                                                   })
        call_post_httpx.push({'url':api_url,
                              'data': dcop_event.model_dump_json(),
                              })
    else:
        pass

    result_list = []
    for inference_item in inference_item_cmd.cmd_items:
        process = subprocess.Popen(args=inference_item.cmd_str, shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        result_list.append((inference_item.cmd_str,stdout.decode(), stderr.decode()))
        # logger.info("{}".format(stdout.decode()))
        # logger.warn("{}".format(stderr.decode()))
    result = Serialization.to_json_str(result_list)
    if study_uid and study_id:
        dcop_event = DCOPEventRequest(study_uid=study_uid, series_uid=None, study_id=study_id,
                                      ope_no=DCOPStatus.STUDY_INFERENCE_COMPLETE.value,
                                      tool_id='INFERENCE_TOOL',
                                      params_data={'inference_item_cmd': inference_item_cmd.cmd_items,
                                                   'func_params': func_params,
                                                   'task': fct.function_result_status.get_status_dict()
                                                   },
                                      result_data = {'result':result,})
        call_post_httpx.push({'url': api_url,
                              'data': dcop_event.model_dump_json(),
                              })
    return result


@Booster(BoosterParamsMyRABBITMQ(queue_name      ='task_subprocess_queue',
                                 concurrent_num  = 3,
                                 qps=1,
                           ))
def task_subprocess_inference(func_params  : Dict[str,any]):
    # Extract path_process from task parameters with environment fallback
    # This enables dual deployment where backends can specify execution paths
    path_process = _extract_path_from_params(func_params, 'path_process', 'PATH_PROCESS')
    path_cmd_tools = os.path.join(path_process, 'Deep_cmd_tools')
    os.makedirs(path_cmd_tools, exist_ok=True)  # 如果資料夾不存在就建立，
    cmd_str = func_params['cmd_str']
    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logger.info(stdout.decode())
    logger.warn(stderr.decode())
    return stdout.decode()