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
from code_ai.utils.database import save_result_status_to_sqlalchemy


def _send_events(api_url: str, event_data: List[dict]) -> None:
    """
    Sends study transfer complete events to the API.

    Args:
        api_url: Base URL for the upload data API.
        event_data: List of serialized DCOPEventRequest objects.
    """
    with httpx.Client(timeout=180) as client:
        url = f"{api_url}"
        event_data_json = json.dumps(event_data)
        rep = client.post(url=url, timeout=180, data=event_data_json)

@Booster(BoosterParamsMyAI(queue_name ='task_pipeline_inference_queue',
                           user_custom_record_process_info_func = save_result_status_to_sqlalchemy,
                           qps=1,
                           ))
def task_pipeline_inference(func_params  : Dict[str,any]):
    upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
    path_process   = os.getenv("PATH_PROCESS")
    path_cmd_tools = os.path.join(path_process, 'Deep_cmd_tools')
    path_json      = os.getenv("PATH_JSON")
    path_log       = os.getenv("PATH_LOG")
    # 建置資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_cmd_tools, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，

    nifti_study_path = func_params['nifti_study_path']
    dicom_study_path = func_params['dicom_study_path']

    inference_item_cmd = build_inference_cmd(pathlib.Path(nifti_study_path),
                                             pathlib.Path(dicom_study_path))
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

        _send_events(api_url,[dcop_event.model_dump()])
    else:
        pass

    result_list = []
    for inference_item in inference_item_cmd.cmd_items:
        process = subprocess.Popen(args=inference_item.cmd_str, shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        result_list.append((inference_item.cmd_str,stdout.decode(), stderr.decode()))
        logger.info(stderr.decode())
        logger.warn(stderr.decode())
    result = Serialization.to_json_str(result_list)
    if study_uid and study_id:
        dcop_event = DCOPEventRequest(study_uid=study_uid, series_uid=None, study_id=study_id,
                                      ope_no=DCOPStatus.STUDY_INFERENCE_COMPLETE.value,
                                      tool_id='INFERENCE_TOOL',
                                      params_data={'inference_item_cmd': inference_item_cmd.cmd_items,
                                                   'func_params': func_params,
                                                   'task': fct.function_result_status.get_status_dict()
                                                   },
                                      result_data = {'result':result,
                                                })

        _send_events(api_url,[dcop_event.model_dump()])
    return result


@Booster(BoosterParamsMyRABBITMQ(queue_name      ='task_subprocess_queue',
                                 concurrent_num  = 3,
                                 qps=1,
                           ))
def task_subprocess_inference(func_params  : Dict[str,any]):
    path_process = os.getenv("PATH_PROCESS")
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