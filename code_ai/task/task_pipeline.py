import json
import os
import pathlib
import subprocess
from typing import Dict
from funboost import Booster
from funboost.core.serialization import Serialization
import nb_log
# 設置日誌記錄器
logger = nb_log.LogManager('task_pipeline_inference_queue').get_logger_and_add_handlers(
    log_filename='task_pipeline_inference_queue.log'
)

from code_ai.task.params import BoosterParamsMyAI,BoosterParamsMyRABBITMQ
from code_ai.utils.inference import build_inference_cmd
from code_ai.utils.database import save_result_status_to_sqlalchemy


@Booster(BoosterParamsMyAI(queue_name ='task_pipeline_inference_queue',
                           user_custom_record_process_info_func = save_result_status_to_sqlalchemy,
                           qps=1,
                           ))
def task_pipeline_inference(func_params  : Dict[str,any]):
    path_process = os.getenv("PATH_PROCESS")
    path_cmd_tools = os.path.join(path_process, 'Deep_cmd_tools')
    path_json = os.getenv("PATH_JSON")
    path_log = os.getenv("PATH_LOG")
    # 建置資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_cmd_tools, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，

    nifti_study_path = func_params['nifti_study_path']
    dicom_study_path = func_params['dicom_study_path']
    inference_item_cmd = build_inference_cmd(pathlib.Path(nifti_study_path),
                                             pathlib.Path(dicom_study_path))
    cmd_output_path = os.path.join(path_cmd_tools, f'{inference_item_cmd.cmd_items[0].study_id}_cmd.json')
    with open(cmd_output_path, 'w') as f:
        f.write(json.dumps(inference_item_cmd.model_dump()['cmd_items']))

    result_list = []
    for inference_item in inference_item_cmd.cmd_items:
        process = subprocess.Popen(args=inference_item.cmd_str, shell=True,
                                   # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        result_list.append((inference_item.cmd_str,stdout.decode(), stderr.decode()))
        logger.info(stderr.decode())
        logger.warn(stderr.decode())
    result = Serialization.to_json_str(result_list)
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