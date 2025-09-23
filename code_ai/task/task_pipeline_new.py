"""
醫學影像推理任務 - PgQueuer 版本
遷移自 task_pipeline.py 的 funboost 實作
"""
import json
import os
import pathlib
import subprocess
from typing import Dict, List

import httpx
import nb_log
from backend.app.sync.schemas import DCOPStatus, DCOPEventRequest
from backend.app.sync.urls import SYNC_PROT_OPE_NO
from code_ai.utils.inference import build_inference_cmd
from code_ai.queue.manager import get_queue_manager

# 設置日誌記錄器
logger = nb_log.LogManager('task_pipeline_inference_queue').get_logger_and_add_handlers(
    log_filename='task_pipeline_inference_queue.log'
)

# 取得全域佇列管理器
queue_manager = get_queue_manager()


@queue_manager.register_processor(
    queue_name='task_pipeline_inference_queue',
    qps=1,
    max_retry_times=3,
    concurrent_num=5
)
async def task_pipeline_inference(func_params: Dict[str, any]):
    """
    醫學影像推理任務
    
    原始 funboost 版本的直接遷移，保持所有業務邏輯不變
    只替換了佇列框架和序列化方法
    """
    # 從 task_dicom2nii_new 匯入 (避免循環匯入)
    from code_ai.task.task_dicom2nii_new import call_post_httpx
    
    upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
    path_process = os.getenv("PATH_PROCESS")
    path_cmd_tools = os.path.join(path_process, 'Deep_cmd_tools')
    path_json = os.getenv("PATH_JSON")
    path_log = os.getenv("PATH_LOG")
    
    # 建置資料夾
    os.makedirs(path_json, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)
    os.makedirs(path_cmd_tools, exist_ok=True)

    nifti_study_path = func_params['nifti_study_path']
    dicom_study_path = func_params['dicom_study_path']

    inference_item_cmd = build_inference_cmd(
        pathlib.Path(nifti_study_path),
        pathlib.Path(dicom_study_path)
    )
    
    if inference_item_cmd.cmd_items:
        cmd_output_path = os.path.join(
            path_cmd_tools, 
            f'{inference_item_cmd.cmd_items[0].study_id}_cmd.json'
        )
    else:
        temp_id = os.path.basename(dicom_study_path)
        cmd_output_path = os.path.join(path_cmd_tools, f'{temp_id}_cmd.json')
    
    with open(cmd_output_path, 'w') as f:
        f.write(json.dumps(inference_item_cmd.model_dump()['cmd_items']))

    study_uid = func_params.get('study_uid', None)
    study_id = func_params.get('study_id', None)
    api_url = f"{upload_data_api_url}{SYNC_PROT_OPE_NO}"

    # 發送開始推理事件
    if study_uid and study_id:
        dcop_event = DCOPEventRequest(
            study_uid=study_uid, 
            series_uid=None, 
            study_id=study_id,
            ope_no=DCOPStatus.STUDY_INFERENCE_RUNNING.value,
            tool_id='INFERENCE_TOOL',
            params_data={
                'inference_item_cmd': inference_item_cmd.cmd_items,
                'func_params': func_params,
                # 注意：移除了 funboost 的 fct.function_result_status
                'task_status': 'running'
            }
        )
        
        # 使用 pgqueuer 發送 HTTP 請求任務
        await call_post_httpx({
            'url': api_url,
            'data': dcop_event.model_dump_json(),
        })

    # 執行推理命令
    result_list = []
    for inference_item in inference_item_cmd.cmd_items:
        process = subprocess.Popen(
            args=inference_item.cmd_str, 
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        result_list.append((inference_item.cmd_str, stdout.decode(), stderr.decode()))
        
        logger.info(f"命令執行完成: {inference_item.cmd_str}")
        if stderr:
            logger.warning(f"命令警告: {stderr.decode()}")

    # 序列化結果 (替代 funboost 的 Serialization.to_json_str)
    result = json.dumps(result_list, ensure_ascii=False, default=str)
    
    # 發送完成事件
    if study_uid and study_id:
        dcop_event = DCOPEventRequest(
            study_uid=study_uid, 
            series_uid=None, 
            study_id=study_id,
            ope_no=DCOPStatus.STUDY_INFERENCE_COMPLETE.value,
            tool_id='INFERENCE_TOOL',
            params_data={
                'inference_item_cmd': inference_item_cmd.cmd_items,
                'func_params': func_params,
                'task_status': 'completed'
            },
            result_data={'result': result}
        )
        
        await call_post_httpx({
            'url': api_url,
            'data': dcop_event.model_dump_json(),
        })
    
    return result


@queue_manager.register_processor(
    queue_name='task_subprocess_queue',
    concurrent_num=3,
    qps=1
)
async def task_subprocess_inference(func_params: Dict[str, any]):
    """
    子進程推理任務
    
    原始 funboost 版本的直接遷移
    """
    path_process = os.getenv("PATH_PROCESS")
    path_cmd_tools = os.path.join(path_process, 'Deep_cmd_tools')
    os.makedirs(path_cmd_tools, exist_ok=True)
    
    cmd_str = func_params['cmd_str']
    process = subprocess.Popen(
        args=cmd_str, 
        shell=True,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    
    logger.info(f"子進程執行結果: {stdout.decode()}")
    if stderr:
        logger.warning(f"子進程警告: {stderr.decode()}")
    
    return stdout.decode()


# 便捷函數：加入任務到佇列
async def enqueue_pipeline_inference(func_params: Dict[str, any]) -> str:
    """將推理任務加入佇列"""
    return await queue_manager.enqueue_job('task_pipeline_inference_queue', func_params)


async def enqueue_subprocess_inference(func_params: Dict[str, any]) -> str:
    """將子進程任務加入佇列"""
    return await queue_manager.enqueue_job('task_subprocess_queue', func_params)
