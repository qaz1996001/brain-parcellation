import os.path
import shutil
import subprocess
from typing import List, Dict, Callable, Optional
import pathlib

import gc
from funboost import BrokerEnum, BoosterParams, Booster, ConcurrentModeEnum, BoosterParamsComplete

from code_ai import PYTHON3
from code_ai.task import resample_one, resampleSynthSEG2original_z_index, save_original_seg_by_argmin_z_index
from code_ai.utils_inference import get_synthseg_args_file
from code_ai.utils_inference import InferenceEnum
from code_ai.task.schema import intput_params


# 定義 Funboost 任務
@Booster('resample_task_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10,
         is_send_consumer_hearbeat_to_redis = True,
         is_push_to_dlx_queue_when_retry_max_times  = True,
         is_using_rpc_mode =True)
# def resample_task(file, resample_file):
def resample_task(func_params  : Dict[str,any]):
    task_params = intput_params.ResampleTaskParams.model_validate(func_params,strict=False)
    file = task_params.file
    resample_file = task_params.resample_file
    if not resample_file.parent.exists():
        resample_file.parent.mkdir(parents=True, exist_ok=True)
    if not resample_file.exists():
        resample_one(str(file), str(resample_file))
    return str(resample_file)

@Booster('synthseg_task_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_num = 2,
         is_send_consumer_hearbeat_to_redis=True,
         is_push_to_dlx_queue_when_retry_max_times=True,
         is_using_rpc_mode=True
         )
def synthseg_task(func_params  : Dict[str,any]):
    task_params     = intput_params.SynthsegTaskParams.model_validate(func_params)
    resample_file   = task_params.resample_file
    synthseg_file   = task_params.synthseg_file
    synthseg33_file = task_params.synthseg33_file

    try:
        if all([resample_file.exists(), synthseg_file.exists(), synthseg33_file.exists()]):
            pass
        else:
            cmd_str = ('export PYTHONPATH={} && '
                       '{} code_ai/pipeline/synthseg_task.py '
                       '--resample_file {} '
                       '--synthseg_file {} '
                       '--synthseg33_file {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                                      PYTHON3,
                                                      resample_file,
                                                      synthseg_file,
                                                      synthseg33_file, )
                       )
            process = subprocess.Popen(args=cmd_str, shell=True,
                                       # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            return synthseg_file, synthseg33_file
    except Exception as e:
        print(e)
        raise e  # Funboost 會自動處理重試

@Booster('process_synthseg_task_queue',
         broker_kind     = BrokerEnum.RABBITMQ_AMQPSTORM, qps=10,
         concurrent_mode = ConcurrentModeEnum.SOLO,
         concurrent_num  = 1 ,
         is_send_consumer_hearbeat_to_redis = True,
         is_using_rpc_mode=True
         )
def process_synthseg_task(func_params  : Dict[str,any]):
    # BoosterParams()
    task_params     = intput_params.ProcessSynthsegTaskParams.model_validate(func_params)
    synthseg_file   = task_params.synthseg_file
    synthseg33_file = task_params.synthseg33_file
    david_file      = task_params.david_file
    wm_file         = task_params.wm_file
    depth_number    = task_params.depth_number


    if os.path.exists(wm_file) and os.path.exists(david_file):
        return synthseg_file, david_file
    else:
        cmd_str = ('export PYTHONPATH={} && '
                   '{} code_ai/pipeline/process_synthseg.py '
                   '--synthseg_file {} '
                   '--synthseg33_file {} '
                   '--david_file {} '
                   '--wm_file {} '
                   '--depth_number {}'.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                              PYTHON3,
                                              synthseg_file,
                                              synthseg33_file,
                                              david_file,
                                              wm_file,
                                              depth_number)
                   )
        process = subprocess.Popen(args=cmd_str, shell=True,  # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        # print(stderr)
        # return stdout, stderr
        return synthseg_file, david_file


@Booster('resample_to_original_task_queue',
         broker_kind     = BrokerEnum.RABBITMQ_AMQPSTORM,
         qps=10,
         is_send_consumer_hearbeat_to_redis = True,
         is_using_rpc_mode=True)
def resample_to_original_task(func_params  : Dict[str,any]):
    task_params    = intput_params.PostProcessSynthsegTaskParams.model_validate(func_params)

    original_seg_file, argmin = resampleSynthSEG2original_z_index(raw_file            = task_params.file,
                                                          resample_image_file = task_params.resample_file,
                                                          resample_seg_file   = task_params.synthseg_file)
    original_synthseg33_seg_file = save_original_seg_by_argmin_z_index(raw_file          = task_params.file,
                                                                       resample_seg_file = task_params.synthseg33_file,
                                                                       argmin=argmin)
    original_david_seg_file = save_original_seg_by_argmin_z_index(raw_file          = task_params.file,
                                                                  resample_seg_file = task_params.david_file,
                                                                  argmin=argmin)
    original_save_seg_file = save_original_seg_by_argmin_z_index(raw_file=task_params.file,
                                                                 resample_seg_file=task_params.save_file_path,
                                                                 argmin=argmin)
    original_file     = task_params.file
    resample_seg_file = task_params.save_file_path

    outpput_raw_file = resample_seg_file.parent.joinpath(original_file.name)
    if str(original_file) == str(outpput_raw_file):
        pass
    else:
        with open(original_file, mode='rb') as raw_file_f:
            with open(outpput_raw_file, mode='wb') as outpput_raw_file_f:
                shutil.copyfileobj(raw_file_f, outpput_raw_file_f)
    return original_file,original_seg_file,original_synthseg33_seg_file,original_david_seg_file,original_save_seg_file


@Booster('save_file_tasks_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         is_send_consumer_hearbeat_to_redis = True,
         is_using_rpc_mode=True, qps=10)
def save_file_tasks(func_params  : Dict[str,any]):
    task_params = intput_params.SaveFileTaskParams.model_validate(func_params)
    synthseg_file = task_params.synthseg_file
    david_file = task_params.david_file
    wm_file = task_params.wm_file
    depth_number = task_params.depth_number
    save_mode = task_params.save_mode
    save_file_path = task_params.save_file_path


    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/save_file.py '
               '--synthseg_file {} '
               '--david_file {} '
               '--wm_file {} '
               '--depth_number {} '
               '--save_mode {} '
               '--save_file_path {}'.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                          PYTHON3,
                                          synthseg_file,
                                          david_file,
                                          wm_file,
                                          depth_number,
                                          save_mode,
                                          save_file_path)
               )
    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(stderr)
    # return stdout, stderr
    return synthseg_file, david_file


@Booster('post_process_synthseg_task_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         is_send_consumer_hearbeat_to_redis = True,
         is_using_rpc_mode=True, qps=10)
def post_process_synthseg_task(func_params  : Dict[str,any]):
    task_params = intput_params.SaveFileTaskParams.model_validate(func_params)
    synthseg_file = task_params.synthseg_file
    david_file = task_params.david_file
    wm_file = task_params.wm_file
    depth_number = task_params.depth_number
    save_mode = task_params.save_mode
    save_file_path = task_params.save_file_path

    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/save_file.py '
               '--synthseg_file {} '
               '--david_file {} '
               '--wm_file {} '
               '--depth_number {} '
               '--save_mode {} '
               '--save_file_path {}'.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                            PYTHON3,
                                            synthseg_file,
                                            david_file,
                                            wm_file,
                                            depth_number,
                                            save_mode,
                                            save_file_path)
               )
    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return synthseg_file, david_file