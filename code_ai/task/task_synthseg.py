import os.path
import shutil
import subprocess
from typing import Dict
import pathlib

from funboost import BrokerEnum, Booster, ConcurrentModeEnum

from code_ai import PYTHON3
from code_ai.task import resample_one, resampleSynthSEG2original_z_index, save_original_seg_by_argmin_z_index

from code_ai.task.schema import intput_params
from code_ai.utils_inference import replace_suffix
from code_ai.utils_synthseg import TemplateProcessor


# 定義 Funboost 任務
@Booster('resample_task_queue',
         broker_kind     = BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode = ConcurrentModeEnum.SOLO,
         concurrent_num  = 8,
         qps=2,
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
         broker_kind     = BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode = ConcurrentModeEnum.SOLO,
         concurrent_num  = 4,
         qps=1,
         # is_using_distributed_frequency_control=True,
         is_send_consumer_hearbeat_to_redis=True,
         is_push_to_dlx_queue_when_retry_max_times=True,
         is_using_rpc_mode=True)
def synthseg_task(func_params  : Dict[str,any]):
    task_params     = intput_params.SynthsegTaskParams.model_validate(func_params)
    resample_file   = task_params.resample_file
    synthseg_file   = task_params.synthseg_file
    synthseg33_file = task_params.synthseg33_file

    try:
        if all([resample_file.exists(), synthseg_file.exists(), synthseg33_file.exists()]):
            return synthseg_file, synthseg33_file
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
         broker_kind     = BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode = ConcurrentModeEnum.SOLO,
         concurrent_num  = 4,
         qps=1,
         # is_using_distributed_frequency_control=True,
         is_send_consumer_hearbeat_to_redis=True,
         is_push_to_dlx_queue_when_retry_max_times=True,
         is_using_rpc_mode=True)
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
         concurrent_mode = ConcurrentModeEnum.THREADING,
         concurrent_num  = 4,
         qps=1,
         is_send_consumer_hearbeat_to_redis = True,
         is_using_rpc_mode=True)
def resample_to_original_task(func_params  : Dict[str,any]):
    task_params    = intput_params.SaveFileTaskParams.model_validate(func_params)
    original_file     = task_params.file
    resample_seg_file = task_params.save_file_path
    base_path = task_params.synthseg_file.parent

    original_seg_file = base_path.joinpath(
        f"synthseg_{task_params.synthseg_file.name.replace('resample', 'original')}")
    original_synthseg33_seg_file = base_path.joinpath(
        f"synthseg_{task_params.synthseg33_file.name.replace('resample', 'original')}")
    original_david_seg_file = base_path.joinpath(
        f"synthseg_{task_params.david_file.name.replace('resample', 'original')}")
    original_save_seg_file = base_path.joinpath(
        f"synthseg_{task_params.save_file_path.name.replace('resample', 'original')}")
    # task_params.wm_file
    if all((original_seg_file.exists(), original_synthseg33_seg_file.exists(),
            original_david_seg_file.exists(),original_save_seg_file.exists())):
        return original_file,original_seg_file,original_synthseg33_seg_file,original_david_seg_file,original_save_seg_file
    else:
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
         concurrent_mode=ConcurrentModeEnum.THREADING,
         concurrent_num=4,
         qps=1,
         # is_using_distributed_frequency_control=True,
         is_send_consumer_hearbeat_to_redis = True,
         is_using_rpc_mode=True)
def save_file_tasks(func_params  : Dict[str,any]):
    task_params = intput_params.SaveFileTaskParams.model_validate(func_params)
    synthseg_file = task_params.synthseg_file
    david_file = task_params.david_file
    wm_file = task_params.wm_file
    depth_number = task_params.depth_number
    save_mode = task_params.save_mode
    save_file_path = task_params.save_file_path

    if save_file_path.exists() and save_file_path.stat().st_size > 20240:
        return save_file_path
    else:
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
        return save_file_path


@Booster('post_process_synthseg_task_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode=ConcurrentModeEnum.THREADING,
         concurrent_num=4,
         qps=1,
         # is_using_distributed_frequency_control=True,
         is_send_consumer_hearbeat_to_redis = True,
         is_using_rpc_mode=True)
def post_process_synthseg_task(func_params  : Dict[str,any]):
    task_params = intput_params.PostProcessSynthsegTaskParams.model_validate(func_params)
    save_mode = task_params.save_mode
    cmb_file_list = task_params.model_dump()['cmb_file_list']

    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/post_process_synthseg.py '
               '--save_mode {} '
               '--cmb_file_list {}'.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                            PYTHON3,
                                            save_mode,
                                            ' '.join(cmb_file_list),)
               )
    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print('post_process_synthseg_task', stdout, stderr)
    return stdout,stderr


@Booster('call_pipeline_synthseg_tensorflow_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode=ConcurrentModeEnum.SOLO,
         concurrent_num=5,
         qps=1,
         is_send_consumer_hearbeat_to_redis=True,
         is_push_to_dlx_queue_when_retry_max_times=True,
         is_using_rpc_mode=True)
def call_pipeline_synthseg_tensorflow(func_params  : Dict[str,any]):
    ID = func_params['ID']
    Inputs = ','.join(func_params['Inputs'])
    Output_folder = func_params['Output_folder']
    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/pipeline_synthseg_tensorflow.py '
               '--ID {} '
               '--Inputs {} '
               '--Output_folder {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                              PYTHON3,
                                              ID,
                                              Inputs,
                                              Output_folder, )
               )
    print('cmd_str',cmd_str)
    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout,stderr

# 16784400_20240918
@Booster('call_pipeline_synthseg5class_tensorflow',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode=ConcurrentModeEnum.SOLO,
         concurrent_num=5,
         qps=1,
         is_send_consumer_hearbeat_to_redis=True,
         is_push_to_dlx_queue_when_retry_max_times=True,
         is_using_rpc_mode=True)
def call_pipeline_synthseg5class_tensorflow(func_params  : Dict[str,any]):
    ID = func_params['ID']
    Inputs = ','.join(func_params['Inputs'])
    Output_folder = func_params['Output_folder']
    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/pipeline_synthseg5class_tensorflow.py '
               '--ID {} '
               '--Inputs {} '
               '--Output_folder {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                              PYTHON3,
                                              ID,
                                              Inputs,
                                              Output_folder, )
               )
    # print('cmd_str',cmd_str)
    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout,stderr


@Booster('call_pipeline_flirt',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode=ConcurrentModeEnum.THREADING,
         concurrent_num=6,
         qps=1,
         is_send_consumer_hearbeat_to_redis=True,
         is_push_to_dlx_queue_when_retry_max_times=True,
         is_using_rpc_mode=True)
def call_pipeline_flirt(func_params  : Dict[str,any]):
    input_file         = pathlib.Path(func_params['input_file'])
    template_file      = pathlib.Path(func_params['template_file'])
    coregistration_mat = func_params.get('coregistration_mat',None)

    template_basename: str = replace_suffix(template_file.name, '')
    input_basename:    str = replace_suffix(input_file.name, '')

    # flirt_cmd_apply
    if coregistration_mat is not None and isinstance(coregistration_mat, str):
        template = template_basename.split('_')[-1]
        input_basename = '_'.join(input_basename.split('_')[:-1])
        input_basename +=  f'_{template}'
        template_coregistration_file_name = input_file.parent.joinpath(f'{input_basename}_from_{template_basename}')
        if coregistration_mat.endswith('.mat'):
            coregistration_mat = coregistration_mat.replace('.mat','')

        cmd_str = TemplateProcessor.flirt_cmd_apply.format(template_file,
                                                           input_file,
                                                           template_coregistration_file_name,
                                                           coregistration_mat)
    else:
    # flirt_cmd_base
        template_coregistration_file_name = input_file.parent.joinpath(f'{input_basename}_from_{template_basename}')
        cmd_str = TemplateProcessor.flirt_cmd_base.format(template_file, input_file, template_coregistration_file_name)
    print('cmd_str', cmd_str)

    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout,stderr


@Booster('call_resample_to_original_task',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode=ConcurrentModeEnum.THREADING,
         concurrent_num = 10,
         qps=1,
         is_send_consumer_hearbeat_to_redis=True,
         is_push_to_dlx_queue_when_retry_max_times=True,
         is_using_rpc_mode=True)
def call_resample_to_original_task(func_params: Dict[str, any]):
    raw_file            = pathlib.Path(func_params['raw_file'])
    resample_image_file = pathlib.Path(func_params['resample_image_file'])
    resample_seg_file   = pathlib.Path(func_params['resample_seg_file'])

    if raw_file.exists() and resample_image_file.exists() and resample_seg_file.exists():
        original_seg_file, argmin = resampleSynthSEG2original_z_index(raw_file=raw_file,
                                                                      resample_image_file=resample_image_file,
                                                                      resample_seg_file=resample_seg_file)
        return original_seg_file
    else:
        output_str = ''
        if not raw_file.exists():
            output_str += f'raw_file {raw_file} does not exist'
        if not resample_image_file.exists():
            output_str += f'resample_image_file {resample_image_file} does not exist'
        if not resample_seg_file.exists():
            output_str += f'resample_seg_file {resample_seg_file} does not exist'
        return output_str