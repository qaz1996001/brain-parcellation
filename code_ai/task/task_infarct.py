import os.path
import pathlib
import subprocess
from typing import Dict

import orjson
from funboost import BrokerEnum, Booster

from code_ai import PYTHON3


@Booster('inference_infarct_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=4,
         concurrent_num=4,
         is_send_consumer_hearbeat_to_redis=True,
         is_using_rpc_mode=True)
def inference_infarct(func_params  : Dict[str,any]):
    study_id = func_params.get('study_id')
    dwi_task = func_params.get('DWI')
    temp_task = func_params.get('Infarct')
    if temp_task is None or dwi_task is None:
        return func_params
    else:
        input_path_list = temp_task.get('input_path_list')
        ADC_file = list(filter(lambda x:x.endswith('.nii.gz') and ('ADC' in x),
                               input_path_list))[0]
        DWI0_file = list(filter(lambda x:x.endswith('.nii.gz') and ('DWI0' in x),
                               input_path_list))[0]
        DWI1000_file = list(filter(lambda x:x.endswith('.nii.gz') and ('DWI1000' in x),
                               input_path_list))[0]
        output_path  =  temp_task.get('output_path')
        if os.path.basename(output_path) == study_id:
            output_path = os.path.dirname(output_path)

        SynthSEG_file = dwi_task.get('output_path_list')[0]

        cmd_str = (' export PYTHONPATH={} && '
                   ' {} code_ai/pipeline/pipeline_infarct_tensorflow.py '
                   ' --ID {} '
                   ' --Inputs {} {} {} {} '
                   ' --Output_folder {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                               PYTHON3,
                                                study_id,
                                                ADC_file,
                                                DWI0_file,
                                                DWI1000_file,
                                                SynthSEG_file,
                                                output_path)
                   )
        if os.path.exists(SynthSEG_file):

            process = subprocess.Popen(args=cmd_str, shell=True,
                                       # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
        else:
            print('inference_infarct retry')



# import os.path
# import bentoml
# import orjson
# from code_ai.task import CMB_INFERENCE_URL,TIME_OUT,MAX_RETRIES,COUNTDOWN
#
#
# def inference_infarct(self,
#                       intput_args):
#     print(f'inference_infarct intput_args {intput_args} ')
#     mapping_inference_data_dict = orjson.loads(intput_args)
#     for study_id, task_dict in mapping_inference_data_dict["analyses"].items():
#         dwi_task = task_dict.get('DWI')
#         temp_task = task_dict.get('Infarct')
#         print(f'inference_infarct dwi {dwi_task} ')
#         print(f'inference_infarct temp {temp_task} ')
#
#         if temp_task is None or dwi_task is None:
#             return intput_args
#         else:
#             input_path_list = temp_task.get('input_path_list')
#             ADC_file = list(filter(lambda x:x.endswith('.nii.gz') and ('ADC' in x),
#                                    input_path_list))[0]
#             DWI0_file = list(filter(lambda x:x.endswith('.nii.gz') and ('DWI0' in x),
#                                    input_path_list))[0]
#             DWI1000_file = list(filter(lambda x:x.endswith('.nii.gz') and ('DWI1000' in x),
#                                    input_path_list))[0]
#             output_path  = temp_task.get('output_path')
#             SynthSEG_file = dwi_task.get('output_path_list')[0]
#             if os.path.exists(SynthSEG_file):
#                 with bentoml.SyncHTTPClient(CMB_INFERENCE_URL, timeout=TIME_OUT) as client:
#                     try:
#                         result = client.infarct_classify(adc_file=ADC_file,
#                                                          dwi0_file=DWI0_file,
#                                                          dwi1000_file=DWI1000_file,
#                                                          synthseg_file=SynthSEG_file,
#                                                          output_path = output_path)
#                         return intput_args[1],result
#                     except:
#                         print('inference_infarct except')
#                         self.retry(countdown=COUNTDOWN, max_retries=MAX_RETRIES)  # 重試任務
#
#             else:
#                 print('inference_infarct retry')
#                 self.retry(countdown=COUNTDOWN, max_retries=MAX_RETRIES)  # 重試任務
