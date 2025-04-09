import os.path
import pathlib
import subprocess

from funboost import BrokerEnum, Booster
from code_ai import PYTHON3
from code_ai.pipeline.cmb import CMBServiceTF


@Booster('inference_cmb_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10,
         concurrent_num=4,
         is_send_consumer_hearbeat_to_redis=True,
         is_using_rpc_mode=True)
def inference_cmb(func_params,):
    print(f'inference_cmb func_params {func_params} ')
    study_id = func_params.get('study_id')
    cmb_task = func_params.get('CMB')
    if cmb_task is None:
        return func_params
    else:
        if 'SWAN' in cmb_task.get('input_path_list')[0]:
            swan_path_str = cmb_task.get('input_path_list')[0]
        else:
            swan_path_str = cmb_task.get('input_path_list')[1]

        # output_path = cmb_task.get('output_path')
        # if os.path.basename(output_path) == study_id:
        #     output_path = os.path.dirname(output_path)

        temp_path_str = list(filter(lambda x: x.endswith('.nii.gz') and ('synthseg_' in x),
                                    cmb_task.get('output_path_list')))
        output_nii_path_str = list(filter(lambda x: ('Pred_CMB' in x) and x.endswith('.nii.gz'),
                                          cmb_task.get('output_path_list')))
        output_json_path_str = list(filter(lambda x: x.endswith('.json'),
                                           cmb_task.get('output_path_list')))
        print('swan_path_str', swan_path_str)
        print('temp_path_str', temp_path_str[0])
        print('output_nii_path_str', output_nii_path_str[0])
        print('output_json_path_str', output_json_path_str[0])
        if os.path.exists(temp_path_str[0]):
            cmd_str = (' export PYTHONPATH={} && '
                       ' {} code_ai/pipeline/cmb.py '
                       ' --swan_path_str {} '
                       ' --temp_path_str {} '
                       ' --output_nii_path_str {} '
                       ' --output_json_path_str {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                                     PYTHON3,
                                                     swan_path_str,
                                                     temp_path_str[0],
                                                     output_nii_path_str[0],
                                                     output_json_path_str[0],)
                       )
            process = subprocess.Popen(args=cmd_str, shell=True,
                                       # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
        else:
            print('inference_cmb retry')

        return swan_path_str, temp_path_str[0],output_nii_path_str[0],output_json_path_str[0]


#
# # @app.task(bind=True,acks_late=True,rate_limit='300/s')
# def inference_cmb(self,
#                   intput_args,
#                   ):
#     print(f'inference_cmb intput_args {intput_args} ')
#     mapping_inference_data_dict = orjson.loads(intput_args)
#     for study_id, task_dict in mapping_inference_data_dict['analyses'].items():
#         cmb_task = task_dict.get('CMB')
#         if cmb_task is None:
#             return intput_args
#         else:
#             if 'SWAN' in cmb_task.get('input_path_list')[0]:
#                 swan_path_str = cmb_task.get('input_path_list')[0]
#             else:
#                 swan_path_str = cmb_task.get('input_path_list')[1]
#
#             temp_path_str = list(filter(lambda x:x.endswith('.nii.gz') and ('synthseg_' in x),
#                                         cmb_task.get('output_path_list')))
#             output_nii_path_str = list(filter(lambda x: ('Pred_CMB' in x) and x.endswith('.nii.gz'),
#                                               cmb_task.get('output_path_list')))
#             output_json_path_str = list(filter(lambda x: x.endswith('.json'),
#                                                cmb_task.get('output_path_list')))
#             print('swan_path_str',swan_path_str)
#             print('temp_path_str', temp_path_str[0])
#             print('output_nii_path_str', output_nii_path_str[0])
#             print('output_json_path_str', output_json_path_str[0])
#             if os.path.exists(temp_path_str[0]):
#                 with bentoml.SyncHTTPClient(CMB_INFERENCE_URL, timeout=TIME_OUT) as client:
#                     try:
#                         result = client.cmb_classify(swan_path_str=swan_path_str,
#                                                      temp_path_str=temp_path_str[0],
#                                                      output_nii_path_str=output_nii_path_str[0],
#                                                      output_json_path_str=output_json_path_str[0])
#                         return intput_args[1],result
#                     except:
#                         print('inference_cmb except')
#                         self.retry(countdown=COUNTDOWN, max_retries=MAX_RETRIES)  # 重試任務
#
#             else:
#                 print('inference_cmb retry')
#                 self.retry(countdown=COUNTDOWN, max_retries=MAX_RETRIES)  # 重試任務

#  uv tree  --universal
#  uv tree  --locked > requirements_1.txt