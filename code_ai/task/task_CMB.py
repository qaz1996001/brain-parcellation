import os.path
import pathlib
import subprocess
from funboost import BrokerEnum, Booster, ConcurrentModeEnum
from code_ai import PYTHON3


@Booster('inference_cmb_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode=ConcurrentModeEnum.SOLO,
         qps=5,
         is_using_distributed_frequency_control=True,
         concurrent_num=5,
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



@Booster('pipeline_cmb_tensorflow_queue',
         broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
         concurrent_mode=ConcurrentModeEnum.SOLO,
         qps=1,
         concurrent_num=5,
         is_send_consumer_hearbeat_to_redis=True,
         is_using_rpc_mode=True)
def call_pipeline_cmb_tensorflow(func_params,):
    ID = func_params['ID']
    Inputs = ' '.join(func_params['Inputs'])
    Output_folder = func_params['Output_folder']

    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/pipeline_cmb_tensorflow.py '
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
    return stdout, stderr
