import os.path
import bentoml
import orjson
from celery import shared_task
from . import CMB_INFERENCE_URL,TIME_OUT,MAX_RETRIES,COUNTDOWN, app


@app.task(bind=True,acks_late=True,rate_limit='30/s')
def inference_wmh(self,intput_args,):
    print(f'inference_wmh intput_args {intput_args} ')
    mapping_inference_data_dict = orjson.loads(intput_args)
    for study_id, task_dict in mapping_inference_data_dict.items():
        temp_task = task_dict.get('WMH')
        if temp_task is None:
            return intput_args
        else:
            if 'T2FLAIR_AXI' in temp_task.get('input_path_list')[0]:
                intput_path_str = temp_task.get('input_path_list')[0]
            else:
                intput_path_str = temp_task.get('input_path_list')[1]

            output_nii_path_str = list(filter(lambda x: ('Pred_WMH' in x) and x.endswith('.nii.gz'),
                                              temp_task.get('output_path_list')))
            output_json_path_str = list(filter(lambda x: x.endswith('.json'),
                                               temp_task.get('output_path_list')))
            print('intput_path_str', intput_path_str)
            print('output_nii_path_str', output_nii_path_str[0])
            print('output_json_path_str', output_json_path_str[0])
