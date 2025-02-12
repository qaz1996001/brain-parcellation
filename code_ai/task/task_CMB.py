import os.path
import bentoml
import orjson
from celery import shared_task
from . import CMB_INFERENCE_URL,TIME_OUT,MAX_RETRIES,COUNTDOWN

@shared_task(bind=True,acks_late=True)
def inference_cmb(self,args,
                  intput_args,
                  ):
    mapping_inference_data_dict = orjson.loads(intput_args)
    if mapping_inference_data_dict.get('analyses') is None:
        return
    for study_id, task_dict in mapping_inference_data_dict['analyses'].items():
        cmb_task = task_dict.get('CMB')
        if cmb_task is None:
            return intput_args
        else:
            if 'SWAN' in cmb_task.get('input_path_list')[0]:
                swan_path_str = cmb_task.get('input_path_list')[0]
            else:
                swan_path_str = cmb_task.get('input_path_list')[1]

            temp_path_str = list(filter(lambda x:x.endswith('.nii.gz') and ('synthseg_' in x),
                                        cmb_task.get('output_path_list')))
            output_nii_path_str = list(filter(lambda x: ('Pred_CMB' in x) and x.endswith('.nii.gz'),
                                              cmb_task.get('output_path_list')))
            output_json_path_str = list(filter(lambda x: x.endswith('.json'),
                                               cmb_task.get('output_path_list')))
            print('swan_path_str',swan_path_str)
            print('temp_path_str', temp_path_str[0])
            print('output_nii_path_str', output_nii_path_str[0])
            print('output_json_path_str', output_json_path_str[0])
            if os.path.exists(temp_path_str[0]):
                with bentoml.SyncHTTPClient(CMB_INFERENCE_URL, timeout=TIME_OUT) as client:
                    result = client.cmb_classify(swan_path_str=swan_path_str,
                                                 temp_path_str=temp_path_str[0],
                                                 output_nii_path_str=output_nii_path_str[0],
                                                 output_json_path_str=output_json_path_str[0])
                return intput_args[1],result
            else:
                print('inference_cmb retry')
                self.retry(countdown=COUNTDOWN, max_retries=MAX_RETRIES)  # 重試任務