import bentoml
import orjson

from code_ai.utils_inference import Analysis,InferenceEnum,Dataset,Task,Result
from celery import shared_task
from . import app



# swan_path_str ="/mnt/e/rename_nifti_20250204/02695350_20240109_MR_21210300104/SWAN.nii.gz",
# temp_path_str = "/mnt/e/rename_nifti_20250204/02695350_20240109_MR_21210300104/synthseg_SWAN_original_CMB_from_synthseg_T1BRAVO_AXI_original_CMB.nii.gz",
# out_root_str = "/mnt/e/rename_nifti_20250204/02695350_20240109_MR_21210300104"



@shared_task(bind=True,acks_late=True)
def inference_cmb(self, intput_args,
                  output_nifti_str
                  ):
    print('inference_cmb',self)
    print('intput_args',intput_args)
    print('output_nifti_str', output_nifti_str)
    mapping_inference_data_dict = orjson.loads(intput_args[1])

    # for study_id, task_dict in mapping_inference_data_dict['analyses'].items():
    #     cmb_task = task_dict.get('CMB')
    #     if cmb_task is None:
    #         return intput_args[1]
    #     else:
    #         if 'SWAN' in cmb_task.get('input_path_list')[0]:
    #             swan_path_str = cmb_task.get('input_path_list')[0]
    #         else:
    #             swan_path_str = cmb_task.get('input_path_list')[1]
    #
    #         temp_path_str = list(filter(lambda x:x.endswith('.nii.gz') and ('synthseg_' in x),
    #                                     cmb_task.get('output_path_list')))
    #         output_nii_path_str = list(filter(lambda x: ('Pred_CMB' in x) and x.endswith('.nii.gz'),
    #                                           cmb_task.get('output_path_list')))
    #         output_json_path_str = list(filter(lambda x: x.endswith('.json'),
    #                                            cmb_task.get('output_path_list')))
    #         print('swan_path_str',swan_path_str)
    #         print('temp_path_str', temp_path_str[0])
    #         print('output_nii_path_str', output_nii_path_str[0])
    #         print('output_json_path_str', output_json_path_str[0])
    #
    #         with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    #             result = client.cmb_classify(swan_path_str=swan_path_str,
    #                                          temp_path_str=temp_path_str[0],
    #                                          output_nii_path_str=output_nii_path_str[0],
    #                                          output_json_path_str=output_json_path_str[0])
    #
    #         return intput_args[1]
