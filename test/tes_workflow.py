import argparse
import pathlib
import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
from celery import Celery


app = Celery('tasks',
             broker='pyamqp://guest:guest@localhost:5672/celery',
             backend='redis://localhost:10079/1'
             )
app.config_from_object('code_ai.celery_config')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dicom', dest='input_dicom', type=str,
                        help="input the raw dicom folder.\r\n")
    parser.add_argument('--output_dicom', dest='output_dicom', type=str,
                        help="output the rename dicom folder.\r\n")
    parser.add_argument('--output_nifti', dest='output_nifti', type=str,
                        help="rename dicom output to nifti folder.\r\n"
                             "Example ： python tes_dicom2nii_and_synthseg_task.py --input_dicom raw_dicom_path --output_dicom rename_dicom_path "
                             "--output_nifti output_nifti_path")
    parser.add_argument('--output_inference', dest='output_inference', type=str,
                        help="model inference output folder.\r\n")

    # python test/tes_dicom2nii_and_synthseg_task.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom --output_nifti /mnt/e/rename_nifti --output_inference /mnt/e/rename_nifti
    # python test/tes_dicom2nii_and_synthseg_task.py --input_dicom /mnt/d/00_Chen/Task08/data/Study_Glymphatics --output_dicom /mnt/e/rename_dicom --output_nifti /mnt/e/rename_nifti --output_inference /mnt/e/rename_nifti
    # python test/tes_dicom2nii_and_synthseg_task.py --input_dicom /mnt/d/00_Chen/Task08/data/study_VCI --output_dicom /mnt/e/rename_dicom --output_nifti /mnt/e/rename_nifti --output_inference /mnt/e/rename_nifti

    args = parser.parse_args()
    input_dicom_path = args.input_dicom
    output_dicom_path = args.output_dicom
    output_nifti_path = args.output_nifti
    print(args.output_nifti)
    print(args.output_inference)
    result = app.send_task('code_ai.task.workflow.celery_workflow', args=(input_dicom_path,
                                                                          output_dicom_path,
                                                                          output_nifti_path),
                           queue='default',
                           routing_key='celery')
    # 946a02a5-79ea-4b0f-9133-793fc2db9beb
    # if collect_list:
    #     print('result', result, type(result))
    #     print('result', type(result))
    #     print('args', args)
    #     model_inference(args)
    #     print(10000)

    # # 获取任务的 ID
    # task_id = '65cb5a30-c480-4827-bf24-5bc2a938d867'
    # # 创建 AsyncResult 实例
    # result = AsyncResult(task_id)
    # print('result',result,type(result))
    # print(list(result.collect()))
    # print('result', type(result))
    # print('collect_list start')

# python test/tes_dicom2nii_and_synthseg_task.py --input_dicom /mnt/e/raw_dicom1 --output_dicom /mnt/e/rename_dicom1 --output_nifti /mnt/e/rename_nifti1 --output_inference /mnt/e/rename_nifti1
# python test/tes_dicom2nii_and_synthseg_task.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom1 --output_nifti /mnt/e/rename_nifti1 --output_inference /mnt/e/rename_nifti1

# D:\00_Chen\Task08\data\raw_dicom\stroke
# python test/tes_dicom2nii_and_synthseg_task.py --input_dicom /mnt/d/00_Chen/Task08/data/raw_dicom/stroke --output_dicom /mnt/e/rename_dicom1 --output_nifti /mnt/e/rename_nifti1 --output_inference /mnt/e/rename_nifti1
# python test/tes_dicom2nii_and_synthseg_task.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom_20250204 --output_nifti /mnt/e/rename_nifti_20250204 --output_inference /mnt/e/rename_nifti_20250204