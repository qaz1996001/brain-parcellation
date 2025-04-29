import argparse
import pathlib
import sys

import orjson

sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
from code_ai.utils_inference import check_study_id,check_study_mapping_inference, generate_output_files
from code_ai.utils_inference import get_synthseg_args_file
from code_ai.utils_inference import Analysis,InferenceEnum,Dataset,Task,Result



def model_inference(intput_args):
    miss_inference = {}
    input_dir = pathlib.Path(intput_args.output_nifti)
    output_inference_path = pathlib.Path(intput_args.output_inference)
    input_dir_list = sorted(input_dir.iterdir())
    study_list = list(filter(check_study_id, input_dir_list))
    mapping_inference_list = list(map(check_study_mapping_inference, study_list))
    analyses = {}
    for mapping_inference in mapping_inference_list:
        study_id = mapping_inference.keys()
        model_dict_values = mapping_inference.values()
        base_output_path = str(output_inference_path.joinpath(*study_id))
        for task_dict in model_dict_values:
            tasks = {}
            for model_name, input_paths in task_dict.items():
                task_output_files = generate_output_files(input_paths, model_name, base_output_path)
                tasks[model_name] = Task(
                    intput_path_list=input_paths,
                    output_path=base_output_path,
                    output_path_list=task_output_files,
                    # result=Result(output_file=task_output_files)
                )
        analyses[str(*study_id)] = Analysis(**tasks)
    dataset = Dataset(analyses=analyses)
    return dataset


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
    args = parser.parse_args()
    input_dicom_path = args.input_dicom
    output_dicom_path = args.output_dicom
    output_nifti_path = args.output_nifti
    print(args.output_nifti)
    print(args.output_inference)


    dataset = model_inference(args)
    intput_args = (
    '< GroupResult: 779d7309-10e5-4cf7-a786-b4769acdf768[0b84ded4-01c5-4d0f-b95f-5c7378cc3ace] >,',
    dataset.model_dump_json()
    )


    result = app.send_task('code_ai.task.task_CMB.inference_cmb', args=(intput_args,args.output_inference),
                           queue='synthseg_queue',
                           routing_key='celery')
    print('result', result, type(result))
    print('collect_list end')


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
# python test/tes_CMB.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom_20250206 --output_nifti /mnt/e/rename_nifti_20250206 --output_inference /mnt/e/rename_nifti_20250206