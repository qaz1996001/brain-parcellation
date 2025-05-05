import argparse
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dicom', dest='input_dicom', type=str,
                        help="input the raw dicom folder.\r\n")
    parser.add_argument('--output_dicom', dest='output_dicom', type=str,
                        help="output the rename dicom folder.\r\n")
    parser.add_argument('--output_nifti', dest='output_nifti', type=str,
                        help="rename dicom output to nifti folder.\r\n"
                             "Example ： python main.py --input_dicom raw_dicom_path --output_dicom rename_dicom_path "
                             "--output_nifti output_nifti_path")
    parser.add_argument('--work', dest='work', type=int, default=4,
                        help="Thread cont .\r\n"
                             "Example ： "
                             "--output_nifti output_nifti_path --work 4")
    parser.add_argument('--upload_all', dest='upload_all', type=str,
                        help="Thread cont .\r\n"
                             "Example ： "
                             "--upload_all True")
    parser.add_argument('--upload_nifti', dest='upload_nifti', type=str,
                        help="upload rename nifti folder to sql and object storage\r\n"
                             "Example ： "
                             "--upload_nifti True")
    parser.add_argument('--upload_dicom', dest='upload_dicom', type=str,
                        help="upload rename dicom all file to NAS\r\n"
                             "Example ： "
                             "--upload_dicom True")


    return parser.parse_args()


def run_dicom_rename_mr(work:int,
                        input_path,
                        output_path
                        ):
    from convert.dicom_rename_mr import ConvertManager
    convert_manager = ConvertManager(input_path=input_path, output_path=output_path)
    output_study_set = convert_manager.run_with_work(work)
    return output_study_set


def run_dicom_rename_postprocess(output_study_set):
    from convert.dicom_rename_mr_postprocess import PostProcessManager
    post_process_manager = PostProcessManager()
    for output_study in output_study_set:
        post_process_manager.post_process(study_path=output_study)


def run_list_dicom(output_dicom_path: str):
    from convert.list_dicom import list_dicom
    data_path = pathlib.Path(output_dicom_path)
    list_dicom(data_path=data_path)


def convert_dicom_task(converter):
    return converter.convert_dicom_to_nifti()


def run_convert_nifti(executor: ProcessPoolExecutor,
                      output_study_set,
                      output_path
                      ):
    from convert.convert_nifti import Dicm2NiixConverter
    converter_objects = [Dicm2NiixConverter(input_path=input_path, output_path=output_path)
                         for input_path in output_study_set]

    with executor:
        results = list(executor.map(convert_dicom_task, converter_objects))
        # 攤平結果
        flattened_result = set()
        for result in results:
            if isinstance(result, set):
                flattened_result.update(result)
            else:
                flattened_result.add(result)

        return flattened_result


def run_list_nifti(output_nifti_path: str):
    from convert.list_nii import list_nifti
    data_path = pathlib.Path(output_nifti_path)
    list_nifti(data_path=data_path)


def run_convert_nifti_postprocess(nifti_output_study_set):
    from convert.convert_nifti_postprocess import PostProcessManager
    post_process_manager = PostProcessManager()
    for study_path in nifti_output_study_set:
        post_process_manager.post_process(study_path=study_path)



if __name__ == '__main__':
    args = parse_arguments()
    input_dicom_path = args.input_dicom
    output_dicom_path = args.output_dicom
    output_nifti_path = args.output_nifti
    work_count = args.work
    file_path = pathlib.Path(__file__).absolute().parent
    sys.path.append(str(file_path))
    if input_dicom_path and output_dicom_path and output_nifti_path:
        dicom_work = min(4, max(1, args.work))
        output_study_set = run_dicom_rename_mr(work=dicom_work,
                                               input_path=input_dicom_path,
                                               output_path=output_dicom_path)
        run_dicom_rename_postprocess(output_study_set=output_study_set)

        nii_work = min(4, max(1, args.work))
        convert_nifti_executor = ProcessPoolExecutor(max_workers=nii_work)
        if output_study_set is not None:
            nifti_output_study_set =  run_convert_nifti(executor=convert_nifti_executor,
                                                        output_study_set=output_study_set,
                                                        output_path=output_nifti_path)
            run_convert_nifti_postprocess(nifti_output_study_set)

