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


def run_dicom_rename_mr(executor: ProcessPoolExecutor,
                        input_path,
                        output_path
                        ):
    from convert.dicom_rename_mr import ConvertManager
    start = time.time()
    # convert_manager = ConvertManager(input_path=input_path, output_path=output_path)
    # convert_manager.run()
    with executor:
        convert_manager = ConvertManager(input_path=input_path, output_path=output_path)
        convert_manager.run(executor=executor)
    end = time.time()
    print(start, end, end - start)


def run_dicom_rename_postprocess(output_dicom_path):
    from convert.dicom_rename_mr_postprocess import PostProcessManager
    input_path = pathlib.Path(output_dicom_path)
    post_process_manager = PostProcessManager(input_path=input_path)
    post_process_manager.run(executor=None)


def run_list_dicom(output_dicom_path: str):
    from convert.list_dicom import list_dicom
    data_path = pathlib.Path(output_dicom_path)
    list_dicom(data_path=data_path)


def run_convert_nifti(executor: ProcessPoolExecutor,
                      input_path,
                      output_path
                      ):
    from convert.convert_nifti import Dicm2NiixConverter
    with executor:
        converter = Dicm2NiixConverter(input_path=input_path, output_path=output_path)
        converter.convert_dicom_to_nifti(executor=executor)


def run_list_nifti(output_nifti_path: str):
    from convert.list_nii import list_nifti
    data_path = pathlib.Path(output_nifti_path)
    list_nifti(data_path=data_path)


def run_convert_nifti_postprocess(output_nifti_path):
    from convert.convert_nifti_postprocess import PostProcessManager
    input_path = pathlib.Path(output_nifti_path)
    post_process_manager = PostProcessManager(input_path=input_path)
    post_process_manager.run(executor=None)


if __name__ == '__main__':
    args = parse_arguments()
    input_dicom_path = args.input_dicom
    output_dicom_path = args.output_dicom
    output_nifti_path = args.output_nifti
    work_count = args.work
    file_path = pathlib.Path(__file__).absolute().parent
    sys.path.append(str(file_path))
    if input_dicom_path and output_dicom_path:
        dicom_work = min(2, max(1, args.work))
        dicom_rename_executor = ProcessPoolExecutor(max_workers=dicom_work)
        run_dicom_rename_mr(executor=dicom_rename_executor,
                            input_path=input_dicom_path,
                            output_path=output_dicom_path)
        run_list_dicom(output_dicom_path=output_dicom_path)
        run_dicom_rename_postprocess(output_dicom_path=output_dicom_path)

    if output_dicom_path and output_nifti_path:
        nii_work = min(4, max(1, args.work))
        convert_nifti_executor = ProcessPoolExecutor(max_workers=nii_work)
        run_convert_nifti(executor=convert_nifti_executor,
                          input_path=output_dicom_path,
                          output_path=output_nifti_path)
        run_list_nifti(output_nifti_path=output_nifti_path)
        print('run_convert_nifti_postprocess')
        run_convert_nifti_postprocess(output_nifti_path)
