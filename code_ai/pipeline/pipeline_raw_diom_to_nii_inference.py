import argparse
import json
import os
import pathlib
import re
import subprocess
from typing import Optional
from code_ai import PYTHON3
from code_ai.utils_inference import build_inference_cmd
from dotenv import load_dotenv
load_dotenv()


def build_dicom_to_nifti_cmd_str(args) -> Optional[str]:
    cmd_str = None
    if all((args.input_dicom, args.output_dicom, args.output_nifti)):
        input_dicom_path = pathlib.Path(args.input_dicom)
        output_dicom_path = pathlib.Path(args.output_dicom)
        output_nifti_path = pathlib.Path(args.output_nifti)
        cmd_str = ('export PYTHONPATH={} && '
                   '{} code_ai/dicom2nii/main_call.py '
                   '--input_dicom \"{}\" '
                   '--output_dicom {} '
                   '--output_nifti {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                               PYTHON3,
                                               input_dicom_path,
                                               output_dicom_path,
                                               output_nifti_path)
                   )

    elif all((args.input_dicom, args.output_dicom)):
        input_dicom_path = pathlib.Path(args.input_dicom)
        output_dicom_path = pathlib.Path(args.output_dicom)
        cmd_str = ('export PYTHONPATH={} && '
                   '{} code_ai/dicom2nii/main_call.py '
                   '--input_dicom {} '
                   '--output_dicom {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                               PYTHON3,
                                               input_dicom_path,
                                               output_dicom_path, )
                   )

    elif all((args.output_dicom, args.output_nifti)):
        output_dicom_path = pathlib.Path(args.output_dicom)
        output_nifti_path = pathlib.Path(args.output_nifti)
        cmd_str = ('export PYTHONPATH={} && '
                   '{} code_ai/dicom2nii/main_call.py '
                   '--output_dicom {} '
                   '--output_nifti {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                               PYTHON3,
                                               output_dicom_path,
                                               output_nifti_path)
                   )

    return cmd_str




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dicom', dest='input_dicom', type=str,
                        help="input the raw dicom folder.\r\n")
    parser.add_argument('--output_dicom', dest='output_dicom', type=str,
                        help="output the rename dicom folder.\r\n")
    parser.add_argument('--output_nifti', dest='output_nifti', type=str,
                        help="rename dicom output to nifti folder.\r\n"
                             "Example ： python main.py --input_dicom raw_dicom_path --output_dicom rename_dicom_path "
                             "--output_nifti output_nifti_path")

    args = parser.parse_args()

    path_code = os.getenv("PATH_CODE")
    path_process = os.getenv("PATH_PROCESS")
    path_cmd_tools = os.path.join(path_process, 'Deep_cmd_tools')
    path_json = os.getenv("PATH_JSON")
    path_log = os.getenv("PATH_LOG")
    # 建置資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_cmd_tools, exist_ok=True)  # 如果資料夾不存在就建立，

    dicom_to_nifti_cmd_str = build_dicom_to_nifti_cmd_str(args=args)
    print(dicom_to_nifti_cmd_str)
    process = subprocess.Popen(args=dicom_to_nifti_cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    pattern = re.compile('nifti_output_study_set\s(.*)\snifti_output_study_set',re.MULTILINE)
    match_result = pattern.match(stdout.decode('utf-8'))
    match_result.groups()
    if match_result is not None :
        output_dicom = pathlib.Path(args.output_dicom)
        study_str:str = match_result.groups()[0]
        study_str_list = study_str.split(',')
        nifti_study_list = list(map(lambda x:pathlib.Path(x),study_str_list))
        inference_item_cmd_list = list(map(lambda x:build_inference_cmd(x,output_dicom), nifti_study_list))
        print('inference_item_cmd', inference_item_cmd_list)
        for inference_item_cmd in inference_item_cmd_list:
            cmd_output_path = os.path.join(path_cmd_tools,f'{inference_item_cmd.cmd_items[0].study_id}_cmd.json')
            with open(cmd_output_path, 'w') as f:
                f.write(json.dumps(inference_item_cmd.model_dump()['cmd_items']))


