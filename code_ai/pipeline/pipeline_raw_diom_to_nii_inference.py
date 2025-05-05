import argparse
import os
import pathlib
import re
import subprocess
from typing import List,Optional
from code_ai import PYTHON3,PATH_DICOM2NII
from code_ai.utils_inference import build_analysis, Analysis, InferenceEnum


def build_dicom_to_nifti_cmd_str(args) -> Optional[str]:
    cmd_str = None
    if all((args.input_dicom, args.output_dicom, args.output_nifti)):
        input_dicom_path = pathlib.Path(args.input_dicom)
        output_dicom_path = pathlib.Path(args.output_dicom)
        output_nifti_path = pathlib.Path(args.output_nifti)
        cmd_str = ('export PYTHONPATH={} && '
                   '{} code_ai/dicom2nii/main_call.py '
                   '--input_dicom {} '
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


def build_inference_cmd_str(nifti_study_path :pathlib.Path) -> Optional[str]:
    from code_ai.pipeline import pipelines
    analysis :Analysis = build_analysis(nifti_study_path)
    # 使用管道配置
    cmd_list = []
    for key, value in analysis.model_dump().items():
        if key in pipelines:
            task = getattr(analysis,key)
            cmd_str = pipelines[key].generate_cmd(analysis.study_id,task)
            cmd_list.append((key, cmd_str))


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
    dicom_to_nifti_cmd_str = build_dicom_to_nifti_cmd_str(args=args)

    process = subprocess.Popen(args=dicom_to_nifti_cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    pattern = re.compile('nifti_output_study_set\s(.*)\snifti_output_study_set',re.MULTILINE)
    match_result = pattern.match(stdout.decode('utf-8'))
    match_result.groups()
    if match_result is not None :
        study_str:str = match_result.groups()[0]
        print('study_str',study_str)
        study_str_list = study_str.split(',')
        nifti_study_list = list(map(lambda x:pathlib.Path(x),study_str_list))
        print('nifti_study_list', nifti_study_list)
        inference_cmd_str_list = list(map(lambda x:build_inference_cmd_str(x),nifti_study_list))

