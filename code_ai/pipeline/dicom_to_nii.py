import argparse
import pathlib
import subprocess
from typing import List

import nibabel as nib
import numpy as np

if __name__ == '__main__':
    from code_ai.task.task_dicom2nii import dicom_to_nii, process_dir
    from code_ai.task.schema.intput_params import Dicom2NiiParams

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dicom', type=str, required=True,
                        help='input_dicom')
    parser.add_argument('--output_dicom', type=str, required=True,
                        help='output_dicom')
    parser.add_argument('--output_nifti', type=str, required=True,
                        help='output_nifti')

    args = parser.parse_args()
    input_dicom  = pathlib.Path(args.input_dicom)
    output_dicom_path = pathlib.Path(args.output_dicom)
    output_nifti_path = pathlib.Path(args.output_nifti)

    input_dicom_list = sorted(input_dicom.iterdir())
    input_dicom_list = list(filter(lambda x: x.is_dir(),input_dicom_list))

    for input_dicom_path in input_dicom_list:
        task_params = Dicom2NiiParams(
            sub_dir=input_dicom_path,
            output_dicom_path=output_dicom_path,
            output_nifti_path=output_nifti_path,)
        task = dicom_to_nii.push(task_params.get_str_dict())