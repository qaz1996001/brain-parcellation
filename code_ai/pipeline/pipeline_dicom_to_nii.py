import argparse
import os
import pathlib
from typing import List,Optional
from code_ai import PYTHON3,PATH_DICOM2NII
from code_ai.utils_inference import build_analysis

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
    if all((args.input_dicom, args.output_dicom, args.output_nifti)):
        input_dicom_path  = pathlib.Path(args.input_dicom)
        output_dicom_path = pathlib.Path(args.output_dicom)
        output_nifti_path = pathlib.Path(args.output_nifti)

    elif all((args.input_dicom, args.output_dicom)):
        input_dicom_path = pathlib.Path(args.input_dicom)
        output_dicom_path = pathlib.Path(args.output_dicom)

    elif all((args.output_dicom, args.output_nifti)):
        output_dicom_path = pathlib.Path(args.output_dicom)
        output_nifti_path = pathlib.Path(args.output_nifti)

    study_path_list = sorted(output_nifti_path.iterdir())
    analysis = build_analysis(study_path_list[0])
    print(analysis)
    # 00_Chen/Task04_git/code_ai/pipeline/pipeline_dicom_to_nii.sh --input_dicom /mnt/e/raw_dicom/02695350_21210300104  --output_dicom /mnt/e/rename_dicom_20250421/202504211716 --output_nifti /mnt/e/rename_nifti_20250421/202504211716