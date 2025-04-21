import argparse
import os
import pathlib
from typing import List,Optional
from code_ai import PYTHON3,PATH_DICOM2NII


def pipeline_raw_dicom_rename_dicom_to_nii(input_dicom_path  : Optional[pathlib.Path] = None,
                                           output_dicom_path : Optional[pathlib.Path] = None,
                                           output_nifti_path : Optional[pathlib.Path] = None,) :
    cmd_line = '{} {} --input_dicom {} --output_dicom {} --output_nifti {}'.format(
        PYTHON3,PATH_DICOM2NII,
        input_dicom_path, output_dicom_path,output_nifti_path)

    print(cmd_line)
    os.system(cmd_line)


def pipeline_raw_dicom_rename_dicom(input_dicom_path  : Optional[pathlib.Path] = None,
                                    output_dicom_path : Optional[pathlib.Path] = None,) :
    cmd_line = '{} {} --input_dicom {} --output_dicom {}'.format(
        PYTHON3,PATH_DICOM2NII,input_dicom_path,output_dicom_path)

    print(cmd_line)
    os.system(cmd_line)


def pipeline_rename_dicom_to_nii(output_dicom_path : Optional[pathlib.Path] = None,
                                 output_nifti_path : Optional[pathlib.Path] = None,):
    cmd_line = '{} {} --output_dicom {} --output_nifti {}'.format(
        PYTHON3, PATH_DICOM2NII, output_dicom_path, output_nifti_path)
    print(cmd_line)
    pass


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
        pipeline_raw_dicom_rename_dicom_to_nii(input_dicom_path=input_dicom_path,
                                               output_dicom_path=output_dicom_path,
                                               output_nifti_path=output_nifti_path)
    elif all((args.input_dicom, args.output_dicom)):
        input_dicom_path = pathlib.Path(args.input_dicom)
        output_dicom_path = pathlib.Path(args.output_dicom)
        pipeline_raw_dicom_rename_dicom(input_dicom_path=input_dicom_path,
                                        output_dicom_path=output_dicom_path)
    elif all((args.output_dicom, args.output_nifti)):
        output_dicom_path = pathlib.Path(args.output_dicom)
        output_nifti_path = pathlib.Path(args.output_nifti)
        pipeline_rename_dicom_to_nii(output_dicom_path=output_dicom_path,
                                     output_nifti_path=output_nifti_path)
