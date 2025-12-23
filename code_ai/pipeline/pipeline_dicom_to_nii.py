import argparse
import pathlib
import subprocess
import sys

from code_ai import PYTHON3

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
                                               output_dicom_path,)
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
                                               output_nifti_path )
                   )
    else:
        sys.exit(1)

    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout,stderr)