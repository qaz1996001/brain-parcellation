import argparse
import pathlib
import re
import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
# import code_ai
from code_ai.task.task_dicom2nii import ConvertManager


def get_file_list(input_path, suffixes, filter_name=None):
    if any(suffix in input_path.suffixes for suffix in suffixes):
        file_list = [input_path]
    else:
        file_list = sorted(list(input_path.rglob('*.nii*')))
    if filter_name:
        file_list = [f for f in file_list if filter_name in f.name]
    return file_list


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('false', 'no', 'n', 'f'):
        return False
    elif v.lower() in ('true', 'yes', 'y', 't'):
        return True
    else:
        raise argparse.ArgumentTypeError("Bool value expected")


def prepare_output_file_list(file_list, suffix, output_dir=None):
    return [output_dir.joinpath(x.parent.name,replace_suffix(f'{x.name}',suffix)) if output_dir else x.parent.joinpath(
        replace_suffix(x.name, suffix)) for x in file_list]


def replace_suffix(filename, new_suffix):
    pattern = r'\.nii\.gz$|\.nii$'
    return re.sub(pattern, new_suffix, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dicom', dest='input_dicom', type=str,
                        help="input the raw dicom folder.\r\n")
    parser.add_argument('--output_dicom', dest='output_dicom', type=str,
                        help="output the rename dicom folder.\r\n")

    # python test/tes_dicom2nii_task.py --input_dicom /mnt/d/00_Chen/Task04_git/data_dicom/raw --output_dicom /mnt/d/00_Chen/Task04_git/data_dicom/rename

    args = parser.parse_args()
    input_dicom_path = args.input_dicom
    output_dicom_path = args.output_dicom

    convert_manager = ConvertManager(input_dicom_path, output_dicom_path)
    print(convert_manager.run())
    print(100000000)


    # workflow_group = build_celery_workflow(args, file_list)
    # result = workflow_group.apply_async()
    # print('result',result)
    # result.get()
    # print('result.get()',result.get())

