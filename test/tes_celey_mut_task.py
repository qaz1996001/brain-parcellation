import argparse
import pathlib
import re
import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
# import code_ai
from code_ai.task.task_synthseg import build_celery_workflow


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
    # python D:\00_Chen\Task04_git\code_ai\main.py -i D:\00_Chen\Task04_git\data --input_name BRAVO.nii
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data --input_name BRAVO.nii -o /mnt/d/00_Chen/Task04_git/data_0106
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data --input_name BRAVO.nii -o /mnt/d/00_Chen/Task04_git/data_0106
    # python D:\00_Chen\Task04_git\code_ai\main.py -i D:\00_Chen\Task04_git\data --input_name SWAN.nii --template_name BRAVO.nii --CMB True

    parser.add_argument('-i', '--input', dest='input', type=str, required=True, help="input file or folder path.")
    parser.add_argument('--input_name', dest='input_name', type=str, help="Filter files by name.")
    parser.add_argument('-o', '--output', dest='output', type=str, help="Output directory for result files.")

    parser.add_argument('--template', dest='template', type=str, help="Template file or folder path.")
    parser.add_argument('--template_name', dest='template_name', type=str, help="Filter template files by name.")

    parser.add_argument('--all', dest='all', type=str_to_bool, default=True, help="Run all algorithms.")

    parser.add_argument('--david', dest='wm_file', type=str_to_bool, default=True,
                        help="Output white matter parcellation file.")
    parser.add_argument('--CMB', '--cmb', dest='cmb', type=str_to_bool, default=False, help="Output CMB Mask.")
    parser.add_argument('--CMBFile', '--cmbFile', dest='cmb_file', type=str, default='CMB', help="CMB Mask file name.")
    parser.add_argument('--DWI', '--dwi', dest='dwi', type=str_to_bool, default=False, help="Output DWI Mask.")
    parser.add_argument('--DWIFile', '--dwiFile', dest='dwi_file', type=str, default='DWI', help="DWI Mask file name.")
    parser.add_argument('--WMH', '--wmh', dest='wmh', type=str_to_bool, default=False, help="Output WMH Mask.")
    parser.add_argument('--WMHFile', '--wmhFile', dest='wmh_file', type=str, default='WMHPVS', help="WMH Mask file name.")
    parser.add_argument('--depth_number', dest='depth_number', default=5, type=int, choices=[4, 5, 6, 7, 8, 9, 10],
                        help="Deep white matter parameter.")

    args = parser.parse_args()
    input_path = pathlib.Path(args.input)
    file_list = get_file_list(input_path, ['.nii', '.nii.gz'], args.input_name)
    file_list = file_list
    print('file_list',file_list)
    assert file_list, 'No files found with .nii or .nii.gz extension'

    output_dir = pathlib.Path(args.output) if args.output else None
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    args.intput_file_list = file_list
    args.resample_file_list = prepare_output_file_list(file_list, '_resample.nii.gz', output_dir)
    args.synthseg_file_list = prepare_output_file_list(args.resample_file_list, '_synthseg.nii.gz', output_dir)
    args.synthseg33_file_list = prepare_output_file_list(args.resample_file_list, '_synthseg33.nii.gz', output_dir)
    args.david_file_list = prepare_output_file_list(args.resample_file_list, '_david.nii.gz', output_dir)
    args.wm_file_list = prepare_output_file_list(args.resample_file_list, '_wm.nii.gz', output_dir)
    args.cmb_file_list = prepare_output_file_list(args.resample_file_list, f'_{args.cmb_file}.nii.gz',
                                                  output_dir) if args.cmb else []
    args.dwi_file_list = prepare_output_file_list(args.resample_file_list, f'_{args.dwi_file}.nii.gz',
                                                  output_dir) if args.dwi else []
    args.wmh_file_list = prepare_output_file_list(args.resample_file_list, f'_{args.wmh_file}.nii.gz',
                                                  output_dir) if args.wmh else []
    workflow_group = build_celery_workflow(args, file_list)
    result = workflow_group.apply_async()
    print('result',result)
    result.get()
    print('result.get()',result.get())

