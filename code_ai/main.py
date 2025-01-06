import os
import tensorflow as tf


def set_gpu(gpu_id='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        print(device)

    # select single GPU to use
    try:  # allow GPU memory growth
        tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    except:
        pass

    tf.config.set_visible_devices(devices=physical_devices[0], device_type='GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    print('Seleted logical_devices:', logical_devices)

    tf.debugging.set_log_device_placement(True)
    tf.config.set_soft_device_placement(enabled=True)

import argparse
import gc
import os
import pathlib
import re
import sys
import traceback
from typing import List, Optional

import nibabel as nib
import nibabel.processing
import numpy as np
from utils_resample import resample_one, resampleSynthSEG2original
from utils_parcellation import CMBProcess, DWIProcess, run_wmh, run_with_WhiteMatterParcellation
from utils_synthseg import SynthSeg


class ProcessingStrategy:
    def process(self, args, file_list: List[pathlib.Path], template_file_list: Optional[List[pathlib.Path]]):
        raise NotImplementedError("Subclasses should implement this!")

    @staticmethod
    def data_translate_back(img, nii):
        header = nii.header.copy()  # 抓出nii header 去算體積
        pixdim = header['pixdim']  # 可以借此從nii的header抓出voxel size
        if pixdim[0] > 0:
            img = np.flip(img, 1)
        img = np.flip(img, -1)
        img = np.flip(img, 0)
        img = np.swapaxes(img, 1, 0)
        # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
        return img
    @staticmethod
    # 會使用到的一些predict技巧
    def data_translate(img, nii):
        img = np.swapaxes(img, 0, 1)
        img = np.flip(img, 0)
        img = np.flip(img, -1)
        header = nii.header.copy()  # 抓出nii header 去算體積
        pixdim = header['pixdim']  # 可以借此從nii的header抓出voxel size
        if pixdim[0] > 0:
            img = np.flip(img, 1)
            # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
        return img



# Concrete Strategy for Processing with Template
class TemplateProcessingStrategy(ProcessingStrategy):
    # (
    #     f'flirt -in "{synthseg33_file}" -ref "{template_synthseg33_file}" -out '
    #     f'"{template_coregistration_file_name}" -dof 6 -cost corratio -omat '
    #     f'"{template_coregistration_file_name}.mat" -interp nearestneighbour'
    # )
    flirt_cmd_base = (
        'flirt -in "{0}" -ref "{1}" -out '
        '"{2}" -dof 6 -cost corratio -omat '
        '"{2}.mat" -interp nearestneighbour'
    )
    #             f'flirt -in "{cmb_file}" -ref "{template_synthseg33_file}" '
    #             f'-out "{cmb_coregistration_file_name}" '
    #             f'-init "{coregistration_file_name}.mat" '
    #             f'-applyxfm -interp nearestneighbour'
    flirt_cmd_apply = (
        'flirt -in "{0}" -ref "{1}" '
        '-out "{2}" -init "{3}.mat" '
        '-applyxfm -interp nearestneighbour'
    )

    def process(self, args, file_list: List[pathlib.Path], template_file_list: Optional[List[pathlib.Path]]):
        synth_seg = SynthSeg()
        depth_number = args.depth_number or 5
        # CMB file SWAN
        for i, file in enumerate(file_list):
            try:
                # CMB BRAVO
                resample_file = args.resample_file_list[i]
                synthseg_file = args.synthseg_file_list[i]
                synthseg33_file = args.synthseg33_file_list[i]
                template_file = template_file_list[i]

                self.resample_and_segment(synth_seg, template_file, resample_file, synthseg_file, synthseg33_file)
                synthseg_nii = nib.load(synthseg_file)
                synthseg_array = np.array(synthseg_nii.dataobj)
                synthseg33_nii = nib.load(synthseg33_file)
                synthseg33_array = np.array(synthseg33_nii.dataobj)

                seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(
                    synthseg_array, synthseg33_array, depth_number)

                self.save_output_files(args, synthseg_nii, synthseg_array, seg_array, synthseg_array_wm, i)

                self.process_with_template(
                    args=args, synth_seg=synth_seg, file=file, template_file=template_file,
                    synthseg33_file=synthseg33_file, index=i)

            except Exception as e:
                log_error(file, e)
            gc.collect()

    def resample_and_segment(self, synth_seg:SynthSeg,
                             file:pathlib.Path, resample_file:pathlib.Path,
                             synthseg_file:pathlib.Path, synthseg33_file:pathlib.Path):
        if not resample_file.parent.exists():
            resample_file.parent.mkdir(parents=True, exist_ok=True)
        if not synthseg_file.parent.exists():
            synthseg_file.parent.mkdir(parents=True, exist_ok=True)
        if not synthseg33_file.parent.exists():
            synthseg33_file.parent.mkdir(parents=True, exist_ok=True)

        resample_one(str(file), str(resample_file))
        synth_seg.run(path_images=str(resample_file), path_segmentations=str(synthseg_file),
                      path_segmentations33=str(synthseg33_file))

    def save_output_files(self, args, synthseg_nii, synthseg_array, seg_array, synthseg_array_wm, index):
        FileSaver.save(args, synthseg_nii, synthseg_array, seg_array, synthseg_array_wm, index)

    def process_with_template(self, args, synth_seg, file, template_file, synthseg33_file, index):
        try:
            # Resample the template file
            template_resample_file = file.parent.joinpath(
                replace_suffix(file.name, '_resample.nii.gz'))
            resample_one(str(file), str(template_resample_file))

            # Run segmentation on the resampled template
            template_synthseg_file = template_resample_file.parent.joinpath(
                replace_suffix(template_resample_file.name, '_synthseg.nii.gz'))
            template_synthseg33_file = template_resample_file.parent.joinpath(
                replace_suffix(template_resample_file.name, '_synthseg33.nii.gz'))
            synth_seg.run(path_images=str(template_resample_file),
                          path_segmentations=str(template_synthseg_file),
                          path_segmentations33=str(template_synthseg33_file))

            # Co-registration using FLIRT
            template_synthseg33_basename : pathlib.Path = template_synthseg33_file.parent.joinpath(
                replace_suffix(template_synthseg33_file.name, ''))
            synthseg33_basename :str               = replace_suffix(synthseg33_file.name, '')
            template_coregistration_file_name :str = f'{template_synthseg33_basename}_from_{synthseg33_basename}'

            flirt_cmd_str = self.flirt_cmd_base.format(synthseg33_file, template_synthseg33_file,template_coregistration_file_name)
            print('FSL flirt：', flirt_cmd_str)
            os.system(flirt_cmd_str)

            # Apply transformations for CMB, DWI, and other optional outputs
            apply_transform(args, template_coregistration_file_name, template_synthseg33_file,index)

        except Exception as e:
            log_error(file, e)


# Concrete Strategy for Processing without Template
class NoTemplateProcessingStrategy(ProcessingStrategy):
    def process(self, args, file_list: List[pathlib.Path], template_file_list: Optional[List[pathlib.Path]]):
        synth_seg = SynthSeg()
        depth_number = args.depth_number or 5

        for i, file in enumerate(file_list):
            try:
                resample_file = args.resample_file_list[i]
                synthseg_file = args.synthseg_file_list[i]
                synthseg33_file = args.synthseg33_file_list[i]
                # synthseg_file = resample_file.parent.joinpath(replace_suffix(resample_file.name, '_synthseg.nii.gz'))
                # synthseg33_file = resample_file.parent.joinpath(replace_suffix(resample_file.name, '_synthseg33.nii.gz'))
                if not resample_file.parent.exists():
                    resample_file.parent.mkdir(parents=True, exist_ok=True)
                if not synthseg_file.parent.exists():
                    synthseg_file.parent.mkdir(parents=True, exist_ok=True)
                if not synthseg33_file.parent.exists():
                    synthseg33_file.parent.mkdir(parents=True, exist_ok=True)

                resample_one(str(file), str(resample_file))
                synth_seg.run(path_images=str(resample_file), path_segmentations=str(synthseg_file),
                              path_segmentations33=str(synthseg33_file))

                synthseg_nii = nib.load(synthseg_file)
                synthseg33_nii = nib.load(synthseg33_file)

                synthseg_array = np.array(synthseg_nii.dataobj)
                synthseg33_array = np.array(synthseg33_nii.dataobj)

                seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(
                    synthseg_array, synthseg33_array, depth_number)

                FileSaver.save(args, synthseg_nii, synthseg_array, seg_array, synthseg_array_wm, i)

                #
                self.resample_to_original(file,resample_file,args.wm_file_list[ i])

            except Exception as e:
                log_error(file, e)
            gc.collect()
            # break
    def resample_to_original(self,raw_file:pathlib.Path,
                             resample_image_file:pathlib.Path,
                             resample_seg_file:pathlib.Path):
        resampleSynthSEG2original(raw_file,resample_image_file,resample_seg_file)
        pass

# Context class to use the strategy
class FileProcessor:
    def __init__(self, strategy: ProcessingStrategy):
        self.strategy = strategy

    def execute(self, args, file_list, template_file_list=None):
        self.strategy.process(args, file_list, template_file_list)


# Helper class for saving files
class FileSaver:
    @staticmethod
    def save(args, synthseg_nii, synthseg_array, seg_array, synthseg_array_wm, index):
        if args.wm_file:
            FileSaver.save_nifti(seg_array, synthseg_nii.affine, synthseg_nii.header, args.wm_file_list[index])

        if args.cmb:
            cmb_array = CMBProcess.run(seg_array)
            FileSaver.save_nifti(cmb_array, synthseg_nii.affine, synthseg_nii.header, args.cmb_file_list[index])

        if args.dwi:
            dwi_array = DWIProcess.run(seg_array)
            FileSaver.save_nifti(dwi_array, synthseg_nii.affine, synthseg_nii.header, args.dwi_file_list[index])

        if args.wmh:
            wmh_array = run_wmh(synthseg_array, synthseg_array_wm, args.depth_number)
            FileSaver.save_nifti(wmh_array, synthseg_nii.affine, synthseg_nii.header, args.wmh_file_list[index])

    @staticmethod
    def save_nifti(data, affine, header, filename):
        out_nib = nib.Nifti1Image(data, affine, header)
        nib.save(out_nib, filename)


# Utility functions
def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('false', 'no', 'n', 'f'):
        return False
    elif v.lower() in ('true', 'yes', 'y', 't'):
        return True
    else:
        raise argparse.ArgumentTypeError("Bool value expected")


def replace_suffix(filename, new_suffix):
    pattern = r'\.nii\.gz$|\.nii$'
    return re.sub(pattern, new_suffix, filename)


def get_file_list(input_path, suffixes, filter_name=None):
    if any(suffix in input_path.suffixes for suffix in suffixes):
        file_list = [input_path]
    else:
        file_list = sorted(list(input_path.rglob('*.nii*')))
    if filter_name:
        file_list = [f for f in file_list if filter_name in f.name]
    return file_list


def prepare_output_file_list(file_list, suffix, output_dir=None):
    return [output_dir.joinpath(x.parent.name,replace_suffix(f'{x.name}',suffix)) if output_dir else x.parent.joinpath(
        replace_suffix(x.name, suffix)) for x in file_list]


def apply_transform(args, coregistration_file_name:pathlib.Path, template_synthseg33_file:Optional[pathlib.Path],index:int):
    if args.cmb:
        cmb_file = args.cmb_file_list[index]
        cmb_basename = replace_suffix(cmb_file.name,'')
        cmb_coregistration_file_name = template_synthseg33_file.parent.joinpath(
            f'{template_synthseg33_file.name.replace("synthseg33.nii.gz", f"from_{cmb_basename}.nii.gz")}')
        flirt_cmd_str = (
            f'flirt -in "{cmb_file}" -ref "{template_synthseg33_file}" '
            f'-out "{cmb_coregistration_file_name}" '
            f'-init "{coregistration_file_name}.mat" '
            f'-applyxfm -interp nearestneighbour'
        )
        print(f'cmb {flirt_cmd_str}')
        os.system(flirt_cmd_str)

    if args.dwi:
        dwi_file = args.dwi_file_list[index]
        dwi_basename = dwi_file.replace('.nii.gz', '')
        dwi_coregistration_file_name = f'{dwi_basename}_{synthseg33_file.replace(".nii.gz", "")}'
        flirt_cmd_str = (
            f'flirt -in "{dwi_file}" -ref "{synthseg33_file}" '
            f'-out "{dwi_coregistration_file_name}" '
            f'-init "{coregistration_file_name}.mat" '
            f'-applyxfm -interp nearestneighbour'
        )
        print(f'dwi {flirt_cmd_str}')
        os.system(flirt_cmd_str)


def log_error(file, exception):
    print(f'{file} is Error')
    error_class = exception.__class__.__name__
    detail = exception.args[0]
    cl, exc, tb = sys.exc_info()
    last_call_stack = traceback.extract_tb(tb)[-1]
    file_name = last_call_stack[0]
    line_num = last_call_stack[1]
    func_name = last_call_stack[2]
    errMsg = f"File \"{file_name}\", line {line_num}, in {func_name}: [{error_class}] {detail}"
    print(errMsg)


# Main function
def main(args):
    input_path = pathlib.Path(args.input)
    file_list = get_file_list(input_path, ['.nii', '.nii.gz'], args.input_name)

    assert file_list, 'No files found with .nii or .nii.gz extension'

    output_dir = pathlib.Path(args.output) if args.output else None
    print('if output_dir and output_dir.is_file()',(output_dir and output_dir.is_file()))
    if output_dir and output_dir.is_file():
        output_dir = output_dir.parent
        os.makedirs(output_dir, exist_ok=True)
    elif output_dir :
        os.makedirs(output_dir, exist_ok=True)
    else:
        pass

    template_file_list = None
    strategy = NoTemplateProcessingStrategy()

    if args.template:
        template_path = pathlib.Path(args.template)
        template_file_list = get_file_list(template_path, ['.nii', '.nii.gz'], args.template_name)
        strategy = TemplateProcessingStrategy()
        args.resample_file_list   = prepare_output_file_list(template_file_list, '_resample.nii.gz', output_dir)

    elif args.template_name is not None and (args.template is None):
        template_path = pathlib.Path(args.input)
        template_file_list = get_file_list(template_path, ['.nii', '.nii.gz'], args.template_name)
        strategy = TemplateProcessingStrategy()
        args.resample_file_list = prepare_output_file_list(template_file_list, '_resample.nii.gz', output_dir)
    else:
        args.resample_file_list = prepare_output_file_list(file_list, '_resample.nii.gz', output_dir)


    args.synthseg_file_list = prepare_output_file_list(args.resample_file_list, '_synthseg.nii.gz', output_dir)
    args.synthseg33_file_list = prepare_output_file_list(args.resample_file_list, '_synthseg33.nii.gz', output_dir)
    args.wm_file_list = prepare_output_file_list(args.resample_file_list, '_david.nii.gz', output_dir)
    args.cmb_file_list = prepare_output_file_list(args.resample_file_list, f'_{args.cmb_file}.nii.gz',
                                                  output_dir) if args.cmb else []
    args.dwi_file_list = prepare_output_file_list(args.resample_file_list, f'_{args.dwi_file}.nii.gz',
                                                  output_dir) if args.dwi else []
    args.wmh_file_list = prepare_output_file_list(args.resample_file_list, f'_{args.wmh_file}.nii.gz',
                                                  output_dir) if args.wmh else []

    print(args.synthseg_file_list)
    print(args.synthseg33_file_list)
    print(args.wm_file_list)
    print(args.cmb_file_list)
    print(args.dwi_file_list)
    print(args.wmh_file_list)
    print(file_list)

    processor = FileProcessor(strategy)
    processor.execute(args, file_list, template_file_list)


if __name__ == '__main__':
    set_gpu(gpu_id='0')
    parser = argparse.ArgumentParser()
    # python D:\00_Chen\Task04_git\code_ai\main.py -i D:\00_Chen\Task04_git\data --input_name BRAVO.nii
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
    parser.add_argument('--WMHFile', '--wmhFile', dest='wmh_file', type=str, default='WMH', help="WMH Mask file name.")
    parser.add_argument('--depth_number', dest='depth_number', default=5, type=int, choices=[4, 5, 6, 7, 8, 9, 10],
                        help="Deep white matter parameter.")

    args = parser.parse_args()
    # print(args)
    main(args)
