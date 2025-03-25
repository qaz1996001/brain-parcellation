import argparse
import pathlib
import subprocess
from typing import List

import nibabel as nib
import numpy as np

if __name__ == '__main__':
    from code_ai.task import CMBProcess, DWIProcess, run_wmh
    from code_ai.utils_inference import InferenceEnum
    from code_ai.utils_synthseg import TemplateProcessor
    from code_ai.utils_inference import replace_suffix

    parser = argparse.ArgumentParser()
    parser.add_argument('--synthseg_file', type=str, required=True,
                        help='synthseg_file')
    parser.add_argument('--david_file', type=str, required=True,
                        help='david_file')
    parser.add_argument('--wm_file', type=str, required=True,
                        help='wm_file')
    parser.add_argument('--depth_number', type=int, default=5,
                        help='depth_number')
    parser.add_argument('--save_mode', type=str, required=True,
                        help='save_mode')
    parser.add_argument('--save_file_path', type=str, required=True,
                        help='save_file_path')

    parser.add_argument('--cmb_file_list', type=str,nargs='+',
                        help='cmb_file_list')

    args = parser.parse_args()
    synthseg_file = args.synthseg_file
    david_file = args.david_file
    wm_file = args.wm_file
    depth_number = args.depth_number
    save_mode = args.save_mode
    save_file_path = args.save_file_path

    synthseg_nii = nib.load(synthseg_file)
    david_nii = nib.load(david_file)
    seg_array = np.array(david_nii.dataobj)
    affine = synthseg_nii.affine
    header = synthseg_nii.header
    print('save_mode',save_mode)
    match save_mode:
        case InferenceEnum.CMB:
            result_array = CMBProcess.run(seg_array)
        case InferenceEnum.WMH_PVS:
            synthseg_wm_nii = nib.load(wm_file)
            synthseg_array_wm = np.array(synthseg_wm_nii.dataobj)
            result_array = run_wmh(np.array(synthseg_nii.dataobj), synthseg_array_wm, depth_number)
        case InferenceEnum.DWI:
            result_array = DWIProcess.run(seg_array)
        case _:
            result_array = None
    if result_array is not None:
        out_nib = nib.Nifti1Image(result_array, affine, header)
        nib.save(out_nib, save_file_path)
