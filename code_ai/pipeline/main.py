"""
Example：
-------- Command Line 1
    # Command Line for WMH
        one file
            python main.py -i './forler'  -o './forler' --input_name Ax_T2_FLAIR --all False --WMH TRUE
            python main.py -i './forler/Ax_T2_FLAIR.nii.gz' --all False --WMH TRUE
            # -----------before-------------
            #  - folder
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            # -----------after-------------
            #  - folder
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Ax_T2_FLAIR_david.nii.gz
            #  -- Ax_T2_FLAIR_resample.nii.gz
            #  -- Ax_T2_FLAIR_synthseg.nii.gz
            #  -- Ax_T2_FLAIR_synthseg33.nii.gz
            #  -- Ax_T2_FLAIR_WMH.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            # ------------------------------
        more files
            python main.py -i './root' --input_name T2_FLAIR.nii.gz --all False --WMH TRUE
            # -----------before-------------
            #  - root
            #  -- folder_1
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_2
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_3
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_4
               --- ....
            # -----------after-------------
            #  - root
            #  -- folder_1
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Ax_T2_FLAIR_david.nii.gz
            #  --- Ax_T2_FLAIR_resample.nii.gz
            #  --- Ax_T2_FLAIR_synthseg.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33.nii.gz
            #  --- Ax_T2_FLAIR_WMH.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_2
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Ax_T2_FLAIR_david.nii.gz
            #  --- Ax_T2_FLAIR_resample.nii.gz
            #  --- Ax_T2_FLAIR_synthseg.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33.nii.gz
            #  --- Ax_T2_FLAIR_WMH.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_3
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Ax_T2_FLAIR_david.nii.gz
            #  --- Ax_T2_FLAIR_resample.nii.gz
            #  --- Ax_T2_FLAIR_synthseg.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33.nii.gz
            #  --- Ax_T2_FLAIR_WMH.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_4
               --- ....
-------- Command Line 2
    # Command Line for DWI
        one file
            python main.py -i './forler' -o './forler' --input_name Ax_DWI_ASSET_DWI0 --template './forler' --template_name Ax_T2_FLAIR --all False --DWI TRUE
            python main.py -i './forler/Ax_DWI_ASSET_DWI0.nii' --template './forler/Ax_T2_FLAIR.nii' --all False --DWI TRUE
            # -----------before-------------
            #  - folder
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            # -----------after-------------
            #  - folder
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Ax_T2_FLAIR_david.nii.gz
            #  -- Ax_T2_FLAIR_DWI.nii.gz
            #  -- Ax_T2_FLAIR_DWI_Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  -- Ax_T2_FLAIR_resample.nii.gz
            #  -- Ax_T2_FLAIR_synthseg.nii.gz
            #  -- Ax_T2_FLAIR_synthseg33.nii.gz
            #  -- Ax_T2_FLAIR_synthseg33_to_Ax_DWI_ASSET_DWI0_synthseg33.mat
            #  -- Ax_T2_FLAIR_synthseg33_to_Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            # ------------------------------
        more files
            python code/main.py -i './root' --input_name DWI0.nii.gz --template './root' --template_name Ax_T2_FLAIR  --all False --DWI TRUE
            # -----------before-------------
            #  - root
            #  -- folder_1
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  -- folder_2
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  -- folder_3
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  -- folder_4
               --- ....
            # -----------after-------------
            #  - root
            #  -- folder_1
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI0_resample.nii.gz
            #  --- Ax_DWI_ASSET_DWI0_synthseg.nii.gz
            #  --- Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Ax_T2_FLAIR_david.nii.gz
            #  --- Ax_T2_FLAIR_DWI.nii.gz
            #  --- Ax_T2_FLAIR_DWI_Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  --- Ax_T2_FLAIR_resample.nii.gz
            #  --- Ax_T2_FLAIR_synthseg.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33_to_Ax_DWI_ASSET_DWI0_synthseg33.mat
            #  --- Ax_T2_FLAIR_synthseg33_to_Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_2
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI0_resample.nii.gz
            #  --- Ax_DWI_ASSET_DWI0_synthseg.nii.gz
            #  --- Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Ax_T2_FLAIR_david.nii.gz
            #  --- Ax_T2_FLAIR_DWI.nii.gz
            #  --- Ax_T2_FLAIR_DWI_Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  --- Ax_T2_FLAIR_resample.nii.gz
            #  --- Ax_T2_FLAIR_synthseg.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33_to_Ax_DWI_ASSET_DWI0_synthseg33.mat
            #  --- Ax_T2_FLAIR_synthseg33_to_Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_3
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI0_resample.nii.gz
            #  --- Ax_DWI_ASSET_DWI0_synthseg.nii.gz
            #  --- Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Ax_T2_FLAIR_david.nii.gz
            #  --- Ax_T2_FLAIR_DWI.nii.gz
            #  --- Ax_T2_FLAIR_DWI_Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  --- Ax_T2_FLAIR_resample.nii.gz
            #  --- Ax_T2_FLAIR_synthseg.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33_to_Ax_DWI_ASSET_DWI0_synthseg33.mat
            #  --- Ax_T2_FLAIR_synthseg33_to_Ax_DWI_ASSET_DWI0_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_4
               --- ....

-------- Command Line 3
    # Command Line for CMB
        one file
            python main.py -i './forler' -o './forler' --input_name SWAN --template './forler' --template_name FSPGR_BRAVO --all False --CMB TRUE
            python main.py -i './forler/Ax SWAN.nii.gz' --template './forler/Sag_FSPGR_BRAVO.nii.gz' --all False --CMB TRUE
            # -----------before-------------
            #  - folder
            #  -- Ax SWAN.nii.gz
            #  -- Ax T1 FLAIR.nii.gz
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            # -----------after-------------
            #  - folder
            #  -- Ax SWAN.nii.gz
            #  -- Ax SWAN_resample.nii.gz
            #  -- Ax SWAN_synthseg33.nii.gz
            #  -- Ax T1 FLAIR.nii.gz
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            #  -- Sag_FSPGR_BRAVO_CMB.nii.gz
            #  -- Sag_FSPGR_BRAVO_CMB_Ax SWAN_synthseg33.nii.gz
            #  -- Sag_FSPGR_BRAVO_david.nii.gz
            #  -- Sag_FSPGR_BRAVO_resample.nii.gz
            #  -- Sag_FSPGR_BRAVO_synthseg.nii.gz
            #  -- Sag_FSPGR_BRAVO_synthseg33.nii.gz
            #  -- Sag_FSPGR_BRAVO_synthseg33_to_Ax SWAN_synthseg33.mat
            #  -- Sag_FSPGR_BRAVO_synthseg33_to_Ax SWAN_synthseg33.nii.gz
            # ------------------------------
        more files
            python code/main.py -i './root' --input_name SWAN.nii.gz --template './root' --template_name FSPGR_BRAVO  --all False --CMB TRUE
            # -----------before-------------
            #  - root
            #  -- folder_1
            #  --- Ax SWAN.nii.gz
            #  --- Ax T1 FLAIR.nii.gz
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_2
            #  --- Ax SWAN.nii.gz
            #  --- Ax T1 FLAIR.nii.gz
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_3
            #  --- Ax SWAN.nii.gz
            #  --- Ax T1 FLAIR.nii.gz
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_4
               --- ....
            # -----------after-------------
            #  - root
            #  -- folder_1
            #  --- Ax SWAN.nii.gz
            #  --- Ax SWAN_resample.nii.gz
            #  --- Ax SWAN_synthseg33.nii.gz
            #  --- Ax T1 FLAIR.nii.gz
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  --- Sag_FSPGR_BRAVO_CMB.nii.gz
            #  --- Sag_FSPGR_BRAVO_CMB_Ax SWAN_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO_david.nii.gz
            #  --- Sag_FSPGR_BRAVO_resample.nii.gz
            #  --- Sag_FSPGR_BRAVO_synthseg.nii.gz
            #  --- Sag_FSPGR_BRAVO_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO_synthseg33_to_Ax SWAN_synthseg33.mat
            #  --- Sag_FSPGR_BRAVO_synthseg33_to_Ax SWAN_synthseg33.nii.gz
            #  -- folder_2
            #  --- Ax SWAN.nii.gz
            #  --- Ax SWAN_resample.nii.gz
            #  --- Ax SWAN_synthseg33.nii.gz
            #  --- Ax T1 FLAIR.nii.gz
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  --- Sag_FSPGR_BRAVO_CMB.nii.gz
            #  --- Sag_FSPGR_BRAVO_CMB_Ax SWAN_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO_david.nii.gz
            #  --- Sag_FSPGR_BRAVO_resample.nii.gz
            #  --- Sag_FSPGR_BRAVO_synthseg.nii.gz
            #  --- Sag_FSPGR_BRAVO_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO_synthseg33_to_Ax SWAN_synthseg33.mat
            #  --- Sag_FSPGR_BRAVO_synthseg33_to_Ax SWAN_synthseg33.nii.gz
            #  -- folder_3
               --- ....
-------- Command Line 4
    # Command Line for david 、 synthseg33、 synthseg label
        one file
            python main.py -i './forler'  -o './forler' --input_name Sag_FSPGR_BRAVO.nii.gz --all False
            python main.py -i './forler/Sag_FSPGR_BRAVO.nii.gz' --all False
            # -----------before-------------
            #  - folder
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            # -----------after-------------
            #  - folder
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            #  -- Sag_FSPGR_BRAVO_david.nii.gz
            #  -- Sag_FSPGR_BRAVO_resample.nii.gz
            #  -- Sag_FSPGR_BRAVO_synthseg.nii.gz
            #  -- Sag_FSPGR_BRAVO_synthseg33.nii.gz
            # ------------------------------
        more files
            python main.py -i './root' --input_name Sag_FSPGR_BRAVO.nii.gz --all False
            # -----------before-------------
            #  - root
            #  -- folder_1
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_2
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_3
            #  -- Ax_DWI_ASSET_DWI0.nii.gz
            #  -- Ax_DWI_ASSET_DWI1000.nii.gz
            #  -- Ax_T2_FLAIR.nii.gz
            #  -- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_4
               --- ....
            # -----------after-------------
            #  - root
            #  -- folder_1
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Ax_T2_FLAIR_david.nii.gz
            #  --- Ax_T2_FLAIR_resample.nii.gz
            #  --- Ax_T2_FLAIR_synthseg.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_2
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Ax_T2_FLAIR_david.nii.gz
            #  --- Ax_T2_FLAIR_resample.nii.gz
            #  --- Ax_T2_FLAIR_synthseg.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_3
            #  --- Ax_DWI_ASSET_DWI0.nii.gz
            #  --- Ax_DWI_ASSET_DWI1000.nii.gz
            #  --- Ax_T2_FLAIR.nii.gz
            #  --- Ax_T2_FLAIR_david.nii.gz
            #  --- Ax_T2_FLAIR_resample.nii.gz
            #  --- Ax_T2_FLAIR_synthseg.nii.gz
            #  --- Ax_T2_FLAIR_synthseg33.nii.gz
            #  --- Sag_FSPGR_BRAVO.nii.gz
            #  -- folder_4
               --- ....
"""
import os

import pathlib
import re
import sys
import traceback

import nibabel as nib
import argparse

import numpy as np
from code_ai.utils_parcellation import CMBProcess, DWIProcess, run_wmh, run_with_WhiteMatterParcellation
from code_ai.utils_synthseg import SynthSeg, TemplateProcessor
import gc

from code_ai.utils.resample import resampleSynthSEG2original,resampleSynthSEG2original_z_index, resample_one, save_original_seg_by_argmin_z_index


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
    # 匹配 .nii or .nii.gz 改為 XX.nii.gz
    pattern = r'\.nii\.gz$|\.nii$'
    new_filename = re.sub(pattern, new_suffix, filename)
    return new_filename


def main(args):
    # --------- 檢查輸入輸出 start----------------------

    input_path = pathlib.Path(args.input)
    if ('.nii' in input_path.suffixes) or ('.nii.gz' in input_path.suffixes):
        file_list = [input_path]
    else:
        file_list = sorted(list(input_path.rglob('*.nii*')))
    assert len(file_list) > 0, 'Not find the nii.gz file'

    if args.input_name:
        file_list = list(filter(lambda x: args.input_name in x.name, file_list))


    if args.template:
        template_path = pathlib.Path(args.template)
        if ('.nii' in template_path.suffixes) or ('.nii.gz' in template_path.suffixes):
            template_file_list = [template_path]
        else:
            template_file_list = sorted(list(template_path.rglob('*.nii*')))

        if args.template_name:
            template_file_list = list(filter(lambda x: args.template_name in x.name, template_file_list))

        template_resample_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_resample.nii.gz')),
                                               template_file_list))

        template_synthseg_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_synthseg.nii.gz')),
                                               template_resample_file_list))

        template_synthseg33_file_list = list(
            map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_synthseg33.nii.gz')),
                template_resample_file_list))
    else:
        pass
        # if args.cmb or args.dwi:
        #     raise AttributeError('cmb or dwi not use --template and --template_name')

    # --------- 檢查輸入輸出 end------------------------

    # --------- 檢 參數 建立存檔名稱 start---------------

    if args.all:
        args.wm_file = True
        args.cmb = True
        args.dwi = True
        args.wmh = True

    if args.template:
        if args.cmb:
            cmb_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_{args.cmb_file}.nii.gz')),
                                     template_resample_file_list))
        else:
            cmb_file_list = []
        if args.dwi:
            dwi_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_{args.dwi_file}.nii.gz')),
                                     template_resample_file_list))
        else:
            dwi_file_list = []

        if args.wmh:
            wmh_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_{args.wmh_file}.nii.gz')),
                                     template_resample_file_list))
        else:
            wmh_file_list = []

        david_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_david.nii.gz')),
                                template_resample_file_list))
        wm_file_list    = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_wm.nii.gz')),
                                   template_resample_file_list))
        resample_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_resample.nii.gz')),
                                      file_list))
        synthseg_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_synthseg.nii.gz')),
                                      resample_file_list))
        synthseg5_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_synthseg5.nii.gz')),
                                      resample_file_list))
        synthseg33_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_synthseg33.nii.gz')),
                                        resample_file_list))
    else:

        resample_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_resample.nii.gz')),
                                      file_list))
        synthseg_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_synthseg.nii.gz')),
                                      resample_file_list))
        synthseg5_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_synthseg5.nii.gz')),
                                      resample_file_list))
        synthseg33_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_synthseg33.nii.gz')),
                                        resample_file_list))
        if args.cmb:
            cmb_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_{args.cmb_file}.nii.gz')),
                                     resample_file_list))
        else:
            cmb_file_list = []
        if args.dwi:
            dwi_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_{args.dwi_file}.nii.gz')),
                                     resample_file_list))
        else:
            dwi_file_list = []

        if args.wmh:
            wmh_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_{args.wmh_file}.nii.gz')),
                                     resample_file_list))
        else:
            wmh_file_list = []

        david_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_david.nii.gz')),
                                resample_file_list))

        wm_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_wm.nii.gz')),
                                   resample_file_list))


    if args.output:
        out_path = pathlib.Path(args.output)
        if out_path.is_dir():
            pass
        else:
            out_path = out_path.parent

        david_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                david_file_list))
        wm_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                wm_file_list))
        resample_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                      resample_file_list))
        synthseg_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                      synthseg_file_list))
        synthseg5_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                      synthseg5_file_list))
        synthseg33_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                        synthseg33_file_list))
        cmb_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                 cmb_file_list))
        dwi_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                 dwi_file_list))
        wmh_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                 wmh_file_list))
        if args.template:
            template_resample_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                                   template_resample_file_list))

            template_synthseg_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                                   template_synthseg_file_list))

            template_synthseg33_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                                     template_synthseg33_file_list))

        os.makedirs(out_path, exist_ok=True)
    else:
        pass

    if args.depth_number:
        depth_number = args.depth_number
    else:
        depth_number = 5

    # --------- 檢 參數 建立存檔名稱 end----------------
    synth_seg = SynthSeg()
    for i in range(len(file_list)):
        try:
            if args.template:
                resample_one(str(file_list[i]), str(resample_file_list[i]))
                synth_seg.run_segmentations33(path_images=str(resample_file_list[i]),
                                              path_segmentations33=str(synthseg33_file_list[i]),)

                resample_one(str(template_file_list[i]), str(template_resample_file_list[i]))
                synth_seg.run(path_images=str(template_resample_file_list[i]),
                              path_segmentations=str(template_synthseg_file_list[i]),
                              path_segmentations33=str(template_synthseg33_file_list[i]),
                              )

                original_seg_file, argmin = resampleSynthSEG2original_z_index(file_list[i],
                                                                              resample_file_list[i],
                                                                              synthseg33_file_list[i])
                synthseg_nii = nib.load(str(template_synthseg_file_list[i]))
                synthseg33_nii = nib.load(str(template_synthseg33_file_list[i]))

                synthseg_array = np.array(synthseg_nii.dataobj)
                synthseg33_array = np.array(synthseg33_nii.dataobj)
                seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(synthseg_array=synthseg_array,
                                                                                synthseg33=synthseg33_array,
                                                                                depth_number=depth_number)
                if args.wm_file:
                    out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
                    nib.save(out_nib, david_file_list[i])
                if args.cmb:
                    cmb_array = CMBProcess.run(seg_array)
                    out_nib = nib.Nifti1Image(cmb_array, synthseg_nii.affine, synthseg_nii.header)
                    nib.save(out_nib, cmb_file_list[i])
                if args.dwi:
                    dwi_array = DWIProcess.run(seg_array)
                    out_nib = nib.Nifti1Image(dwi_array, synthseg_nii.affine, synthseg_nii.header)
                    nib.save(out_nib, dwi_file_list[i])
                if args.wmh:
                    wmh_array = run_wmh(synthseg_array=synthseg_array, synthseg_array_wm=synthseg_array_wm,
                                        depth_number=depth_number)
                    out_nib = nib.Nifti1Image(wmh_array, synthseg_nii.affine, synthseg_nii.header)
                    nib.save(out_nib, str(wmh_file_list[i]))

                template_synthseg33_basename = template_synthseg33_file_list[i].name.replace('.nii.gz', '')
                synthseg33_basename = synthseg33_file_list[i].name.replace('.nii.gz', '')
                # template_coregistration_file_name = template_synthseg33_file_list[i].parent.joinpath(
                #     f'{template_synthseg33_basename}_to_{synthseg33_basename}')
                template_coregistration_file_name = template_synthseg33_file_list[i].parent.joinpath(
                    f'{synthseg33_basename}_from_{template_synthseg33_basename}')

                flirt_cmd_base_str = TemplateProcessor.flirt_cmd_base.format(template_synthseg33_file_list[i],
                                                                        synthseg33_file_list[i],
                                                                        template_coregistration_file_name)

                print(flirt_cmd_base_str)
                os.system(flirt_cmd_base_str)

                if args.cmb:
                    cmb_basename = str(cmb_file_list[i].name).replace('.nii.gz', '')
                    cmb_coregistration_file_name = '{}_from_{}'.format(synthseg33_file_list[i].parent.joinpath(
                        synthseg33_basename.replace('synthseg33', f"{args.cmb_file}")),
                        cmb_basename)
                    flirt_cmd_apply_str = TemplateProcessor.flirt_cmd_apply.format(cmb_file_list[i],
                                                                                   synthseg33_file_list[i],
                                                                                   cmb_coregistration_file_name,
                                                                                   template_coregistration_file_name,)
                    print(f'cmb {flirt_cmd_apply_str}')
                    os.system(flirt_cmd_apply_str)
                    print('save_original_seg_by_argmin_z_index ',cmb_file_list[i].parent.joinpath(f'{cmb_coregistration_file_name}.nii.gz'))
                    save_original_seg_by_argmin_z_index(file_list[i],
                                                        cmb_file_list[i].parent.joinpath(
                                                            f'{cmb_coregistration_file_name}.nii.gz'
                                                            ),
                                                        argmin=argmin)

                if args.dwi:
                    dwi_basename = str(dwi_file_list[i]).replace('.nii.gz', '')
                    dwi_coregistration_file_name = f'{dwi_basename}_{synthseg33_basename}'
                    flirt_dwi_str = fr'export FSLOUTPUTTYPE=NIFTI_GZ && flirt -in "{dwi_file_list[i]}" -ref "{synthseg33_file_list[i]}" ' \
                                    fr'-out "{dwi_coregistration_file_name}" ' \
                                    fr'-init "{template_coregistration_file_name}.mat" ' \
                                    fr'-applyxfm -interp nearestneighbour'
                    print(f'dwi')
                    print(flirt_dwi_str)
                    os.system(flirt_dwi_str)
                #
            else:
                resample_one(str(file_list[i]), str(resample_file_list[i]))

                synth_seg.run(path_images=str(resample_file_list[i]),
                              path_segmentations=str(synthseg_file_list[i]),
                              path_segmentations33=str(synthseg33_file_list[i]),
                              )
                synth_seg.run_segmentations5(path_images=str(resample_file_list[i]),
                                             path_segmentations5=str(synthseg5_file_list[i]))

                original_seg_file, argmin = resampleSynthSEG2original_z_index(file_list[i],
                                                                              resample_file_list[i],
                                                                              synthseg5_file_list[i])
                synthseg_nii = nib.load(synthseg_file_list[i])
                synthseg_array = np.array(synthseg_nii.dataobj)
                synthseg33_nii = nib.load(synthseg33_file_list[i])
                synthseg33_array = np.array(synthseg33_nii.dataobj)
                seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(synthseg_array=synthseg_array,
                                                                                synthseg33=synthseg33_array,
                                                                                depth_number=depth_number)
                out_nib = nib.Nifti1Image(synthseg_array_wm, synthseg_nii.affine, synthseg_nii.header)
                nib.save(out_nib, wm_file_list[i])
                out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
                nib.save(out_nib, david_file_list[i])
                original_seg_file = save_original_seg_by_argmin_z_index(file_list[i],
                                                                        synthseg_file_list[i],
                                                                        argmin)
                original_seg_file = save_original_seg_by_argmin_z_index(file_list[i],
                                                                        synthseg33_file_list[i],
                                                                        argmin)
                original_seg_file = save_original_seg_by_argmin_z_index(file_list[i],
                                                                        wm_file_list[i],
                                                                        argmin)
                original_seg_file = save_original_seg_by_argmin_z_index(file_list[i],
                                                                        david_file_list[i],
                                                                        argmin)
                if args.cmb:
                    cmb_array = CMBProcess.run(seg_array)
                    out_nib = nib.Nifti1Image(cmb_array, synthseg_nii.affine, synthseg_nii.header)
                    nib.save(out_nib, cmb_file_list[i])
                    original_seg_file = save_original_seg_by_argmin_z_index(file_list[i],
                                                                            cmb_file_list[i],
                                                                            argmin)
                if args.dwi:
                    dwi_array = DWIProcess.run(seg_array)
                    out_nib = nib.Nifti1Image(dwi_array, synthseg_nii.affine, synthseg_nii.header)
                    nib.save(out_nib, dwi_file_list[i])
                    original_seg_file = save_original_seg_by_argmin_z_index(file_list[i],
                                                                            dwi_file_list[i],
                                                                            argmin)
                if args.wmh:
                    wmh_array = run_wmh(synthseg_array=synthseg_array,
                                        synthseg_array_wm=synthseg_array_wm,
                                        depth_number=depth_number)
                    out_nib = nib.Nifti1Image(wmh_array, synthseg_nii.affine, synthseg_nii.header)
                    nib.save(out_nib, wmh_file_list[i])
                    original_seg_file = save_original_seg_by_argmin_z_index(file_list[i],
                                                                            wmh_file_list[i],
                                                                            argmin)
        # file_list
        # coregistration

        except Exception as e:
            print(f'{file_list[i]} is Error')
            error_class = e.__class__.__name__  # 取得錯誤類型
            detail = e.args[0]  # 取得詳細內容
            cl, exc, tb = sys.exc_info()  # 取得Call Stack
            lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
            fileName = lastCallStack[0]  # 取得發生的檔案名稱
            lineNum = lastCallStack[1]  # 取得發生的行號
            funcName = lastCallStack[2]  # 取得發生的函數名稱
            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            print(errMsg)
            print(traceback.print_exc())
        gc.collect()


if __name__ == '__main__':
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus, device_type='GPU')
    # print(gpus, cpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    parser = argparse.ArgumentParser()

    # args, unrecognized_args = parser.parse_known_args()

    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        # default=r'D:\00_Chen\Task03_\VCI_out_4cases_nii_gz_T1',
                        help="input the (SHH seg)synthseg file path (nii , nii.gz) or input the folder path.\r\n"
                             "Example ： python utils_parcellation.py -i input_path ")
    parser.add_argument('--input_name', dest='input_name', type=str,
                        help="")
    parser.add_argument('-o', '--output', dest='output', type=str,
                        # default=r'D:\00_Chen\Task03_\VCI_out_4cases_nii_gz_T1',
                        help="output the result file , if None then output file to input parameter path.\r\n"
                             "Example ： python utils_parcellation.py -i input_path -o output_path")

    parser.add_argument('--template', dest='template', type=str, )
    parser.add_argument('--template_name', dest='template_name', type=str, )

    parser.add_argument('--all', dest='all', type=str_to_bool, default=True,
                        help="run all algorithm with input (file,folder)，default is True.")

    parser.add_argument('--david', dest='wm_file', type=str_to_bool, default=True,
                        help="Output white matter parcellation file ，"
                             "File name  is \'{output}_david.nii.gz\'")

    parser.add_argument('--CMB', '--cmb', dest='cmb', type=str_to_bool, default=False,
                        help="output CMB Mask")
    parser.add_argument('--CMBFile', '--cmbFile', dest='cmb_file', type=str, default='CMB',
                        help="CMB Mask file name，default is \'{output}_CMB.nii.gz\' ， use \'{output}_{"
                             "cmb_file}.nii.gz\'")

    parser.add_argument('--DWI', '--dwi', dest='dwi', type=str_to_bool, default=False,
                        help="output DWI Mask")
    parser.add_argument('--DWIFile', '--dwiFile', dest='dwi_file', type=str, default='DWI',
                        help="DWI Mask file name，default is \'{output}_DWI.nii.gz\'")

    parser.add_argument('--WMH', '--wmh', dest='wmh', type=str_to_bool, default=False,
                        help="output WMH Mask")
    parser.add_argument('--WMHFile', '--wmhFile', dest='wmh_file', type=str, default='WMH_PVS',
                        help="WMH Mask file name，default is \'{output}_WMH.nii.gz\'")

    parser.add_argument('--depth_number', dest='depth_number', default=5, type=int, choices=[4, 5, 6, 7, 8, 9, 10]
                        , help="deep white matter parameter")

    args = parser.parse_args()
    print(args)
    main(args)