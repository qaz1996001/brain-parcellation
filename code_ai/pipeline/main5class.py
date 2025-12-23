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
import argparse
from code_ai.utils_synthseg import SynthSeg
import gc
from code_ai.utils.resample import resampleSynthSEG2original_z_index, resample_one

# from code_ai.utils_resample import resampleSynthSEG2original_z_index, resample_one

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


    # --------- 檢查輸入輸出 end------------------------

    # --------- 檢 參數 建立存檔名稱 start---------------


    resample_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_resample.nii.gz')),
                                  file_list))
    synthseg5_file_list = list(map(lambda x: x.parent.joinpath(replace_suffix(x.name, f'_synthseg5.nii.gz')),
                                  resample_file_list))


    if args.output:
        out_path = pathlib.Path(args.output)
        if out_path.is_dir():
            pass
        else:
            out_path = out_path.parent

        resample_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                      resample_file_list))
        synthseg5_file_list = list(map(lambda x: out_path.joinpath(f'{str(x.parent.name)}_{x.name}'),
                                      synthseg5_file_list))

        os.makedirs(out_path, exist_ok=True)
    else:
        pass


    # --------- 檢 參數 建立存檔名稱 end----------------
    synth_seg = SynthSeg()
    for i in range(len(file_list)):
        try:
            resample_one(str(file_list[i]), str(resample_file_list[i]))
            synth_seg.run_segmentations5(path_images=str(resample_file_list[i]),
                                         path_segmentations5 = str(synthseg5_file_list[i]))
            original_seg_file,argmin = resampleSynthSEG2original_z_index(file_list[i],
                                                                         resample_file_list[i],
                                                                         synthseg5_file_list[i])
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
    args = parser.parse_args()
    print(args)
    main(args)