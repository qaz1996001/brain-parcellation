import os
import glob
import pathlib
import re
import time
import numpy as np
import pandas as pd
import nibabel as nib
import argparse

from review_predict import resample_one, SynthSeg

from utils_parcellation import CMBProcess, DWIProcess, run, run_wmh, run_with_WhiteMatterParcellation

# python D:\00_Chen\Task04_\code\main_bak.py -i 'D:\01_Liu\Task04_OHIF Platform\13385662\original'  -o 'D:\01_Liu\Task04_OHIF Platform\13385662\original' --all False --DWI TRUE
# python D:\00_Chen\Task04_\code\main_bak.py -i 'D:\01_Liu\Task04_OHIF Platform\13385662\original'  -o 'D:\01_Liu\Task04_OHIF Platform\13385662\original' --all TRUE

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('false', 'no', 'n', 'f'):
        return False
    elif v.lower() in ('true', 'yes', 'y', 't'):
        return True
    else:
        raise argparse.ArgumentTypeError("Bool value expected")


def main(args):
    # --------- 檢查輸入輸出 start----------------------
    if args.input.endswith('nii') or args.input.endswith('nii.gz'):
        file_list = [args.input]
    else:
        file_list = glob.glob(f'{args.input}/*.nii*', recursive=True)
    assert len(file_list) > 0, 'Not find the nii.gz file'

    if args.input_name:
        file_list = list(filter(lambda x: args.input_name in x, file_list))

    if args.output:
        out_path = args.output
    else:
        if os.path.isdir(args.input):
            out_path = args.input
        else:
            out_path = os.path.dirname(args.input)
    os.makedirs(out_path, exist_ok=True)

    # --------- 檢查輸入輸出 end------------------------
    def replace_suffix(filename, new_suffix):
        # 匹配 .nii or .nii.gz 改為 XX.nii.gz
        pattern = r'\.nii\.gz$|\.nii$'
        new_filename = re.sub(pattern, new_suffix, filename)
        return new_filename

    # --------- 檢 參數 建立存檔名稱 start---------------
    if args.all:
        cmb_file_list = list(map(lambda x: replace_suffix(x, f'_{args.cmb_file}.nii.gz'), file_list))
        dwi_file_list = list(map(lambda x: replace_suffix(x, f'_{args.dwi_file}.nii.gz'), file_list))
        wmh_file_list = list(map(lambda x: replace_suffix(x, f'_{args.wmh_file}.nii.gz'), file_list))
        wm_file_list = list(map(lambda x: replace_suffix(x, f'_david.nii.gz'), file_list))
    else:
        if args.cmb:
            cmb_file_list = list(map(lambda x: replace_suffix(x, f'_{args.cmb_file}.nii.gz'), file_list))
        else:
            cmb_file_list = []
        if args.dwi:
            dwi_file_list = list(map(lambda x: replace_suffix(x, f'_{args.dwi_file}.nii.gz'), file_list))
        else:
            dwi_file_list = []
        if args.wmh:
            wmh_file_list = list(map(lambda x: replace_suffix(x, f'_{args.wmh_file}.nii.gz'), file_list))
        else:
            wmh_file_list = []
        wm_file_list = list(map(lambda x: replace_suffix(x, f'_david.nii.gz'), file_list))

    if args.depth_number:
        depth_number = args.depth_number
    else:
        depth_number = 5

    resample_file_list = list(map(lambda x: replace_suffix(x, f'_resample.nii.gz'), file_list))
    synthseg_file_list = list(map(lambda x: replace_suffix(x, f'_synthseg.nii.gz'), file_list))
    synthseg33_file_list = list(map(lambda x: replace_suffix(x, f'_synthseg33.nii.gz'), file_list))

    cmb_file_list = list(map(lambda x: os.path.join(out_path, os.path.basename(x)), cmb_file_list))
    dwi_file_list = list(map(lambda x: os.path.join(out_path, os.path.basename(x)), dwi_file_list))
    wmh_file_list = list(map(lambda x: os.path.join(out_path, os.path.basename(x)), wmh_file_list))
    wm_file_list = list(map(lambda x: os.path.join(out_path, os.path.basename(x)), wm_file_list))
    # --------- 檢 參數 建立存檔名稱 end----------------
    synth_seg = SynthSeg()
    for i in range(len(file_list)):
        resample_one(file_list[i], resample_file_list[i])
        synth_seg.run(path_images=resample_file_list[i],
                      path_segmentations=synthseg_file_list[i],
                      path_segmentations33=synthseg33_file_list[i],
                      )
        synthseg_nii = nib.load(synthseg_file_list[i])
        synthseg_array = synthseg_nii.get_fdata()
        synthseg33_nii = nib.load(synthseg33_file_list[i])
        synthseg33_array = synthseg33_nii.get_fdata()
        seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(synthseg_array=synthseg_array,
                                                                        synthseg33=synthseg33_array,
                                                                        depth_number=depth_number)

        # out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
        # nib.save(out_nib, wm_file_list[i])
        #
        # cmb_array = CMBProcess.run(seg_array)
        # out_nib = nib.Nifti1Image(cmb_array, synthseg_nii.affine, synthseg_nii.header)
        # nib.save(out_nib, cmb_file_list[i])
        #
        # dwi_array = DWIProcess.run(seg_array)
        # out_nib = nib.Nifti1Image(dwi_array, synthseg_nii.affine, synthseg_nii.header)
        # nib.save(out_nib, dwi_file_list[i])
        #
        # wmh_array = run_wmh(synthseg_array=synthseg_array, synthseg_array_wm=synthseg_array_wm,
        #                     depth_number=depth_number)
        # out_nib = nib.Nifti1Image(wmh_array, synthseg_nii.affine, synthseg_nii.header)
        # nib.save(out_nib, wmh_file_list[i])
        if args.wm_file:
            out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
            nib.save(out_nib, wm_file_list[i])
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
            nib.save(out_nib, wmh_file_list[i])

        print(file_list[i], resample_file_list[i],
              synthseg_file_list[i],synthseg33_file_list[i],)


        # try:
        #     synth_seg.run(path_images=resample_file_list[i],
        #                   path_segmentations=synthseg_file_list[i],
        #                   path_segmentations33=synthseg33_file_list[i],
        #                   )
        #     synthseg_nii = nib.load(synthseg_file_list[i])
        #     synthseg_array = synthseg_nii.get_fdata()
        #     synthseg33_nii = nib.load(synthseg33_file_list[i])
        #     synthseg33_array = synthseg33_nii.get_fdata()
        #     seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(synthseg_array=synthseg_array,
        #                                                                     synthseg33=synthseg33_array,
        #                                                                     depth_number=depth_number)
        #     if args.all:
        #         out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
        #         nib.save(out_nib, wm_file_list[i])
        #
        #         cmb_array = CMBProcess.run(seg_array)
        #         out_nib = nib.Nifti1Image(cmb_array, synthseg_nii.affine, synhseg_nii.header)
        #         nib.save(out_nib, cmb_file_list[i])
        #
        #         dwi_array = DWIProcess.run(seg_array)
        #         out_nib = nib.Nifti1Image(dwi_array, synthseg_nii.affine, synthseg_nii.header)
        #         nib.save(out_nib, dwi_file_list[i])
        #
        #         wmh_array = run_wmh(synthseg_array=synthseg_array, synthseg_array_wm=synthseg_array_wm,
        #                             depth_number=depth_number)
        #         out_nib = nib.Nifti1Image(wmh_array, synthseg_nii.affine, synthseg_nii.header)
        #         nib.save(out_nib, wmh_file_list[i])
        #     else:
        #         if args.wm_file:
        #             out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
        #             nib.save(out_nib, wm_file_list[i])
        #         if args.cmb:
        #             cmb_array = CMBProcess.run(seg_array)
        #             out_nib = nib.Nifti1Image(cmb_array, synthseg_nii.affine, synthseg_nii.header)
        #             nib.save(out_nib, cmb_file_list[i])
        #         if args.dwi:
        #             dwi_array = DWIProcess.run(seg_array)
        #             out_nib = nib.Nifti1Image(dwi_array, synthseg_nii.affine, synthseg_nii.header)
        #             nib.save(out_nib, dwi_file_list[i])
        #         if args.wmh:
        #             wmh_array = run_wmh(synthseg_array=synthseg_array, synthseg_array_wm=synthseg_array_wm,
        #                                 depth_number=depth_number)
        #             out_nib = nib.Nifti1Image(wmh_array, synthseg_nii.affine, synthseg_nii.header)
        #             nib.save(out_nib, wmh_file_list[i])
        # except:
        #     print(f'{synthseg_file_list[i]} is Error')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # args, unrecognized_args = parser.parse_known_args()

    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        default=r'D:\00_Chen\Task03_\VCI_out_4cases_nii_gz_T1',
                        help="input the (SHH seg)synthseg file path (nii , nii.gz) or input the folder path.\r\n"
                             "Example ： python utils_parcellation.py -i input_path ")
    parser.add_argument('--input_name', dest='input_name', type=str,
                        help="")
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default=r'D:\00_Chen\Task03_\VCI_out_4cases_nii_gz_T1',
                        help="output the result file , if None then output file to input parameter path.\r\n"
                             "Example ： python utils_parcellation.py -i input_path -o output_path")

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
    parser.add_argument('--WMHFile', '--wmhFile', dest='wmh_file', type=str, default='WMH',
                        help="WMH Mask file name，default is \'{output}_WMH.nii.gz\'")

    parser.add_argument('--depth_number', dest='depth_number', default=5, type=int, choices=[4, 5, 6, 7, 8, 9, 10]
                        , help="deep white matter parameter")

    args = parser.parse_args()
    print(args)
    main(args)

    # root_path = pathlib.Path(r'D:\00_Chen\Task03_\Study_VCI_out_4cases')
    # root_path = pathlib.Path(r'D:\00_Chen\Task06_SVD\Study_SVD_20190726_20220105_out_7nii')
    # root_path = pathlib.Path(r'D:\00_Chen\Task06_SVD\Study_SVD_20220916_20230718_out_8nii')
    #
    # print(root_path)
    # # t2_list = list(root_path.rglob('*T2_FLAIR*'))
    # # print(t2_list)
    # t1_list = list(root_path.rglob('*T1*'))
    # print('*'*100)
    # print(t1_list)
    # t1_list = list(root_path.rglob('*BRAVO*'))
    # print('*' * 100)
    # print(t1_list)
    # t2_list = list(root_path.rglob('*FLAIR*'))
    # print('*' * 100)
    # print(t2_list)
