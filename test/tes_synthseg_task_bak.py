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



import argparse
import enum
import pathlib
import re

import numpy as np
import pandas as  pd

from convert.config import T1SeriesRenameEnum,MRSeriesRenameEnum, T2SeriesRenameEnum
import orjson

class InferenceEnum(str,enum.Enum):
    CMB = 'CMB'
    DWI = 'DWI'
    WMH_PVS = 'WMH_PVS'
    Area = 'Area'
    Infarct= 'Infarct'
    WMH = 'WMH'
    Aneurysm = 'Aneurysm'
    AneurysmSynthSeg = 'AneurysmSynthSeg'
    # Lacune

model_mapping_series_dict = {InferenceEnum.Area:[#{T1SeriesRenameEnum.T1BRAVO_AXI,},
                                                 #{T1SeriesRenameEnum.T1BRAVO_SAG,},
                                                 # {T1SeriesRenameEnum.T1BRAVO_COR,},
                                                 # {T1SeriesRenameEnum.T1FLAIR_AXI,},
                                                 # {T1SeriesRenameEnum.T1FLAIR_SAG,},
                                                 #{T1SeriesRenameEnum.T1FLAIR_COR,}
                                                 [T1SeriesRenameEnum.T1BRAVO_AXI,],
                                                 [T1SeriesRenameEnum.T1BRAVO_SAG,],
                                                 [T1SeriesRenameEnum.T1BRAVO_COR,],
                                                 [T1SeriesRenameEnum.T1FLAIR_AXI,],
                                                 [T1SeriesRenameEnum.T1FLAIR_SAG,],
                                                 [T1SeriesRenameEnum.T1FLAIR_COR,],
                                                 ],
                             InferenceEnum.DWI:[ #[MRSeriesRenameEnum.DWI0, T1SeriesRenameEnum.T1BRAVO_AXI,],
                                                 # [MRSeriesRenameEnum.DWI0, T1SeriesRenameEnum.T1FLAIR_AXI,],
                                                   [MRSeriesRenameEnum.DWI0]


                                                 ],
                             InferenceEnum.WMH_PVS:[[T2SeriesRenameEnum.T2FLAIR_AXI,]],
                             InferenceEnum.CMB:[[MRSeriesRenameEnum.SWAN,T1SeriesRenameEnum.T1BRAVO_AXI],
                                                #Ax SWAN_resample_synthseg33_from_Sag_FSPGR_BRAVO_resample_synthseg33.nii.gz
                                                [MRSeriesRenameEnum.SWAN,T1SeriesRenameEnum.T1FLAIR_AXI],
                                                ],
                             InferenceEnum.AneurysmSynthSeg: [[MRSeriesRenameEnum.MRA_BRAIN]],
                             InferenceEnum.Infarct:[[MRSeriesRenameEnum.DWI0,
                                                     MRSeriesRenameEnum.DWI1000,
                                                     MRSeriesRenameEnum.ADC],],
                             InferenceEnum.WMH :[[T2SeriesRenameEnum.T2FLAIR_AXI,]],

                             InferenceEnum.Aneurysm: [[MRSeriesRenameEnum.MRA_BRAIN]]
}

# C:\Users\tmu3090\Desktop\Task\dicom2nii\src\mapping.json
# E:\PC_3090\data\output\PSCL_MRI\08292236_20160707_MR_E42557741501
# "E:\\PC_3090\\data\\rename_nifti\\PSCL_MRI\\00003092_20201007_MR_20910070157\\synthseg_DWI0_original_DWI.json",
# intput
# Infarct":["E:\\PC_3090\\data\\rename_nifti\\PSCL_MRI\\00003092_20201007_MR_20910070157\\ADC.nii.gz",
# "E:\\PC_3090\\data\\rename_nifti\\PSCL_MRI\\00003092_20201007_MR_20910070157\\DWI0.nii.gz",
# "E:\\PC_3090\\data\\rename_nifti\\PSCL_MRI\\00003092_20201007_MR_20910070157\\DWI1000.nii.gz",
# "E:\\PC_3090\\data\\rename_nifti\\PSCL_MRI\\00003092_20201007_MR_20910070157\\synthseg_DWI0_original_DWI.nii.gz",
# ]
# "WMH": ["E:\\PC_3090\\data\\rename_nifti\\PSCL_MRI\\00003092_20201007_MR_20910070157\\T2FLAIR_AXI.nii.gz",
#         "E:\\PC_3090\\data\\rename_nifti\\PSCL_MRI\\00003092_20201007_MR_20910070157\\synthseg_T2FLAIR_AXI_original_WMHPVS.nii.gz",
# ],
# output
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_Infarct.nii.gz
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_Infarct_ADCth.nii.gz
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_Infarct_synthseg.nii.gz
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_Infarct.json
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_WMH.nii.gz
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_WMH_synthseg.nii.gz
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_WMH.json
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_Aneurysm.nii.gz
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Prob_Aneurysm.nii.gz
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_Vessel.nii.gz
# output_path /mnt/e/PC_3090/data/output/PSCL_MRI/00003092_20201007_MR_20910070157/Pred_Aneurysm.json


study_id_pattern = re.compile('^[0-9]{8}_[0-9]{8}_(MR|CT|CR|PR).*$', re.IGNORECASE)
# 00003092_20201007_MR_20910070157
def check_study_id(intput_path :pathlib.Path)->bool:
    global study_id_pattern
    if intput_path.is_dir():
        result = study_id_pattern.match(intput_path.name)
        if result is not None:
            return True
    return False


def check_study_mapping_inference(study_path:pathlib.Path):
    file_list = sorted(study_path.iterdir())
    if any(filter(lambda x: x.name.endswith('nii.gz') or x.name.endswith('nii') , file_list)):
        df_file = pd.DataFrame(file_list,columns=['file_path'])
        df_file['file_name'] = df_file['file_path'].map(lambda x: x.name.replace('.nii.gz', ''))
        model_mapping_dict = {}
        for model_name, model_mapping_series_list in model_mapping_series_dict.items():
            for mapping_series in model_mapping_series_list:
                mapping_series_str = list(map(lambda x:x.value,mapping_series))
                result = np.intersect1d(df_file['file_name'], mapping_series_str,return_indices=True)

                if result[0].shape[0] >= len(mapping_series_str):
                    df_result = df_file.iloc()[result[1]]
                    file_path = list(map(lambda x:str(x),df_result['file_path'].to_list()))
                    model_mapping_dict.update({model_name.value:file_path})
                    break
        return {study_path.name:model_mapping_dict}



def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dicom', dest='input_dicom', type=str,
    #                     help="input the rename dicom folder.\r\n")
    parser.add_argument('--input_nifti', dest='input_nifti', type=str,
                        help="input the rename nifti folder.\r\n")

    return parser.parse_args()


def main(args):
    parser = argparse.ArgumentParser()
    pass


if __name__ == '__main__':
    input_dir = pathlib.Path(r'E:\PC_3090\data\rename_nifti\PSCL_MRI')
    input_dir_list = sorted(input_dir.iterdir())
    study_list = list(filter(check_study_id, input_dir_list))
    mapping_inference_list = list(map(check_study_mapping_inference, study_list))
    with open('mapping.json', mode='wb+') as f:
        f.write(orjson.dumps(mapping_inference_list))

    parser = argparse.ArgumentParser()
    # python D:\00_Chen\Task04_git\code_ai\main.py -i D:\00_Chen\Task04_git\data --input_name BRAVO.nii
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data --input_name BRAVO.nii -o /mnt/d/00_Chen/Task04_git/data_0106
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data --input_name BRAVO.nii -o /mnt/d/00_Chen/Task04_git/data_0106
    # python D:\00_Chen\Task04_git\code_ai\main.py -i D:\00_Chen\Task04_git\data --input_name SWAN.nii --template_name BRAVO.nii --CMB True
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data --input_name BRAVO.nii -o /mnt/d/00_Chen/Task04_git/data_0106

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
    # workflow_group = build_celery_workflow(args, file_list)
    # result = workflow_group.apply_async()
    # print('result',result)
    # result.get()
    # print('result.get()',result.get())

