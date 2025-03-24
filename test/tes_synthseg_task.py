import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
import argparse
import enum
import os
import re
from typing import List, Any, Dict, Optional

import pathlib
import numpy as np
import pandas as pd
from celery import Celery
import orjson
from pydantic import BaseModel, Field

from code_ai.dicom2nii.convert.config import T1SeriesRenameEnum, MRSeriesRenameEnum, T2SeriesRenameEnum


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
    return [
        output_dir.joinpath(x.parent.name, replace_suffix(f'{x.name}', suffix)) if output_dir else x.parent.joinpath(
            replace_suffix(x.name, suffix)) for x in file_list]


def replace_suffix(filename, new_suffix):
    pattern = r'\.nii\.gz$|\.nii$'
    return re.sub(pattern, new_suffix, filename)


class InferenceEnum(str, enum.Enum):
    CMB = 'CMB'
    DWI = 'DWI'
    WMH_PVS = 'WMH_PVS'
    Area = 'Area'
    Infarct = 'Infarct'
    WMH = 'WMH'
    Aneurysm = 'Aneurysm'
    AneurysmSynthSeg = 'AneurysmSynthSeg'
    # Lacune


model_mapping_series_dict = {InferenceEnum.Area: [  #{T1SeriesRenameEnum.T1BRAVO_AXI,},
    #{T1SeriesRenameEnum.T1BRAVO_SAG,},
    # {T1SeriesRenameEnum.T1BRAVO_COR,},
    # {T1SeriesRenameEnum.T1FLAIR_AXI,},
    # {T1SeriesRenameEnum.T1FLAIR_SAG,},
    #{T1SeriesRenameEnum.T1FLAIR_COR,}
    [T1SeriesRenameEnum.T1BRAVO_AXI, ],
    [T1SeriesRenameEnum.T1BRAVO_SAG, ],
    [T1SeriesRenameEnum.T1BRAVO_COR, ],
    [T1SeriesRenameEnum.T1FLAIR_AXI, ],
    [T1SeriesRenameEnum.T1FLAIR_SAG, ],
    [T1SeriesRenameEnum.T1FLAIR_COR, ],
],
    InferenceEnum.DWI: [  #[MRSeriesRenameEnum.DWI0, T1SeriesRenameEnum.T1BRAVO_AXI,],
        # [MRSeriesRenameEnum.DWI0, T1SeriesRenameEnum.T1FLAIR_AXI,],
        [MRSeriesRenameEnum.DWI0]
    ],
    InferenceEnum.WMH_PVS: [[T2SeriesRenameEnum.T2FLAIR_AXI, ]],
    InferenceEnum.CMB: [[MRSeriesRenameEnum.SWAN, T1SeriesRenameEnum.T1BRAVO_AXI],
                        #Ax SWAN_resample_synthseg33_from_Sag_FSPGR_BRAVO_resample_synthseg33.nii.gz
                        [MRSeriesRenameEnum.SWAN, T1SeriesRenameEnum.T1FLAIR_AXI],
                        ],
    InferenceEnum.AneurysmSynthSeg: [[MRSeriesRenameEnum.MRA_BRAIN]],
    InferenceEnum.Infarct: [[MRSeriesRenameEnum.DWI0,
                             MRSeriesRenameEnum.DWI1000,
                             MRSeriesRenameEnum.ADC], ],
    InferenceEnum.WMH: [[T2SeriesRenameEnum.T2FLAIR_AXI, ]],

    InferenceEnum.Aneurysm: [[MRSeriesRenameEnum.MRA_BRAIN]]
}

study_id_pattern = re.compile('^[0-9]{8}_[0-9]{8}_(MR|CT|CR|PR).*$', re.IGNORECASE)


def check_study_id(intput_path: pathlib.Path) -> bool:
    global study_id_pattern
    if intput_path.is_dir():
        result = study_id_pattern.match(intput_path.name)
        if result is not None:
            return True
    return False


def check_study_mapping_inference(study_path: pathlib.Path):
    file_list = sorted(study_path.iterdir())
    if any(filter(lambda x: x.name.endswith('nii.gz') or x.name.endswith('nii'), file_list)):
        df_file = pd.DataFrame(file_list, columns=['file_path'])
        df_file['file_name'] = df_file['file_path'].map(lambda x: x.name.replace('.nii.gz', ''))
        model_mapping_dict = {}
        for model_name, model_mapping_series_list in model_mapping_series_dict.items():
            for mapping_series in model_mapping_series_list:
                mapping_series_str = list(map(lambda x: x.value, mapping_series))
                result = np.intersect1d(df_file['file_name'], mapping_series_str, return_indices=True)

                if result[0].shape[0] >= len(mapping_series_str):
                    df_result = df_file.iloc()[result[1]]
                    file_path = list(map(lambda x: str(x), df_result['file_path'].to_list()))
                    model_mapping_dict.update({model_name.value: file_path})
                    break
        return {study_path.name: model_mapping_dict}


class Result(BaseModel):
    output_file: List[str]


class Task(BaseModel):
    input_path_list: List[str] = Field(..., alias="intput_path_list")
    output_path: str
    result: Result


class Analysis(BaseModel):
    Area: Optional[Task] = None
    DWI: Optional[Task] = None
    WMH_PVS: Optional[Task] = None
    CMB: Optional[Task] = None
    AneurysmSynthSeg: Optional[Task] = None
    Infarct: Optional[Task] = None
    WMH: Optional[Task] = None
    Aneurysm: Optional[Task] = None


class Dataset(BaseModel):
    analyses: Dict[str, Analysis]


def generate_output_files(input_paths: List[str], task_name: str, base_output_path: str) -> List[str]:
    """
    Generate output file names based on input paths and task names.
    """
    output_files = []
    match task_name:
        case InferenceEnum.Area:
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split('.')[0]
                output_file = os.path.join(base_output_path, f"synthseg_{base_name}_original_synthseg33.nii.gz")
                output_files.append(output_file)
        case InferenceEnum.WMH_PVS:
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split('.')[0]
                output_file = os.path.join(base_output_path, f"synthseg_{base_name}_original_{task_name}.nii.gz")
                output_files.append(output_file)
        # synthseg_DWI0_original_DWI.nii.gz
        case InferenceEnum.DWI:
            output_files.append(os.path.join(base_output_path, f"synthseg_DWI0_original_{task_name}.nii.gz"))
        case InferenceEnum.CMB:
            pass
        case InferenceEnum.AneurysmSynthSeg:
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split('.')[0]
                output_file = os.path.join(base_output_path, f"synthseg_{base_name}_original_synthseg33.nii.gz")
                output_files.append(output_file)
            pass
        case InferenceEnum.Infarct:
            output_files.append(os.path.join(base_output_path, f"Pred_Infarct.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Infarct_ADCth.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Infarct_synthseg.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Infarct.json"))
        case InferenceEnum.WMH:
            output_files.append(os.path.join(base_output_path, f"Pred_WMH.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_WMH_synthseg.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_WMH.json"))
        case InferenceEnum.Aneurysm:
            output_files.append(os.path.join(base_output_path, f"Pred_Aneurysm.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Prob_Aneurysm.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Vessel.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Aneurysm.json"))

        case _:
            pass
    return output_files


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dicom', dest='input_dicom', type=str,
    #                     help="input the rename dicom folder.\r\n")
    parser.add_argument('-i', '--input_nifti', dest='input_nifti', type=str,
                        help="input the rename nifti folder.\r\n",
                        default='/mnt/d/00_Chen/Task04_git/data_0106/')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help="result output folder.\r\n",
                        default='/mnt/d/00_Chen/Task04_git/data_0106')
    return parser.parse_args()


def main(args):
    input_dir = pathlib.Path(args.input_nifti)
    input_dir_list = sorted(input_dir.iterdir())
    study_list = list(filter(check_study_id, input_dir_list))
    output_dir = pathlib.Path(args.output)
    # base_output_path = "/mnt/d/00_Chen/Task04_git/output"
    mapping_inference_list = list(map(check_study_mapping_inference, study_list))
    analyses = {}
    for mapping_inference in mapping_inference_list:

        study_id = mapping_inference.keys()
        model_dict_values = mapping_inference.values()
        for task_dict in model_dict_values:
            tasks = {}
            for model_name, input_paths in task_dict.items():
                task_output_files = generate_output_files(input_paths, model_name, str(output_dir))
                # if len(task_output_files) >
                tasks[model_name] = Task(
                    intput_path_list=input_paths,
                    output_path=base_output_path,
                    result=Result(output_file=task_output_files)
                )
        analyses[str(study_id)] = Analysis(**tasks)
    return Dataset(analyses=analyses).model_dump()



def build_Area(mode,file_dict):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cmb = False
    args.wmh = False
    args.dwi = False
    args.wm_file = False
    args.all = False
    args.depth_number = 5
    setattr(args, mode, True)
    output_path = pathlib.Path(file_dict['output_path'])
    args.intput_file_list = list(map(lambda x: pathlib.Path(x), file_dict['input_path_list']))
    args.resample_file_list = prepare_output_file_list(args.intput_file_list, '_resample.nii.gz', output_path)
    args.synthseg_file_list = prepare_output_file_list(args.resample_file_list, '_synthseg.nii.gz', output_path)
    args.synthseg33_file_list = prepare_output_file_list(args.resample_file_list, '_synthseg33.nii.gz', output_path)
    args.david_file_list = prepare_output_file_list(args.resample_file_list, '_david.nii.gz', output_path)
    args.wm_file_list = prepare_output_file_list(args.resample_file_list, '_wm.nii.gz', output_path)
    return args,args.intput_file_list


def get_synthseg_args_file(inference_name, file_dict):
    output_path = pathlib.Path(file_dict['output_path'])
    match inference_name:
        case InferenceEnum.Area:
            args,file_list = build_Area('wm_file',file_dict)
            return args, file_list
        case InferenceEnum.WMH_PVS:
            args,file_list = build_Area('wmh',file_dict)
            args.wmh_file_list = prepare_output_file_list(args.resample_file_list, '_WMHPVS.nii.gz', output_path)
            return args, file_list
        case InferenceEnum.DWI:
            args, file_list = build_Area('dwi', file_dict)
            args.dwi_file_list = prepare_output_file_list(args.resample_file_list, '_DWI.nii.gz', output_path)
            return args, file_list
        case InferenceEnum.AneurysmSynthSeg:
            args, file_list = build_Area('wm_file', file_dict)
            return args, file_list


app = Celery('tasks',
             broker='pyamqp://guest:guest@localhost:5672/celery',
             backend='redis://localhost:10079/1'
             )
app.config_from_object('code_ai.celery_config')

if __name__ == '__main__':
    # input_dir = pathlib.Path(r'E:\PC_3090\data\rename_nifti\PSCL_MRI')
    # input_dir_list = sorted(input_dir.iterdir())
    # study_list = list(filter(check_study_id, input_dir_list))
    # mapping_inference_list = list(map(check_study_mapping_inference, study_list))
    # with open('mapping.json', mode='wb+') as f:
    #     f.write(orjson.dumps(mapping_inference_list))

    # python D:\00_Chen\Task04_git\code_ai\main.py -i D:\00_Chen\Task04_git\data --input_name BRAVO.nii
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data --input_name BRAVO.nii -o /mnt/d/00_Chen/Task04_git/data_0106
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data --input_name BRAVO.nii -o /mnt/d/00_Chen/Task04_git/data_0106
    # python D:\00_Chen\Task04_git\code_ai\main.py -i D:\00_Chen\Task04_git\data --input_name SWAN.nii --template_name BRAVO.nii --CMB True
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data --input_name BRAVO.nii -o /mnt/d/00_Chen/Task04_git/data_0106
    # python /mnt/d/00_Chen/Task04_git/code_ai/main.py -i /mnt/d/00_Chen/Task04_git/data -o /mnt/d/00_Chen/Task04_git/data_0106
    args = parse_arguments()
    print(args,type(args))
    mapping_inference_data = main(args)

    # miss_inference = { InferenceEnum.WMH,InferenceEnum.CMB,InferenceEnum.AneurysmSynthSeg,
    #                    InferenceEnum.DWI,InferenceEnum.WMH_PVS}
    miss_inference = { InferenceEnum.CMB,InferenceEnum.AneurysmSynthSeg,}
    # miss_inference = {}
    for study_id, mapping_inference in mapping_inference_data['analyses'].items():
        for inference_name, file_dict in mapping_inference.items():
            if file_dict is None:
                continue
            if inference_name in miss_inference:
                continue

            match inference_name:
                case InferenceEnum.Area:
                    args,file_list = get_synthseg_args_file(inference_name, file_dict)
                    result = app.send_task('code_ai.task.task_synthseg.celery_workflow',args=(args,file_list),
                                           queue='default',
                                           routing_key='celery')
                case InferenceEnum.WMH_PVS:
                    args,file_list = get_synthseg_args_file(inference_name, file_dict)
                    result = app.send_task('code_ai.task.task_synthseg.celery_workflow',args=(args,file_list),
                                           queue='default',
                                           routing_key='celery')
                case InferenceEnum.DWI:
                    # DWI
                    args,file_list = get_synthseg_args_file(inference_name, file_dict)
                    result = app.send_task('code_ai.task.task_synthseg.celery_workflow',args=(args,file_list),
                                           queue='default',
                                           routing_key='celery')
                case InferenceEnum.CMB:
                    pass
                case InferenceEnum.AneurysmSynthSeg:
                    args, file_list = get_synthseg_args_file(inference_name, file_dict)
                    result = app.send_task('code_ai.task.task_synthseg.celery_workflow', args=(args, file_list),
                                           queue='default',
                                           routing_key='celery')
                case InferenceEnum.Infarct:
                    pass
                case InferenceEnum.WMH:
                    pass
                case InferenceEnum.Aneurysm:
                    pass
                case _:
                    pass
        # break



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


            match inference_name:
                case InferenceEnum.Area:
                    args,file_list = get_synthseg_args_file(inference_name, file_dict)
                    print(args,file_list)
                case InferenceEnum.WMH_PVS:
                    args,file_list = get_synthseg_args_file(inference_name, file_dict)

                case InferenceEnum.DWI:
                    # DWI
                    args,file_list = get_synthseg_args_file(inference_name, file_dict)

                case InferenceEnum.CMB:
                    pass
                case InferenceEnum.AneurysmSynthSeg:
                    args, file_list = get_synthseg_args_file(inference_name, file_dict)

                case InferenceEnum.Infarct:
                    pass
                case InferenceEnum.WMH:
                    pass
                case InferenceEnum.Aneurysm:
                    pass
                case _:
                    pass