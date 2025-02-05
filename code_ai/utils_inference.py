import argparse
import enum
import os
import pathlib
import re
from typing import List,  Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from code_ai.dicom2nii.convert.config import T1SeriesRenameEnum, MRSeriesRenameEnum, T2SeriesRenameEnum



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


class InferenceEnum(str, enum.Enum):
    SynthSeg = 'SynthSeg'
    Area = 'Area'

    CMB = 'CMB'
    CMBSynthSeg = 'CMBSynthSeg'

    DWI = 'DWI'
    Infarct = 'Infarct'

    WMH = 'WMH'
    WMH_PVS = 'WMH_PVS'
    # Lacune
    Aneurysm = 'Aneurysm'
    AneurysmSynthSeg = 'AneurysmSynthSeg'



model_mapping_series_dict = {
    InferenceEnum.Area: [[T1SeriesRenameEnum.T1BRAVO_AXI, ],
                         [T1SeriesRenameEnum.T1BRAVO_SAG, ],
                         [T1SeriesRenameEnum.T1BRAVO_COR, ],
                         [T1SeriesRenameEnum.T1FLAIR_AXI, ],
                         [T1SeriesRenameEnum.T1FLAIR_SAG, ],
                         [T1SeriesRenameEnum.T1FLAIR_COR, ],],
    InferenceEnum.DWI: [
        #[MRSeriesRenameEnum.DWI0, T1SeriesRenameEnum.T1BRAVO_AXI,],
        # [MRSeriesRenameEnum.DWI0, T1SeriesRenameEnum.T1FLAIR_AXI,],
        [MRSeriesRenameEnum.DWI0]
    ],
    InferenceEnum.WMH_PVS: [[T2SeriesRenameEnum.T2FLAIR_AXI, ]],

    #Ax SWAN_resample_synthseg33_from_Sag_FSPGR_BRAVO_resample_synthseg33.nii.gz
    InferenceEnum.CMB: [[MRSeriesRenameEnum.SWAN, T1SeriesRenameEnum.T1BRAVO_AXI],
                        [MRSeriesRenameEnum.SWAN, T1SeriesRenameEnum.T1FLAIR_AXI],
                        ],
    # InferenceEnum.CMBSynthSeg
    InferenceEnum.AneurysmSynthSeg: [[MRSeriesRenameEnum.MRA_BRAIN]],
    InferenceEnum.Infarct: [[MRSeriesRenameEnum.DWI0,MRSeriesRenameEnum.DWI1000,MRSeriesRenameEnum.ADC],
                            ],
    InferenceEnum.WMH: [[T2SeriesRenameEnum.T2FLAIR_AXI,
                         ]],

    InferenceEnum.Aneurysm: [[MRSeriesRenameEnum.MRA_BRAIN,
                              ]]
}


study_id_pattern = re.compile('^[0-9]{8}_[0-9]{8}_(MR|CT|CR|PR).*$', re.IGNORECASE)


def get_file_list(input_path:pathlib.Path, suffixes:str, filter_name=None) -> List[pathlib.Path]:
    if any(suffix in input_path.suffixes for suffix in suffixes):
        file_list = [input_path]
    else:
        file_list = sorted(list(input_path.rglob('*.nii*')))
    if filter_name:
        file_list = [f for f in file_list if filter_name in f.name]
    return file_list


def prepare_output_file_list(file_list:List[pathlib.Path],
                             suffix:str,
                             output_dir :Optional[pathlib.Path] =None) -> List[pathlib.Path]:
    return [
        output_dir.joinpath(x.parent.name, replace_suffix(f'{x.name}', suffix)) if output_dir else x.parent.joinpath(
            replace_suffix(x.name, suffix)) for x in file_list]


def replace_suffix(filename:str, new_suffix:str):
    pattern = r'\.nii\.gz$|\.nii$'
    return re.sub(pattern, new_suffix, filename)


def check_study_mapping_inference(study_path: pathlib.Path) -> Dict[str, Dict[str, str]]:
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


def check_study_id(intput_path: pathlib.Path) -> bool:
    global study_id_pattern
    if intput_path.is_dir():
        result = study_id_pattern.match(intput_path.name)
        if result is not None:
            return True
    return False


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
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split('.')[0]
                output_file = os.path.join(base_output_path, f"synthseg_{base_name}_original_{task_name}.nii.gz")
                output_files.append(output_file)
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



def build_Area(mode,file_dict) -> Tuple :
    """
    Build Area model based on input files.

    """
    parser = argparse.ArgumentParser(prog='build_Area')
    args = parser.parse_known_args()[0]
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


def get_synthseg_args_file(inference_name, file_dict) -> Tuple :
    """
        args
        args.cmb = False
        args.wmh = False
        args.dwi = False
        args.wm_file = False
        args.to_original_mode
        args.all = False
        args.depth_number = 5
        args.intput_file_list
        args.resample_file_list
        args.synthseg_file_list
        args.synthseg33_file_list
        args.david_file_list
        args.wm_file_list
    """
    output_path = pathlib.Path(file_dict['output_path'])
    match inference_name:
        case InferenceEnum.WMH_PVS:
            args,file_list = build_Area('wmh',file_dict)
            args.wmh_file_list = prepare_output_file_list(args.resample_file_list, '_WMHPVS.nii.gz', output_path)
            return args, file_list
        case InferenceEnum.DWI:
            args, file_list = build_Area('dwi', file_dict)
            args.dwi_file_list = prepare_output_file_list(args.resample_file_list, '_DWI.nii.gz', output_path)
            return args, file_list
        case InferenceEnum.CMB:
            args, file_list = build_Area('cmb', file_dict)
            args.cmb_file_list = prepare_output_file_list(args.resample_file_list, '_CMB.nii.gz', output_path)
            return args, file_list
        case InferenceEnum.Area:
            args,file_list = build_Area('wm_file',file_dict)
            return args, file_list
        case InferenceEnum.AneurysmSynthSeg:
            args, file_list = build_Area('wm_file', file_dict)
            return args, file_list




