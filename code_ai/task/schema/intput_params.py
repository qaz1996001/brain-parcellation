from pathlib import Path
from typing import List,Optional

from funboost.core.func_params_model import BaseJsonAbleModel
from pydantic import ConfigDict


class ResampleTaskParams(BaseJsonAbleModel):
    file :Path
    resample_file :Path
    model_config = ConfigDict(extra="allow")


class SynthsegTaskParams(ResampleTaskParams):
    synthseg_file   :Path
    synthseg33_file :Path


class ProcessSynthsegTaskParams(SynthsegTaskParams):
    depth_number : int = 5
    david_file   : Path
    wm_file      : Path


class ResampleToOriginalTaskParams(BaseJsonAbleModel):
    original_file          :Path
    resample_image_file    :Path
    resample_seg_file      :Path
    resample_seg_file_list :Optional[List[Path]] = None


class SaveFileTaskParams(ProcessSynthsegTaskParams):
    save_mode : str
    save_file_path : Path


class PostProcessSynthsegTaskParams(SaveFileTaskParams):
    pass
    # cmb_file_list : List[Path]


class BuildSynthsegTaskParams(BaseJsonAbleModel):
    inference_name : str
    cmb_file_list : List[Path]


# *************************************************************************** #
class Dicom2NiiParams(BaseJsonAbleModel):
    sub_dir           : Optional[Path]
    output_dicom_path : Optional[Path]
    output_nifti_path : Optional[Path]

class Dicom2NiiFileParams(BaseJsonAbleModel):
    dicom_study_folder_path : Path
    output_nifti_path       : Path


class ProcessInstancesParams(BaseJsonAbleModel):
    instance          : Path
    output_dicom_path : Path


class CallDcm2niixParams(BaseJsonAbleModel):
    output_series_file_path  : Path
    output_series_path       : Path
    series_path              : Path


# *************************************************************************** #
class TaskInferenceParams(BaseJsonAbleModel):
    input_study_nifti_path  : Path
    output_study_nifti_path : Path