from pathlib import Path
from typing import List

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
    original_file       :Path
    resample_image_file :Path
    resample_seg_file   :Path


class SaveFileTaskParams(ProcessSynthsegTaskParams):
    save_mode : str
    save_file_path : Path


class PostProcessSynthsegTaskParams(BaseJsonAbleModel):
    save_mode : str
    cmb_file_list : List[Path]