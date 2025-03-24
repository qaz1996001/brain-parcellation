from pathlib import Path
from funboost.core.func_params_model import BaseJsonAbleModel


class ResampleTaskParams(BaseJsonAbleModel):
    file :Path
    resample_file :Path


class SynthsegTaskParams(ResampleTaskParams):
    synthseg_file   :Path
    synthseg33_file :Path


class ProcessSynthsegTaskParams(SynthsegTaskParams):
    depth_number : int = 5
    david_file   : Path
    wm_file      : Path