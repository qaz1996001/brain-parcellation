from pathlib import Path
from typing import List, Optional, Dict
import os

from funboost.core.func_params_model import BaseJsonAbleModel
from pydantic import ConfigDict, field_serializer, model_validator


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


class PostProcessSynthsegTaskParams(BaseJsonAbleModel):
    save_mode: str
    cmb_file_list : List[Path]
    model_config = ConfigDict(extra="allow")

    @field_serializer('cmb_file_list')
    def serialize_cmb_file_list(self, cmb_file_list: List[Path], _info):
        return list(map(lambda cmb_file: str(cmb_file), cmb_file_list))


# *************************************************************************** #
class Dicom2NiiParams(BaseJsonAbleModel):
    sub_dir             : Optional[Path]
    output_dicom_path   : Optional[Path]
    output_nifti_path   : Optional[Path]
    upload_data_api_url : Optional[str] = None  # API URL for data upload callbacks

    @model_validator(mode='after')
    def validate_upload_url(self):
        """Fallback to environment variable if not provided, with URL format validation."""
        if self.upload_data_api_url is None:
            self.upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
            if self.upload_data_api_url is None:
                raise ValueError(
                    "upload_data_api_url must be provided or UPLOAD_DATA_API_URL environment variable must be set"
                )

        # Validate URL format
        if not self.upload_data_api_url.startswith(('http://', 'https://')):
            raise ValueError(
                f"upload_data_api_url must be a valid HTTP/HTTPS URL, got: {self.upload_data_api_url}"
            )

        return self


class Dicom2NiiFileParams(BaseJsonAbleModel):

    dicom_study_folder_path : Path
    output_nifti_path       : Path
# *************************************************************************** #


class Dicom2NiiSeriesParams(Dicom2NiiParams):
    study_uid  : Optional[str]
    series_uid : Optional[str]


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


# *************************************************************************** #



class StudyTaskInferenceParams(BaseJsonAbleModel):
    mapping_inference       : Dict[str, any]
    output_study_nifti_path : Path
    model_config = ConfigDict(extra="allow")