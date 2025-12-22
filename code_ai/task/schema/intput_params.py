from pathlib import Path
from typing import List, Optional, Dict, Any
from funboost.core.func_params_model import BaseJsonAbleModel
from pydantic import ConfigDict, field_serializer, Field, field_validator


class ResampleTaskParams(BaseJsonAbleModel):
    file: Path
    resample_file: Path
    model_config = ConfigDict(extra="allow")


class SynthsegTaskParams(ResampleTaskParams):
    synthseg_file: Path
    synthseg33_file: Path


class ProcessSynthsegTaskParams(SynthsegTaskParams):
    depth_number: int = 5
    david_file: Path
    wm_file: Path


class ResampleToOriginalTaskParams(BaseJsonAbleModel):
    original_file: Path
    resample_image_file: Path
    resample_seg_file: Path
    resample_seg_file_list: Optional[List[Path]] = None


class SaveFileTaskParams(ProcessSynthsegTaskParams):
    save_mode: str
    save_file_path: Path


class PostProcessSynthsegTaskParams(BaseJsonAbleModel):
    save_mode: str
    cmb_file_list: List[Path]
    model_config = ConfigDict(extra="allow")

    @field_serializer("cmb_file_list")
    def serialize_cmb_file_list(self, cmb_file_list: List[Path], _info):
        return list(map(lambda cmb_file: str(cmb_file), cmb_file_list))


# *************************************************************************** #
class Dicom2NiiParams(BaseJsonAbleModel):
    sub_dir: Optional[Path]
    output_dicom_path: Optional[Path]
    output_nifti_path: Optional[Path]


class Dicom2NiiFileParams(BaseJsonAbleModel):
    dicom_study_folder_path: Path
    output_nifti_path: Path


# *************************************************************************** #


class Dicom2NiiSeriesParams(Dicom2NiiParams):
    study_uid: Optional[str]
    series_uid: Optional[str]


class ProcessInstancesParams(BaseJsonAbleModel):
    instance: Path
    output_dicom_path: Path


class CallDcm2niixParams(BaseJsonAbleModel):
    output_series_file_path: Path
    output_series_path: Path
    series_path: Path


# *************************************************************************** #
class TaskInferenceParams(BaseJsonAbleModel):
    input_study_nifti_path: Path
    output_study_nifti_path: Path


# *************************************************************************** #


class StudyTaskInferenceParams(BaseJsonAbleModel):
    mapping_inference: Dict[str, Any]
    output_study_nifti_path: Path
    model_config = ConfigDict(extra="allow")


# *************************************************************************** #
# Pipeline Inference Task Parameters
# *************************************************************************** #


class InferenceTaskParams(BaseJsonAbleModel):
    """推論任務參數模型 - 用於 task_pipeline_inference"""
    nifti_study_path: str = Field(..., description="NIFTI Study 路徑")
    dicom_study_path: str = Field(..., description="DICOM Study 路徑")
    study_uid: Optional[str] = Field(None, description="Study UID")
    study_id: Optional[str] = Field(None, description="Study ID")
    
    @field_validator('nifti_study_path', 'dicom_study_path')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """驗證路徑格式（不強制要求路徑存在，因為可能是遠程路徑）"""
        if not v or not v.strip():
            raise ValueError("路徑不能為空")
        return v.strip()
    
    model_config = ConfigDict(extra="allow")


class SubprocessTaskParams(BaseJsonAbleModel):
    """子進程任務參數模型 - 用於 task_subprocess_inference"""
    cmd_str: str = Field(..., description="要執行的命令字串")
    
    @field_validator('cmd_str')
    @classmethod
    def validate_cmd_str(cls, v: str) -> str:
        """驗證命令字串"""
        if not v or not v.strip():
            raise ValueError("命令字串不能為空")
        return v.strip()
    
    model_config = ConfigDict(extra="allow")
