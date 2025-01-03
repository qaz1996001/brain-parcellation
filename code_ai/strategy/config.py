import enum
import enum
from typing import Any, Dict, Union, Optional, List
from pydantic import BaseModel,Field
from fastapi import File,UploadFile


class InferenceEnum(str,enum.Enum):
    CMB = 'CMB'
    DWI = 'DWI'
    WMH = 'WMH'


class RequestIn(BaseModel):
    input_file     : Union[File,UploadFile] = File(...)
    template_file  : Optional[Union[File,UploadFile]] = None
    depth_number   : int = Field(5,ge=4,le=10, description="Deep white matter parameter.")
    inference_mode : Union[InferenceEnum] = InferenceEnum.CMB

    class Config:
        arbitrary_types_allowed = True