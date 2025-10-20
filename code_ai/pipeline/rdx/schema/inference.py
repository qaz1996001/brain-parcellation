from typing import  List,  Optional
from pydantic import Field, ConfigDict, BaseModel



class InferenceCompleteBase(BaseModel):
    model_config       = ConfigDict(from_attributes=True)
    study_instance_uid :str  = Field(...,serialization_alias="studyInstanceUid")
    series_instance_uid: str = Field(..., serialization_alias="seriesInstanceUid")