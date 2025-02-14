import uuid
from typing import Any, List, Optional, Dict,Union
from pydantic import BaseModel


class ProjectStudyOutput(BaseModel):
    project_study_uid : Union[uuid.UUID,str]
    project_name      : Optional[str]
    patient_id        : Optional[str]
    gender            : Optional[str]
    study_date        : Optional[str]
    study_time        : Optional[str]
    study_description : Optional[str]
    accession_number  : Optional[str]
    extra_data        : Optional[Dict[str, Any]]
    class Config:
        extra = "ignore"



class ProjectStudyPost(BaseModel):
    project_uid    : Union[uuid.UUID,str]
    study_uid_list : List[Union[uuid.UUID,str]]
    class Config:
        extra = "ignore"


class ProjectStudyAccessionNumberPost(BaseModel):
    project_uid    : Union[uuid.UUID,str]
    accession_number_list : List[str]
    class Config:
        extra = "ignore"