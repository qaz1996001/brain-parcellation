import datetime
from typing import Optional
from pydantic import BaseModel


class StudyOutput(BaseModel):
    study_uid: str
    patient_uid: str
    patient_id: str
    study_date: datetime.date
    study_time: datetime.time
    study_description: str
    accession_number: str
    text: str

    class Config:
        extra = "ignore"


class ProjectInput(BaseModel):
    name : Optional[str]

