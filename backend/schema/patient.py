import datetime
import typing
from typing import Union, Any, List, Optional
from uuid import UUID
from fastapi import Body
from fastapi_pagination import Page
from pydantic import BaseModel, Field, model_serializer
from .base import op_list,CustomParams


class PatientBase(BaseModel):
    patient_id: str
    gender: str
    birth_date: datetime.date


class PatientIn(PatientBase):
    orthanc_patient_ID : Optional[str] = Field(default=None)


class PatientOut(PatientBase):
    patient_uid: str

    @model_serializer(when_used='always')
    def sort_model(self) -> typing.Dict[str, Any]:
        return {
            "patient_uid": self.patient_uid,
            "patient_id": self.patient_id,
            "gender": self.gender,
            "birth_date": self.birth_date
        }


class PatientPostOut(BaseModel):
    size : int
    patient_out_list: List[PatientOut]


class PatientDeleteIn(BaseModel):
    patient_uid: Union[UUID, None] = None
    patient_id: Union[str, None] = None


class StudyOut(BaseModel):
    #study_uid: UUID
    study_date: datetime.date
    study_time: datetime.time
    study_description: str
    accession_number: str
    # text: str


class PatientStudyOut(PatientBase):
    study : List[StudyOut]


class FilterItemSchema(BaseModel):
    field: str = Body('gender')
    op: str = Body('eq')
    value: Any = Body('M')


class FilterSchema(BaseModel):
    filter_: List[FilterItemSchema] = Body(...)


field_model = dict(
    patient_uid='PatientModel',
    patient_id='PatientModel',
    gender='PatientModel',
    study_uid='StudyModel',
    study_date='StudyModel',
    study_time='StudyModel',
    study_description='StudyModel',
    accession_number='StudyModel')


class PatientGroupPage(Page):
    group_key          : typing.Dict[str, List[str]]
    op_list            : List[str] = op_list
    __params_type__ = CustomParams