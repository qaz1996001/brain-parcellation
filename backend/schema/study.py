import datetime
from typing import Any, List, Dict
from fastapi_pagination import Page

from pydantic import BaseModel, Field
from .patient import PatientIn
from .base import op_list,CustomParams


class StudyPostIn(BaseModel):
    patient_id        : str
    study_date        : datetime.date
    study_time        : datetime.time
    study_description : str
    accession_number  : str


class StudyPatientPostIn(PatientIn):
    study_date        : datetime.date
    study_time        : datetime.time
    study_description : str
    accession_number  : str


class StudyOut(BaseModel):
    study_uid: str
    patient_uid: str
    patient_id: str
    gender: str
    study_date: datetime.date
    study_time: datetime.time
    study_description: str
    accession_number: str
    text: str
    impression: str


class FilterItemSchema(BaseModel):
    field :str = Field(default=None,examples = ['accession_number'])
    op    :str = Field(default=None,examples = ['eq'])
    value :Any = Field(default=None,examples = ['21301160043'])


class FilterSchema(BaseModel):
    filter_ :List[FilterItemSchema] = Field(...)


field_model = dict(
    study_uid='StudyModel',
    patient_uid='PatientModel',
    patient_id='PatientModel',
    gender='PatientModel',
    study_date='StudyModel',
    study_time='StudyModel',
    study_description='StudyModel',
    accession_number='StudyModel',
    series='SeriesModel',
    text= 'TextReportModel',
    series_description = 'SeriesModel',
)



class StudySeriesOut(BaseModel):
    study_uid: str
    patient_uid: str
    study_date: datetime.date
    study_time: datetime.time
    study_description: str
    accession_number: str
    series_description: List[str]
    # series_description: List[Dict[str, int]]

class StudySeriesTextOut(BaseModel):
    study_uid: str
    patient_uid: str
    study_date: datetime.date
    study_time: datetime.time
    study_description: str
    accession_number: str
    series_description: List[str]
    text: str

class StudySeriesOut2(BaseModel):
    patient_id: str
    gender: str
    age: str
    study_date: datetime.date
    study_time: datetime.time
    study_description: str
    accession_number: str
    # series_description: List[str]
    series_description: List[Dict[str, int]]


class StudySeriesPage(Page):
    series_description : List[str]





class StudySeriesGroupPage(Page):

    series_description : List[str]
    group_key          : Dict[str, List[str]]
    op_list            : List[str] = op_list
    # general_keys       : List[str]
    # structure_keys     : List[str]
    # special_keys       : List[str]
    # perfusion_keys     : List[str]
    # functional_keys    : List[str]
    __params_type__ = CustomParams

