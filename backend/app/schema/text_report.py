from typing import List
from fastapi_pagination import Page
from pydantic import BaseModel
from .base import CustomParams

raw_field_model = dict(
    text= 'TextReportRawModel',
    accession_number = 'TextReportRawModel'
)


class TextReportOut(BaseModel):
    patient_id: str
    gender: str
    # age: str
    study_description: str
    accession_number: str
    text: str
    impression: str


class TextReportOut2(BaseModel):
    accession_number: str
    text: str
    impression: str


class TextReportPage(Page):
    op_list            : List[str] = ["like","==","!=","regexp",]
    __params_type__ = CustomParams