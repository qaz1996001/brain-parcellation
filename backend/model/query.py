from sqlalchemy import Integer, String, TIMESTAMP, Date, Time,Uuid, JSON
from sqlalchemy.orm import Mapped, mapped_column
from ..core.db import Base 
from . import gen_id


class ListPatientModel(Base):
    __tablename__ = 'list_patient'
    patient_uid : Mapped[Uuid] = mapped_column(Uuid, default=gen_id, primary_key=True)
    patient_id  : Mapped[str]  = mapped_column(String,index=True)
    gender      : Mapped[str]  = mapped_column(String,index=True)
    age         : Mapped[int]  = mapped_column(Integer)

    def to_dict(self):
        dict_ = {
            'patient_uid' : self.patient_uid.hex,
            'patient_id'  : self.patient_id,
            'gender'      : self.gender,
            'age'         : self.age,
        }
        return dict_



class ListStudyModel(Base):
    __tablename__ = 'list_study'
    study_uid          : Mapped[Uuid] = mapped_column(Uuid, default=gen_id, primary_key=True)
    patient_id         : Mapped[str]  = mapped_column(String,index=True)
    gender             : Mapped[str]  = mapped_column(String,index=True)
    age                : Mapped[int]  = mapped_column(Integer)
    study_date        : Mapped[Date]  = mapped_column(Date, index=True)
    study_time        : Mapped[Time]  = mapped_column(Time)
    study_description  : Mapped[str]  = mapped_column(String)
    accession_number   : Mapped[str]  = mapped_column(String, index=True)
    series_description : Mapped[JSON] = mapped_column(JSON)

    def to_dict(self):
        dict_ = {
            'study_uid'          : self.study_uid.hex,
            'patient_id'         : self.patient_id,
            'gender'             : self.gender,
            'age'                : self.age,
            'study_date'         : self.study_date,
            'study_time'         : self.study_time,
            'study_description'  : self.study_description,
            'accession_number'   : self.accession_number,
            'series_description' : self.series_description,
        }
        return dict_


class ListStudyTextReportModel(Base):
    __tablename__ = 'list_study_text_report'
    study_uid         : Mapped[Uuid] = mapped_column(Uuid, default=gen_id, primary_key=True)
    patient_id        : Mapped[str]  = mapped_column(String,index=True)
    gender            : Mapped[str]  = mapped_column(String,index=True)
    age               : Mapped[int]  = mapped_column(Integer)
    study_description : Mapped[str]  = mapped_column(String)
    accession_number  : Mapped[str]  = mapped_column(String, index=True)
    is_success        : Mapped[int]  = mapped_column(Integer, index=True)
    text              : Mapped[str]  = mapped_column(String)


    def to_dict(self):
        dict_ = {
            'study_uid'         : self.study_uid.hex,
            'patient_id'        : self.patient_id,
            'gender'            : self.gender,
            'age'               : self.age,
            'study_description' : self.study_description,
            'accession_number'  : self.accession_number,
            'is_success'        : self.is_success,
            'text'              : self.text,
        }
        return dict_


class ListProjectStudyModel(Base):
    __tablename__ = 'list_project_study'
    uid                : Mapped[Uuid] = mapped_column(Uuid, default=gen_id, primary_key=True)
    project_uid       : Mapped[Uuid] = mapped_column(Uuid)
    project_name      : Mapped[str]  = mapped_column(String,index=True)
    patient_id         : Mapped[str]  = mapped_column(String)
    gender             : Mapped[str]  = mapped_column(String,index=True)
    age                : Mapped[int]  = mapped_column(Integer)
    accession_number   : Mapped[str]  = mapped_column(String, index=True)
    study_date         : Mapped[Date] = mapped_column(Date, index=True)
    study_description  : Mapped[str]  = mapped_column(String)
    modality           : Mapped[str] = mapped_column(String, index=True)
    series_description : Mapped[JSON] = mapped_column(JSON)

    def to_dict(self):
        dict_ = {
            'project_uid'       : self.project_uid.hex,
            'project_name'      : self.project_name,
            'patient_id'         : self.patient_id,
            'gender'             : self.gender,
            'age'                : self.age,
            'accession_number'   : self.accession_number,
            'study_date'         : self.study_date,
            'study_description'  : self.study_description,
            'series_description' : self.series_description,
        }
        return dict_
