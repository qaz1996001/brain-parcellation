import datetime
from typing import List
from sqlalchemy import Integer, String, DateTime, UUID, TIMESTAMP, Date, Time, Uuid,ForeignKey
from sqlalchemy.orm import Mapped, mapped_column,relationship
from ..core.db import Base 
from . import gen_id


class StudyModel(Base):
    __tablename__ = 'study'
    uid               : Mapped[Uuid]      = mapped_column(Uuid, default=gen_id, primary_key=True)
    # patient_uid       : Mapped[Uuid]      = mapped_column(Uuid, index=True)
    patient_uid       : Mapped[Uuid]         = mapped_column(Uuid,ForeignKey('patient.uid'), index=True)
    patient           : Mapped["PatientModel"] = relationship(back_populates='study', uselist=False)
    study_date        : Mapped[datetime.date]      = mapped_column(Date, index=True)
    study_time        : Mapped[datetime.time]      = mapped_column(Time)
    study_description : Mapped[str]       = mapped_column(String)
    accession_number  : Mapped[str]       = mapped_column(String, index=True)
    series            : Mapped[List["SeriesModel"]] = relationship(back_populates="study")

    # text              : Mapped[List["TextReportModel"]] = relationship(back_populates="study")
    text              : Mapped["TextReportModel"] = relationship(back_populates="study", uselist=False)
    project           : Mapped[List["ProjectModel"]] = relationship(secondary="project_study",
                                                                    back_populates="study")
    project_associations : Mapped[List["ProjectStudyModel"]] = relationship(back_populates="study")
    orthanc_study_ID  : Mapped[str]       = mapped_column(String, nullable=True)
    created_at        : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    updated_at        : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    deleted_at        : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)

    def to_dict(self):
        dict_ = {
            'uid': self.uid.hex,
            'patient_uid': self.patient_uid.hex,
            'study_date': str(self.study_date),
            'study_time': str(self.study_time),
            'study_description': self.study_description,
            'accession_number': self.accession_number,
            'orthanc_study_ID': self.orthanc_study_ID if self.orthanc_study_ID else 'None',
            'created_at': str(self.created_at),
            'updated_at': str(self.updated_at),
            'deleted_at': str(self.deleted_at),
        }
        return dict_

    def to_dict_view(self):
        dict_ = {
            'study_date': str(self.study_date),
            'study_time': str(self.study_time),
            'study_description': self.study_description,
            'accession_number': self.accession_number,}
        return dict_

    def __repr__(self):
        return f'<StudyModel {self.uid.hex} {self.study_date}>'


class TextReportModel(Base):
    __tablename__ = 'text_report'
    uid              : Mapped[Uuid]      = mapped_column(Uuid, default=gen_id, primary_key=True)
    study_uid        : Mapped[Uuid]      = mapped_column(Uuid,ForeignKey('study.uid'), index=True)
    accession_number : Mapped[str]       = mapped_column(String, index=True)
    text             : Mapped[str]       = mapped_column(String)
    is_success       : Mapped[int]       = mapped_column(Integer, index=True)
    study            : Mapped["StudyModel"] = relationship(back_populates='text', uselist=False)
    created_at       : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    updated_at       : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    deleted_at       : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)

    def to_dict(self):
        dict_ = {
            'uid': self.uid.hex,
            'study_uid': self.patient_uid.hex,
            'text': self.patient_uid.hex,
            'is_success': self.patient_uid.hex,
            'created_at': str(self.created_at),
            'updated_at': str(self.updated_at),
            'deleted_at': str(self.deleted_at),
        }
        return dict_



class TextReportRawModel(Base):
    __tablename__ = 'text_report_raw'
    uid              : Mapped[Uuid]      = mapped_column(Uuid, default=gen_id, primary_key=True)
    accession_number : Mapped[str]       = mapped_column(String, index=True)
    text             : Mapped[str]       = mapped_column(String)
    created_at       : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    updated_at       : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    deleted_at       : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)

    def to_dict(self):
        dict_ = {
            'uid': self.uid.hex,
            'accession_number': self.accession_number,
            'text': self.text,
            'created_at': str(self.created_at),
            'updated_at': str(self.updated_at),
            'deleted_at': str(self.deleted_at),
        }
        return dict_