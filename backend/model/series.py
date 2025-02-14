from typing import List
from sqlalchemy import Integer, String, DateTime, TIMESTAMP, Date, Time,Uuid,ForeignKey
from sqlalchemy.orm import Mapped, mapped_column,relationship
from ..core.db import Base 
from . import gen_id

class SeriesModel(Base):
    __tablename__ = 'series'
    uid                : Mapped[Uuid]      = mapped_column(Uuid, default=gen_id, primary_key=True)
    # study_uid          : Mapped[Uuid]      = mapped_column(Uuid,index=True)
    study_uid          : Mapped[Uuid] = mapped_column(Uuid,ForeignKey('study.uid'), index=True)
    study              : Mapped["StudyModel"] = relationship(back_populates='series', uselist=False)
    series_date        : Mapped[Date]      = mapped_column(Date)
    series_time        : Mapped[Time]      = mapped_column(Time)
    series_description : Mapped[str]       = mapped_column(String,index=True)
    modality           : Mapped[str]       = mapped_column(String,index=True)
    #series_type        : Mapped[str]       = mapped_column(String)
    orthanc_series_ID  : Mapped[str]       = mapped_column(String, nullable=True)
    created_at         : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    updated_at         : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    deleted_at         : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    project            : Mapped[List["ProjectModel"]] = relationship(secondary="project_series",
                                                                     back_populates="series")
    def to_dict(self):
        dict_ = {
            'uid'                : self.uid.hex,
            'study_uid'          : self.study_uid.hex,
            'series_date'        : str(self.series_date),
            'series_time'        : str(self.series_time),
            'series_description' : self.series_description,
            'modality'           : self.modality,
            'orthanc_series_ID'  : self.orthanc_series_ID if self.orthanc_series_ID else 'None',
            'created_at'         : str(self.created_at),
            'updated_at'         : str(self.updated_at),
            'deleted_at'         : str(self.deleted_at),
        }
        return dict_
