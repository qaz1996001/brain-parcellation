from sqlalchemy import Integer, String, TIMESTAMP, Uuid,ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from ..core.db import Base
from . import gen_id


class FileModel(Base):
    __tablename__ = 'file'
    uid           : Mapped[Uuid]      = mapped_column(Uuid, default=gen_id, primary_key=True)
    content_uid   : Mapped[Uuid]      = mapped_column(Uuid,nullable=True)
    file_name     : Mapped[str]       = mapped_column(String, default='')
    file_size     : Mapped[int]       = mapped_column(Integer)
    file_datetime : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP)
    file_type     : Mapped[str]       = mapped_column(String, nullable=True)
    file_status   : Mapped[str]       = mapped_column(String, nullable=True)
    file_url      : Mapped[str]       = mapped_column(String)

    def to_dict(self):
        dict_ = {
            'uid'           : self.uid.hex,
            'series_uid'    : self.series_uid.hex,
            'file_name'     : self.file_name,
            'file_size'     : self.file_size,
            'file_datetime' : str(self.file_datetime),
            'file_type'     : self.file_type,
            'file_status'   : self.file_status,
            'file_url'      : self.file_url,
        }
        return dict_