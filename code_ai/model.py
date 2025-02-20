import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import String, TIMESTAMP, Uuid
from sqlalchemy.orm import Mapped, mapped_column

# 創建SQLAlchemy的一個class，然後在其它地方使用
Base = declarative_base()


class TaskModel(Base):
    __tablename__ = 'task'
    task_id            : Mapped[Uuid]      = mapped_column(Uuid, primary_key=True)
    name               : Mapped[String]    = mapped_column(String, nullable=True)
    args               : Mapped[String]    = mapped_column(String, nullable=True)
    status             : Mapped[String]    = mapped_column(String, nullable=True)
    error_massage      : Mapped[String]    = mapped_column(String, nullable=True)
    start_time_at      : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP,default=datetime.datetime.now, nullable=True)
    end_time_at        : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)
    exec_time_at       : Mapped[TIMESTAMP] = mapped_column(TIMESTAMP, nullable=True)


    def to_dict(self):
        dict_ = {
            'task_id'        : self.task_id.hex,
            'name'           : self.name,
            'status'         : self.status,
            'start_time_at'  : str(self.start_time_at),
            'end_time_at'    : str(self.end_time_at),
            'exec_time_at'   : str(self.exec_time_at),
        }
        return dict_