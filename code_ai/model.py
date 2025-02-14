import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import List
from sqlalchemy import String, TIMESTAMP, Uuid, DOUBLE
from sqlalchemy.orm import Mapped, mapped_column, relationship

# 用create_engine對這個URL_DATABASE建立一個引擎
engine = create_engine('postgresql+psycopg2://postgres_n:postgres_p@127.0.0.1:15433/db_name')
# 使用sessionmaker來與資料庫建立一個對話，記得要bind=engine，這才會讓專案和資料庫連結
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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