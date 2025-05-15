import uuid
import json
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import mapped_column,Mapped
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def gen_id():
    return str(uuid.uuid4().hex)


class FunboostConsumeResult(Base):
    __tablename__ = 'funboost_consume_results'

    _id = Column(String, primary_key=True)
    function = Column(String)
    host_name = Column(String)
    host_process = Column(String)
    insert_minutes = Column(String)
    insert_time = Column(DateTime)
    insert_time_str = Column(String)
    msg_dict = Column(Text)  # 存儲為 JSON 字串
    params = Column(Text)  # 存儲為 JSON 字串
    params_str = Column(String)
    process_id = Column(Integer)
    publish_time = Column(Float)
    publish_time_str = Column(String)
    queue_name = Column(String)
    result = Column(Text)
    run_times = Column(Integer)
    script_name = Column(String)
    script_name_long = Column(String)
    success = Column(Boolean)  # SQLite INTEGER 轉為 Python Boolean
    task_id = Column(String)
    thread_id = Column(Integer)
    time_cost = Column(Float)
    time_end = Column(Float)
    time_start = Column(Float)
    total_thread = Column(Integer)
    utime = Column(String)
    exception = Column(Text)
    rpc_result_expire_seconds = Column(Integer)
    run_status = Column(String)
    # JSON 處理的 helper 方法
    @property
    def msg_dict_obj(self):
        """將 JSON 字串轉換為 Python 對象"""
        if self.msg_dict:
            return json.loads(self.msg_dict)
        return None

    @msg_dict_obj.setter
    def msg_dict_obj(self, value):
        """將 Python 對象轉換為 JSON 字串"""
        if value is not None:
            self.msg_dict = json.dumps(value)
        else:
            self.msg_dict = None

    @property
    def params_obj(self):
        """將 JSON 字串轉換為 Python 對象"""
        if self.params:
            return json.loads(self.params)
        return None

    @params_obj.setter
    def params_obj(self, value):
        """將 Python 對象轉換為 JSON 字串"""
        if value is not None:
            self.params = json.dumps(value)
        else:
            self.params = None

    def __repr__(self):
        return f"<FunboostConsumeResult(id='{self._id}', function='{self.function}', task_id='{self.task_id}')>"


class RawDicomToNiiInference(Base):
    __tablename__ = 'raw_dicom_to_inference'
    _id               :Mapped[str] = Column(String,default=gen_id, primary_key=True)
    name              :Mapped[str] = Column(String)
    sub_dir           :Mapped[Optional[str]] = Column(String,nullable=True)
    output_dicom_path :Mapped[Optional[str]] = Column(String,nullable=True)
    output_nifti_path :Mapped[Optional[str]] = Column(String,nullable=True)
    created_time      :Mapped[str] = Column(String)
