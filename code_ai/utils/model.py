from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Application-wide SQLAlchemy declarative base."""


def gen_id() -> str:
    """Generate a deterministic UUID hex string."""
    return uuid.uuid4().hex


class FunboostConsumeResult(Base):
    """ORM model that mirrors funboost's consume result table."""

    __tablename__ = "funboost_consume_results"

    _id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_id)
    function: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    host_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    host_process: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    insert_minutes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    insert_time: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    insert_time_str: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    msg_dict: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    params: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    params_str: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    process_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    publish_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    publish_time_str: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    queue_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    run_times: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    script_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    script_name_long: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    success: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    task_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    thread_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    time_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    time_end: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    time_start: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_thread: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    utime: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    exception: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rpc_result_expire_seconds: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    run_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    @property
    def msg_dict_obj(self) -> Optional[Any]:
        """Return the JSON-decoded message payload."""
        return json.loads(self.msg_dict) if self.msg_dict else None

    @msg_dict_obj.setter
    def msg_dict_obj(self, value: Any) -> None:
        self.msg_dict = json.dumps(value) if value is not None else None

    @property
    def params_obj(self) -> Optional[Any]:
        """Return the JSON-decoded params payload."""
        return json.loads(self.params) if self.params else None

    @params_obj.setter
    def params_obj(self, value: Any) -> None:
        self.params = json.dumps(value) if value is not None else None

    def __repr__(self) -> str:
        return f"<FunboostConsumeResult(id='{self._id}', function='{self.function}', task_id='{self.task_id}')>"


class RawDicomToNiiInference(Base):
    """Minimal ORM model that records raw dicom2nii inference metadata."""

    __tablename__ = "raw_dicom_to_nii_inference"

    _id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_id)
    name: Mapped[str] = mapped_column(String)
    sub_dir: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    output_dicom_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    output_nifti_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_time: Mapped[str] = mapped_column(String)
