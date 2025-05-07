# app/models/backup_job.py
from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer,
    String,Text
)
from sqlalchemy.orm import relationship, Mapped
from typing import List, Optional
from datetime import datetime

from .base import Base



class BackupJob(Base):
    """Model for backup jobs."""
    __tablename__ = "backup_jobs"

    job_id: int = Column(Integer, primary_key=True, index=True)
    job_name: str = Column(String(255), nullable=False)
    source_system: str = Column(String(255), nullable=False)
    start_time: datetime = Column(DateTime, nullable=False)
    end_time: Optional[datetime] = Column(DateTime)
    status: str = Column(String(50), nullable=False, index=True)  # 'pending', 'running', 'completed', 'failed'
    total_files: int = Column(Integer, default=0)
    total_size: int = Column(Integer, default=0)  # in bytes
    bucket_id: int = Column(Integer, ForeignKey("buckets.bucket_id"), nullable=False)
    error_message: Optional[str] = Column(Text)

    # Relationships
    bucket: Mapped["Bucket"] = relationship("Bucket", back_populates="backup_jobs")
    backup_objects: Mapped[List["BackupObject"]] = relationship("BackupObject", back_populates="backup_job")



