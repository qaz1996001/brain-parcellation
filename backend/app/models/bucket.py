# app/models/bucket.py
from sqlalchemy import (
    Boolean, Column, DateTime, Integer, String, Text
)
from sqlalchemy.orm import relationship, Mapped
from typing import List, Optional
from datetime import datetime

from .base import Base  # or Base = declarative_base() if not imported
from .backup_job import BackupJob
from .backup_object import BackupObject



class Bucket(Base):
    """Model for MinIO buckets."""
    __tablename__ = "buckets"

    bucket_id: int = Column(Integer, primary_key=True, index=True)
    bucket_name: str = Column(String(255), unique=True, nullable=False)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    versioning_enabled: bool = Column(Boolean, nullable=False, default=True)
    description: Optional[str] = Column(Text)

    # Relationships
    backup_jobs: Mapped[List["BackupJob"]] = relationship("BackupJob", back_populates="bucket")
    backup_objects: Mapped[List["BackupObject"]] = relationship("BackupObject", back_populates="bucket")
