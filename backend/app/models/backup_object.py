# app/models/backup_object.py
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer,
    String, Float, Index
)
from sqlalchemy.orm import relationship, Mapped
from typing import List, Optional
from datetime import datetime

from .base import Base



class BackupObject(Base):
    """Model for backup objects stored in MinIO."""
    __tablename__ = "backup_objects"

    object_id: int = Column(Integer, primary_key=True, index=True)
    file_id: int = Column(Integer, ForeignKey("files.file_id"), nullable=False)
    job_id: int = Column(Integer, ForeignKey("backup_jobs.job_id"), nullable=False)
    bucket_id: int = Column(Integer, ForeignKey("buckets.bucket_id"), nullable=False)
    object_key: str = Column(String(1024), nullable=False)
    storage_size: int = Column(Integer, nullable=False)
    latest_version: bool = Column(Boolean, nullable=False, default=True)
    is_compressed: bool = Column(Boolean, nullable=False, default=False)
    compression_ratio: Optional[float] = Column(Float)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    file: Mapped["File"] = relationship("File", back_populates="backup_objects")
    backup_job: Mapped["BackupJob"] = relationship("BackupJob", back_populates="backup_objects")
    bucket: Mapped["Bucket"] = relationship("Bucket", back_populates="backup_objects")
    object_versions: Mapped[List["ObjectVersion"]] = relationship("ObjectVersion", back_populates="backup_object")
    tags: Mapped[List["Tag"]] = relationship("Tag", back_populates="backup_object")
    object_retentions: Mapped[List["ObjectRetention"]] = relationship("ObjectRetention", back_populates="backup_object")

    # Indexes
    __table_args__ = (
        Index("idx_backup_objects_file_id", file_id),
        Index("idx_backup_objects_job_id", job_id),
        Index("idx_backup_objects_bucket_id", bucket_id),
    )