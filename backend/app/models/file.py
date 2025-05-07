# app/models/backup_object.py
from sqlalchemy import (
    Column, DateTime, Integer,
    String, Index
)
from sqlalchemy.orm import relationship, Mapped
from typing import List, Optional
from datetime import datetime

from .base import Base  #
from .backup_object import BackupObject


class File(Base):
    """Model for original file metadata."""
    __tablename__ = "files"

    file_id: int = Column(Integer, primary_key=True, index=True)
    file_path: str = Column(String(1024), nullable=False)
    file_name: str = Column(String(255), nullable=False)
    file_extension: Optional[str] = Column(String(50))
    original_size: int = Column(Integer, nullable=False)
    original_hash: str = Column(String(128), nullable=False, index=True)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    source_system: str = Column(String(255), nullable=False)

    # Relationships
    backup_objects: Mapped[List["BackupObject"]] = relationship("BackupObject", back_populates="file")

    # Indexes
    __table_args__ = (
        Index("idx_files_hash", original_hash),
    )