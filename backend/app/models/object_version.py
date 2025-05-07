# app/models/object_version.py
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer,
    String, Index
)
from sqlalchemy.orm import relationship, Mapped
from datetime import datetime

from .base import Base


class ObjectVersion(Base):
    """Model for object versions."""
    __tablename__ = "object_versions"

    version_id: int = Column(Integer, primary_key=True, index=True)
    object_id: int = Column(Integer, ForeignKey("backup_objects.object_id"), nullable=False)
    minio_version_id: str = Column(String(128), nullable=False, unique=True)
    version_number: int = Column(Integer, nullable=False)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active: bool = Column(Boolean, nullable=False, default=True)
    is_delete_marker: bool = Column(Boolean, nullable=False, default=False)

    # Relationships
    backup_object: Mapped["BackupObject"] = relationship("BackupObject", back_populates="object_versions")

    # Indexes
    __table_args__ = (
        Index("idx_object_versions_object_id", object_id),
        Index("idx_object_versions_minio_id", minio_version_id),
    )
