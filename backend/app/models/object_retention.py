# app/models/object_retention.py
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer
)
from sqlalchemy.orm import relationship, Mapped
from datetime import datetime

from .base import Base
from .backup_object import BackupObject
from .retention_policy import RetentionPolicy

class ObjectRetention(Base):
    """Model for object retention settings."""
    __tablename__ = "object_retention"

    retention_id: int = Column(Integer, primary_key=True, index=True)
    object_id: int = Column(Integer, ForeignKey("backup_objects.object_id"), nullable=False)
    policy_id: int = Column(Integer, ForeignKey("retention_policies.policy_id"), nullable=False)
    expiry_date: datetime = Column(DateTime, nullable=False)
    is_locked: bool = Column(Boolean, nullable=False, default=False)
    legal_hold: bool = Column(Boolean, nullable=False, default=False)

    # Relationships
    backup_object: Mapped["BackupObject"] = relationship("BackupObject", back_populates="object_retentions")
    retention_policy: Mapped["RetentionPolicy"] = relationship("RetentionPolicy", back_populates="object_retentions")

