# app/models/object_version.py
from sqlalchemy import (
    Column, Integer, String, Text,
)
from sqlalchemy.orm import relationship, Mapped
from typing import List, Optional

from .base import Base


class RetentionPolicy(Base):
    """Model for retention policies."""
    __tablename__ = "retention_policies"

    policy_id: int = Column(Integer, primary_key=True, index=True)
    policy_name: str = Column(String(255), unique=True, nullable=False)
    retention_period: int = Column(Integer, nullable=False)  # in days
    mode: str = Column(String(50), nullable=False)  # 'governance' or 'compliance'
    description: Optional[str] = Column(Text)

    # Relationships
    object_retentions: Mapped[List["ObjectRetention"]] = relationship("ObjectRetention",
                                                                      back_populates="retention_policy")

