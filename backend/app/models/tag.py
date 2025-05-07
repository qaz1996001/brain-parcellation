from sqlalchemy import ( Column, ForeignKey, Integer, String, UniqueConstraint)
from sqlalchemy.orm import relationship, Mapped

from .base import Base
from .backup_object import BackupObject

class Tag(Base):
    """Model for object tags."""
    __tablename__ = "tags"

    tag_id: int = Column(Integer, primary_key=True, index=True)
    object_id: int = Column(Integer, ForeignKey("backup_objects.object_id"), nullable=False)
    key: str = Column(String(128), nullable=False)
    value: str = Column(String(256), nullable=False)

    # Relationships
    backup_object: Mapped["BackupObject"] = relationship("BackupObject", back_populates="tags")

    # Constraints
    __table_args__ = (
        UniqueConstraint("object_id", "key", name="unique_object_tag"),
    )
