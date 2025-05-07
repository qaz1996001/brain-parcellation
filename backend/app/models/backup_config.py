# app/models/backup_config.py
from sqlalchemy import Column, Integer, String, Text, UniqueConstraint
# Use the base from another model file to maintain consistency
from .base import Base


class BackupConfig(Base):
    """Model for backup configuration."""
    __tablename__ = "backup_configs"

    config_id: int = Column(Integer, primary_key=True, index=True)
    config_key: str = Column(String(255), nullable=False, unique=True)
    config_value: str = Column(String(1024), nullable=False)
    description: str = Column(Text, nullable=True)

    # Ensure unique constraint on config_key
    __table_args__ = (
        UniqueConstraint('config_key', name='uq_backup_config_key'),
    )