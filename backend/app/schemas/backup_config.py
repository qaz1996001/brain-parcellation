# app/schemas/backup_config.py
from pydantic import BaseModel, Field
from typing import Optional


class BackupConfigBase(BaseModel):
    """Base schema for backup configuration."""
    config_key: str = Field(
        ...,
        description="Unique configuration key",
        min_length=1,
        max_length=255
    )
    config_value: str = Field(
        ...,
        description="Configuration value",
        min_length=1,
        max_length=1024
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of the configuration",
        max_length=1024
    )


class BackupConfigCreate(BackupConfigBase):
    """Schema for creating a new backup configuration."""
    pass


class BackupConfigUpdate(BaseModel):
    """Schema for updating a backup configuration."""
    config_value: Optional[str] = Field(
        None,
        description="Updated configuration value",
        min_length=1,
        max_length=1024
    )
    description: Optional[str] = Field(
        None,
        description="Updated configuration description",
        max_length=1024
    )


class BackupConfigResponse(BackupConfigBase):
    """Schema for backup configuration response."""
    config_id: int

    class Config:
        orm_mode = True