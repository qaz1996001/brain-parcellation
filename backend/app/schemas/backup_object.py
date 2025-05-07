# app/schemas/backup_object.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class BackupObjectBase(BaseModel):
    """Base schema for backup object data."""
    file_id: int
    job_id: int
    bucket_id: int
    object_key: str
    storage_size: int
    latest_version: bool = True
    is_compressed: bool = False
    compression_ratio: Optional[float] = None


class BackupObjectCreate(BackupObjectBase):
    """Schema for creating a new backup object."""
    pass


class BackupObjectUpdate(BaseModel):
    """Schema for updating a backup object."""
    object_key: Optional[str] = None
    storage_size: Optional[int] = None
    latest_version: Optional[bool] = None
    is_compressed: Optional[bool] = None
    compression_ratio: Optional[float] = None


class BackupObjectResponse(BackupObjectBase):
    """Schema for backup object response."""
    object_id: int
    created_at: datetime

    class Config:
        orm_mode = True

