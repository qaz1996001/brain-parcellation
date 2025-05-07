# app/schemas/backup_job.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class BackupJobBase(BaseModel):
    """Base schema for backup job data."""
    job_name: str
    source_system: str
    start_time: datetime
    bucket_id: int


class BackupJobCreate(BackupJobBase):
    """Schema for creating a new backup job."""
    status: str = "pending"
    total_files: Optional[int] = 0
    total_size: Optional[int] = 0
    error_message: Optional[str] = None


class BackupJobUpdate(BaseModel):
    """Schema for updating a backup job."""
    job_name: Optional[str] = None
    source_system: Optional[str] = None
    end_time: Optional[datetime] = None
    status: Optional[str] = None
    total_files: Optional[int] = None
    total_size: Optional[int] = None
    error_message: Optional[str] = None


class BackupJobResponse(BackupJobBase):
    """Schema for backup job response."""
    job_id: int
    end_time: Optional[datetime] = None
    status: str
    total_files: int
    total_size: int
    error_message: Optional[str] = None

    class Config:
        orm_mode = True
