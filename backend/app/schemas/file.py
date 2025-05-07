# app/schemas/file.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class FileBase(BaseModel):
    """Base schema for file data."""
    file_path: str
    file_name: str
    file_extension: Optional[str] = None
    original_size: int
    original_hash: str
    source_system: str


class FileCreate(FileBase):
    """Schema for creating a new file."""
    pass


class FileUpdate(BaseModel):
    """Schema for updating a file."""
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_extension: Optional[str] = None
    original_size: Optional[int] = None
    original_hash: Optional[str] = None
    source_system: Optional[str] = None



class FileResponse(FileBase):
    """Schema for file response."""
    file_id: int
    created_at: datetime

    class Config:
        orm_mode = True