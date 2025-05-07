# app/schemas/object_version.py

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime




class ObjectVersionBase(BaseModel):
    """Base schema for object version data."""
    object_id: int
    minio_version_id: str
    version_number: int
    is_active: bool = True
    is_delete_marker: bool = False


class ObjectVersionCreate(ObjectVersionBase):
    """Schema for creating a new object version."""
    pass


class ObjectVersionResponse(ObjectVersionBase):
    """Schema for object version response."""
    version_id: int
    created_at: datetime

    class Config:
        orm_mode = True
