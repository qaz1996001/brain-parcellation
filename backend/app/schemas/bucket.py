# app/schemas/bucket.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class BucketBase(BaseModel):
    """Base schema for bucket data."""
    bucket_name: str
    versioning_enabled: bool = True
    description: Optional[str] = None


class BucketCreate(BucketBase):
    """Schema for creating a new bucket."""
    pass


class BucketUpdate(BaseModel):
    """Schema for updating a bucket."""
    bucket_name: Optional[str] = None
    versioning_enabled: Optional[bool] = None
    description: Optional[str] = None


class BucketResponse(BucketBase):
    """Schema for bucket response."""
    bucket_id: int
    created_at: datetime

    class Config:
        orm_mode = True
