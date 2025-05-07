# app/schemas/tag.py

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime




class TagBase(BaseModel):
    """Base schema for tag data."""
    object_id: int
    key: str
    value: str


class TagCreate(TagBase):
    """Schema for creating a new tag."""
    pass


class TagResponse(TagBase):
    """Schema for tag response."""
    tag_id: int

    class Config:
        orm_mode = True