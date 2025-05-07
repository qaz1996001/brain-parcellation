# app/schemas/object_retention.py
from pydantic import BaseModel


class ObjectRetentionBase(BaseModel):
    """Base schema for object retention data."""
    object_id: int
    policy_id: int
    expiry_date: datetime
    is_locked: bool = False
    legal_hold: bool = False


class ObjectRetentionCreate(ObjectRetentionBase):
    """Schema for creating a new object retention entry."""
    pass


class ObjectRetentionUpdate(BaseModel):
    """Schema for updating an object retention entry."""
    expiry_date: Optional[datetime] = None
    is_locked: Optional[bool] = None
    legal_hold: Optional[bool] = None


class ObjectRetentionResponse(ObjectRetentionBase):
    """Schema for object retention response."""
    retention_id: int

    class Config:
        orm_mode = True