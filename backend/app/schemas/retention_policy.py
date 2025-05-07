# app/schemas/retention_policy.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime



class RetentionPolicyBase(BaseModel):
    """Base schema for retention policy data."""
    policy_name: str
    retention_period: int  # in days
    mode: str  # 'governance' or 'compliance'
    description: Optional[str] = None

    @validator('mode')
    def validate_mode(cls, v: str) -> str:
        """Validate that mode is either 'governance' or 'compliance'."""
        if v not in ["governance", "compliance"]:
            raise ValueError("Mode must be either 'governance' or 'compliance'")
        return v


class RetentionPolicyCreate(RetentionPolicyBase):
    """Schema for creating a new retention policy."""
    pass


class RetentionPolicyUpdate(BaseModel):
    """Schema for updating a retention policy."""
    policy_name: Optional[str] = None
    retention_period: Optional[int] = None
    mode: Optional[str] = None
    description: Optional[str] = None

    @validator('mode')
    def validate_mode(cls, v: Optional[str]) -> Optional[str]:
        """Validate that mode is either 'governance' or 'compliance'."""
        if v is not None and v not in ["governance", "compliance"]:
            raise ValueError("Mode must be either 'governance' or 'compliance'")
        return v


class RetentionPolicyResponse(RetentionPolicyBase):
    """Schema for retention policy response."""
    policy_id: int

    class Config:
        orm_mode = True

