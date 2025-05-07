# app/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


# Bucket schemas
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


# Backup Job schemas
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


# File schemas
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


# Backup Object schemas
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


# Object Version schemas
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


# Tag schemas
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


# Retention Policy schemas
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


# Object Retention schemas
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