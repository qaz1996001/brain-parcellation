# app/models.py
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer,
    String, Float, Text, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import List, Optional

Base = declarative_base()


class Bucket(Base):
    """Model for MinIO buckets."""
    __tablename__ = "buckets"

    bucket_id: int = Column(Integer, primary_key=True, index=True)
    bucket_name: str = Column(String(255), unique=True, nullable=False)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    versioning_enabled: bool = Column(Boolean, nullable=False, default=True)
    description: Optional[str] = Column(Text)

    # Relationships
    backup_jobs: List["BackupJob"] = relationship("BackupJob", back_populates="bucket")
    backup_objects: List["BackupObject"] = relationship("BackupObject", back_populates="bucket")


class BackupJob(Base):
    """Model for backup jobs."""
    __tablename__ = "backup_jobs"

    job_id: int = Column(Integer, primary_key=True, index=True)
    job_name: str = Column(String(255), nullable=False)
    source_system: str = Column(String(255), nullable=False)
    start_time: datetime = Column(DateTime, nullable=False)
    end_time: Optional[datetime] = Column(DateTime)
    status: str = Column(String(50), nullable=False, index=True)  # 'pending', 'running', 'completed', 'failed'
    total_files: int = Column(Integer, default=0)
    total_size: int = Column(Integer, default=0)  # in bytes
    bucket_id: int = Column(Integer, ForeignKey("buckets.bucket_id"), nullable=False)
    error_message: Optional[str] = Column(Text)

    # Relationships
    bucket: Bucket = relationship("Bucket", back_populates="backup_jobs")
    backup_objects: List["BackupObject"] = relationship("BackupObject", back_populates="backup_job")


class File(Base):
    """Model for original file metadata."""
    __tablename__ = "files"

    file_id: int = Column(Integer, primary_key=True, index=True)
    file_path: str = Column(String(1024), nullable=False)
    file_name: str = Column(String(255), nullable=False)
    file_extension: Optional[str] = Column(String(50))
    original_size: int = Column(Integer, nullable=False)
    original_hash: str = Column(String(128), nullable=False, index=True)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    source_system: str = Column(String(255), nullable=False)

    # Relationships
    backup_objects: List["BackupObject"] = relationship("BackupObject", back_populates="file")

    # Indexes
    __table_args__ = (
        Index("idx_files_hash", original_hash),
    )


class BackupObject(Base):
    """Model for backup objects stored in MinIO."""
    __tablename__ = "backup_objects"

    object_id: int = Column(Integer, primary_key=True, index=True)
    file_id: int = Column(Integer, ForeignKey("files.file_id"), nullable=False)
    job_id: int = Column(Integer, ForeignKey("backup_jobs.job_id"), nullable=False)
    bucket_id: int = Column(Integer, ForeignKey("buckets.bucket_id"), nullable=False)
    object_key: str = Column(String(1024), nullable=False)
    storage_size: int = Column(Integer, nullable=False)
    latest_version: bool = Column(Boolean, nullable=False, default=True)
    is_compressed: bool = Column(Boolean, nullable=False, default=False)
    compression_ratio: Optional[float] = Column(Float)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    file: File = relationship("File", back_populates="backup_objects")
    backup_job: BackupJob = relationship("BackupJob", back_populates="backup_objects")
    bucket: Bucket = relationship("Bucket", back_populates="backup_objects")
    object_versions: List["ObjectVersion"] = relationship("ObjectVersion", back_populates="backup_object")
    tags: List["Tag"] = relationship("Tag", back_populates="backup_object")
    object_retentions: List["ObjectRetention"] = relationship("ObjectRetention", back_populates="backup_object")

    # Indexes
    __table_args__ = (
        Index("idx_backup_objects_file_id", file_id),
        Index("idx_backup_objects_job_id", job_id),
        Index("idx_backup_objects_bucket_id", bucket_id),
    )


class ObjectVersion(Base):
    """Model for object versions."""
    __tablename__ = "object_versions"

    version_id: int = Column(Integer, primary_key=True, index=True)
    object_id: int = Column(Integer, ForeignKey("backup_objects.object_id"), nullable=False)
    minio_version_id: str = Column(String(128), nullable=False, unique=True)
    version_number: int = Column(Integer, nullable=False)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active: bool = Column(Boolean, nullable=False, default=True)
    is_delete_marker: bool = Column(Boolean, nullable=False, default=False)

    # Relationships
    backup_object: BackupObject = relationship("BackupObject", back_populates="object_versions")

    # Indexes
    __table_args__ = (
        Index("idx_object_versions_object_id", object_id),
        Index("idx_object_versions_minio_id", minio_version_id),
    )


class Tag(Base):
    """Model for object tags."""
    __tablename__ = "tags"

    tag_id: int = Column(Integer, primary_key=True, index=True)
    object_id: int = Column(Integer, ForeignKey("backup_objects.object_id"), nullable=False)
    key: str = Column(String(128), nullable=False)
    value: str = Column(String(256), nullable=False)

    # Relationships
    backup_object: BackupObject = relationship("BackupObject", back_populates="tags")

    # Constraints
    __table_args__ = (
        UniqueConstraint("object_id", "key", name="unique_object_tag"),
    )


class RetentionPolicy(Base):
    """Model for retention policies."""
    __tablename__ = "retention_policies"

    policy_id: int = Column(Integer, primary_key=True, index=True)
    policy_name: str = Column(String(255), unique=True, nullable=False)
    retention_period: int = Column(Integer, nullable=False)  # in days
    mode: str = Column(String(50), nullable=False)  # 'governance' or 'compliance'
    description: Optional[str] = Column(Text)

    # Relationships
    object_retentions: List["ObjectRetention"] = relationship("ObjectRetention", back_populates="retention_policy")


class ObjectRetention(Base):
    """Model for object retention settings."""
    __tablename__ = "object_retention"

    retention_id: int = Column(Integer, primary_key=True, index=True)
    object_id: int = Column(Integer, ForeignKey("backup_objects.object_id"), nullable=False)
    policy_id: int = Column(Integer, ForeignKey("retention_policies.policy_id"), nullable=False)
    expiry_date: datetime = Column(DateTime, nullable=False)
    is_locked: bool = Column(Boolean, nullable=False, default=False)
    legal_hold: bool = Column(Boolean, nullable=False, default=False)

    # Relationships
    backup_object: BackupObject = relationship("BackupObject", back_populates="object_retentions")
    retention_policy: RetentionPolicy = relationship("RetentionPolicy", back_populates="object_retentions")
