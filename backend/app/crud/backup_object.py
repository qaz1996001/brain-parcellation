# app/crud/backup_object.py
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional, Union

from backend.app.models import BackupObject
from backend.app.schemas import BackupObjectCreate, BackupObjectUpdate
from .base import CRUDBase


class CRUDBackupObject(CRUDBase[BackupObject, BackupObjectCreate, BackupObjectUpdate]):
    """CRUD operations for backup object table."""

    def get_by_job(
            self, db: Session, *, job_id: int, skip: int = 0, limit: int = 100
    ) -> List[BackupObject]:
        """
        Get backup objects for a specific job.

        Args:
            db: Database session
            job_id: Backup job ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of backup object objects
        """
        return db.query(BackupObject).filter(
            BackupObject.job_id == job_id
        ).offset(skip).limit(limit).all()

    def get_by_bucket(
            self, db: Session, *, bucket_id: int, skip: int = 0, limit: int = 100
    ) -> List[BackupObject]:
        """
        Get backup objects for a specific bucket.

        Args:
            db: Database session
            bucket_id: Bucket ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of backup object objects
        """
        return db.query(BackupObject).filter(
            BackupObject.bucket_id == bucket_id
        ).offset(skip).limit(limit).all()

    def get_by_file(
            self, db: Session, *, file_id: int, skip: int = 0, limit: int = 100
    ) -> List[BackupObject]:
        """
        Get backup objects for a specific file.

        Args:
            db: Database session
            file_id: File ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of backup object objects
        """
        return db.query(BackupObject).filter(
            BackupObject.file_id == file_id
        ).offset(skip).limit(limit).all()

    def get_by_object_key(
            self, db: Session, *, object_key: str, bucket_id: int
    ) -> Optional[BackupObject]:
        """
        Get a backup object by object key and bucket ID.

        Args:
            db: Database session
            object_key: Object key in MinIO
            bucket_id: Bucket ID

        Returns:
            Backup object or None if not found
        """
        return db.query(BackupObject).filter(
            BackupObject.object_key == object_key,
            BackupObject.bucket_id == bucket_id
        ).first()

    def get_latest(
            self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[BackupObject]:
        """
        Get latest versions of backup objects.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of latest backup object versions
        """
        return db.query(BackupObject).filter(
            BackupObject.latest_version == True
        ).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: BackupObjectCreate) -> BackupObject:
        """
        Create a new backup object record.

        Args:
            db: Database session
            obj_in: Backup object creation schema

        Returns:
            Created backup object
        """
        db_obj = BackupObject(
            file_id=obj_in.file_id,
            job_id=obj_in.job_id,
            bucket_id=obj_in.bucket_id,
            object_key=obj_in.object_key,
            storage_size=obj_in.storage_size,
            latest_version=obj_in.latest_version,
            is_compressed=obj_in.is_compressed,
            compression_ratio=obj_in.compression_ratio,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
            self,
            db: Session,
            *,
            db_obj: BackupObject,
            obj_in: Union[BackupObjectUpdate, Dict[str, Any]]
    ) -> BackupObject:
        """
        Update a backup object record.

        Args:
            db: Database session
            db_obj: Existing backup object
            obj_in: Update schema or dictionary

        Returns:
            Updated backup object
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def mark_as_latest_version(
            self, db: Session, *, object_id: int
    ) -> BackupObject:
        """
        Mark a backup object as the latest version and unmark previous latest.

        Args:
            db: Database session
            object_id: Backup object ID

        Returns:
            Updated backup object
        """
        # Get the backup object to be marked as latest
        obj = db.query(BackupObject).filter(BackupObject.object_id == object_id).first()
        if not obj:
            return None

        # Get the file ID to find other versions
        file_id = obj.file_id

        # Unmark all other versions of the file
        db.query(BackupObject).filter(
            BackupObject.file_id == file_id,
            BackupObject.object_id != object_id
        ).update({"latest_version": False})

        # Mark the current object as latest
        obj.latest_version = True
        db.add(obj)
        db.commit()
        db.refresh(obj)

        return obj


# Create a singleton instance
backup_object_crud = CRUDBackupObject(BackupObject)