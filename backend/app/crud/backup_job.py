# app/crud/backup_job.py
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional, Union

from app.models import BackupJob
from app.schemas import BackupJobCreate, BackupJobUpdate
from .base import CRUDBase


class CRUDBackupJob(CRUDBase[BackupJob, BackupJobCreate, BackupJobUpdate]):
    """CRUD operations for backup job table."""

    def get_by_status(
            self, db: Session, *, status: str, skip: int = 0, limit: int = 100
    ) -> List[BackupJob]:
        """
        Get backup jobs by status.

        Args:
            db: Database session
            status: Job status ('pending', 'running', 'completed', 'failed')
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of backup job objects
        """
        return db.query(BackupJob).filter(
            BackupJob.status == status
        ).offset(skip).limit(limit).all()

    def get_by_bucket(
            self, db: Session, *, bucket_id: int, skip: int = 0, limit: int = 100
    ) -> List[BackupJob]:
        """
        Get backup jobs for a specific bucket.

        Args:
            db: Database session
            bucket_id: Bucket ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of backup job objects
        """
        return db.query(BackupJob).filter(
            BackupJob.bucket_id == bucket_id
        ).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: BackupJobCreate) -> BackupJob:
        """
        Create a new backup job.

        Args:
            db: Database session
            obj_in: Backup job creation schema

        Returns:
            Created backup job object
        """
        db_obj = BackupJob(
            job_name=obj_in.job_name,
            source_system=obj_in.source_system,
            start_time=obj_in.start_time,
            bucket_id=obj_in.bucket_id,
            status=obj_in.status,
            total_files=obj_in.total_files,
            total_size=obj_in.total_size,
            error_message=obj_in.error_message,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
            self,
            db: Session,
            *,
            db_obj: BackupJob,
            obj_in: Union[BackupJobUpdate, Dict[str, Any]]
    ) -> BackupJob:
        """
        Update a backup job.

        Args:
            db: Database session
            db_obj: Existing backup job object
            obj_in: Update schema or dictionary

        Returns:
            Updated backup job object
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def get_pending_jobs(self, db: Session) -> List[BackupJob]:
        """
        Get all pending backup jobs.

        Args:
            db: Database session

        Returns:
            List of pending backup job objects
        """
        return db.query(BackupJob).filter(
            BackupJob.status == "pending"
        ).all()


# Create a singleton instance
backup_job_crud = CRUDBackupJob(BackupJob)