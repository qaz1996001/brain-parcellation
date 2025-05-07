# app/crud/bucket.py
from sqlalchemy.orm import Session, selectinload
from typing import Any, Dict, List, Optional, Union

from app.models import Bucket, Base
from app.schemas import BucketCreate, BucketUpdate
from .base import CRUDBase


class CRUDBucket(CRUDBase[Bucket, BucketCreate, BucketUpdate]):
    """CRUD operations for bucket table."""

    def get_by_name(self, db: Session, *, name: str) -> Optional[Bucket]:
        """
        Get a bucket by name.

        Args:
            db: Database session
            name: Bucket name

        Returns:
            Bucket object or None if not found
        """
        return db.query(Bucket).filter(Bucket.bucket_name == name).first()

    def create(self, db: Session, *, obj_in: BucketCreate) -> Bucket:
        """
        Create a new bucket.

        Args:
            db: Database session
            obj_in: Bucket creation schema

        Returns:
            Created bucket object
        """
        db_obj = Bucket(
            bucket_name=obj_in.bucket_name,
            versioning_enabled=obj_in.versioning_enabled,
            description=obj_in.description,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
            self,
            db: Session,
            *,
            db_obj: Bucket,
            obj_in: Union[BucketUpdate, Dict[str, Any]]
    ) -> Bucket:
        """
        Update a bucket.

        Args:
            db: Database session
            db_obj: Existing bucket object
            obj_in: Update schema or dictionary

        Returns:
            Updated bucket object
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def get_all_with_backup_jobs(self, db: Session) -> List[Bucket]:
        """
        Get all buckets with their backup jobs.

        Args:
            db: Database session

        Returns:
            List of buckets with joined backup jobs
        """
        return db.query(Bucket).options(
            selectinload(Bucket.backup_jobs)
        ).all()


# Create a singleton instance
bucket_crud = CRUDBucket(Bucket)