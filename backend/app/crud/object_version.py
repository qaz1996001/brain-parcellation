# app/crud/object_version.py
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional, Union

from app.models import ObjectVersion
from app.schemas import ObjectVersionCreate
from .base import CRUDBase


class CRUDObjectVersion(CRUDBase[ObjectVersion, ObjectVersionCreate, Any]):
    """CRUD operations for object version table."""

    def get_by_object(
            self, db: Session, *, object_id: int, skip: int = 0, limit: int = 100
    ) -> List[ObjectVersion]:
        """
        Get versions for a specific backup object.

        Args:
            db: Database session
            object_id: Backup object ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of object version objects
        """
        return db.query(ObjectVersion).filter(
            ObjectVersion.object_id == object_id
        ).order_by(ObjectVersion.version_number.desc()).offset(skip).limit(limit).all()

    def get_by_minio_id(
            self, db: Session, *, minio_version_id: str
    ) -> Optional[ObjectVersion]:
        """
        Get object version by MinIO version ID.

        Args:
            db: Database session
            minio_version_id: MinIO version ID

        Returns:
            Object version or None if not found
        """
        return db.query(ObjectVersion).filter(
            ObjectVersion.minio_version_id == minio_version_id
        ).first()

    def get_latest_version(
            self, db: Session, *, object_id: int
    ) -> Optional[ObjectVersion]:
        """
        Get latest version for a specific backup object.

        Args:
            db: Database session
            object_id: Backup object ID

        Returns:
            Latest object version or None if not found
        """
        return db.query(ObjectVersion).filter(
            ObjectVersion.object_id == object_id,
            ObjectVersion.is_active == True,
            ObjectVersion.is_delete_marker == False
        ).order_by(ObjectVersion.version_number.desc()).first()

    def create(self, db: Session, *, obj_in: ObjectVersionCreate) -> ObjectVersion:
        """
        Create a new object version record.

        Args:
            db: Database session
            obj_in: Object version creation schema

        Returns:
            Created object version
        """
        # Calculate the next version number
        latest_version = db.query(ObjectVersion).filter(
            ObjectVersion.object_id == obj_in.object_id
        ).order_by(ObjectVersion.version_number.desc()).first()

        next_version_number = 1
        if latest_version:
            next_version_number = latest_version.version_number + 1

        db_obj = ObjectVersion(
            object_id=obj_in.object_id,
            minio_version_id=obj_in.minio_version_id,
            version_number=next_version_number,
            is_active=obj_in.is_active,
            is_delete_marker=obj_in.is_delete_marker,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def mark_as_delete_marker(
            self, db: Session, *, version_id: int
    ) -> ObjectVersion:
        """
        Mark an object version as a delete marker.

        Args:
            db: Database session
            version_id: Object version ID

        Returns:
            Updated object version
        """
        obj = db.query(ObjectVersion).filter(
            ObjectVersion.version_id == version_id
        ).first()

        if obj:
            obj.is_delete_marker = True
            db.add(obj)
            db.commit()
            db.refresh(obj)

        return obj

    def deactivate_version(
            self, db: Session, *, version_id: int
    ) -> ObjectVersion:
        """
        Deactivate an object version.

        Args:
            db: Database session
            version_id: Object version ID

        Returns:
            Updated object version
        """
        obj = db.query(ObjectVersion).filter(
            ObjectVersion.version_id == version_id
        ).first()

        if obj:
            obj.is_active = False
            db.add(obj)
            db.commit()
            db.refresh(obj)

        return obj


# Create a singleton instance
object_version_crud = CRUDObjectVersion(ObjectVersion)