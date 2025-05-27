# app/crud/object_retention.py
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from backend.app.models import ObjectRetention
from backend.app.schemas import ObjectRetentionCreate, ObjectRetentionUpdate
from .base import CRUDBase


class CRUDObjectRetention(CRUDBase[ObjectRetention, ObjectRetentionCreate, ObjectRetentionUpdate]):
    """CRUD operations for object retention table."""

    def get_by_object(
            self, db: Session, *, object_id: int, skip: int = 0, limit: int = 100
    ) -> List[ObjectRetention]:
        """
        Get retention entries for a specific backup object.

        Args:
            db: Database session
            object_id: Backup object ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of object retention objects
        """
        return db.query(ObjectRetention).filter(
            ObjectRetention.object_id == object_id
        ).offset(skip).limit(limit).all()

    def get_by_policy(
            self, db: Session, *, policy_id: int, skip: int = 0, limit: int = 100
    ) -> List[ObjectRetention]:
        """
        Get retention entries for a specific retention policy.

        Args:
            db: Database session
            policy_id: Retention policy ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of object retention objects
        """
        return db.query(ObjectRetention).filter(
            ObjectRetention.policy_id == policy_id
        ).offset(skip).limit(limit).all()

    def get_by_object_and_policy(
            self, db: Session, *, object_id: int, policy_id: int
    ) -> Optional[ObjectRetention]:
        """
        Get retention entry for a specific object and policy.

        Args:
            db: Database session
            object_id: Backup object ID
            policy_id: Retention policy ID

        Returns:
            Object retention or None if not found
        """
        return db.query(ObjectRetention).filter(
            ObjectRetention.object_id == object_id,
            ObjectRetention.policy_id == policy_id
        ).first()

    def get_expired(
            self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[ObjectRetention]:
        """
        Get retention entries that have expired.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of expired object retention objects
        """
        now = datetime.utcnow()
        return db.query(ObjectRetention).filter(
            ObjectRetention.expiry_date < now,
            ObjectRetention.is_locked == False,
            ObjectRetention.legal_hold == False
        ).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: ObjectRetentionCreate) -> ObjectRetention:
        """
        Create a new object retention entry.

        Args:
            db: Database session
            obj_in: Object retention creation schema

        Returns:
            Created object retention object
        """
        db_obj = ObjectRetention(
            object_id=obj_in.object_id,
            policy_id=obj_in.policy_id,
            expiry_date=obj_in.expiry_date,
            is_locked=obj_in.is_locked,
            legal_hold=obj_in.legal_hold,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
            self,
            db: Session,
            *,
            db_obj: ObjectRetention,
            obj_in: Union[ObjectRetentionUpdate, Dict[str, Any]]
    ) -> ObjectRetention:
        """
        Update an object retention entry.

        Args:
            db: Database session
            db_obj: Existing object retention object
            obj_in: Update schema or dictionary

        Returns:
            Updated object retention object
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def apply_legal_hold(
            self, db: Session, *, retention_id: int
    ) -> ObjectRetention:
        """
        Apply legal hold to an object retention entry.

        Args:
            db: Database session
            retention_id: Object retention ID

        Returns:
            Updated object retention object
        """
        obj = db.query(ObjectRetention).filter(
            ObjectRetention.retention_id == retention_id
        ).first()

        if obj:
            obj.legal_hold = True
            db.add(obj)
            db.commit()
            db.refresh(obj)

        return obj

    def release_legal_hold(
            self, db: Session, *, retention_id: int
    ) -> ObjectRetention:
        """
        Release legal hold from an object retention entry.

        Args:
            db: Database session
            retention_id: Object retention ID

        Returns:
            Updated object retention object
        """
        obj = db.query(ObjectRetention).filter(
            ObjectRetention.retention_id == retention_id
        ).first()

        if obj:
            obj.legal_hold = False
            db.add(obj)
            db.commit()
            db.refresh(obj)

        return obj

    def lock_retention(
            self, db: Session, *, retention_id: int
    ) -> ObjectRetention:
        """
        Lock an object retention entry.

        Args:
            db: Database session
            retention_id: Object retention ID

        Returns:
            Updated object retention object
        """
        obj = db.query(ObjectRetention).filter(
            ObjectRetention.retention_id == retention_id
        ).first()

        if obj:
            obj.is_locked = True
            db.add(obj)
            db.commit()
            db.refresh(obj)

        return obj


# Create a singleton instance
object_retention_crud = CRUDObjectRetention(ObjectRetention)