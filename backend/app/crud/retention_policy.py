# app/crud/retention_policy.py
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional, Union

from app.models import RetentionPolicy
from app.schemas import RetentionPolicyCreate, RetentionPolicyUpdate
from .base import CRUDBase


class CRUDRetentionPolicy(CRUDBase[RetentionPolicy, RetentionPolicyCreate, RetentionPolicyUpdate]):
    """CRUD operations for retention policy table."""

    def get_by_name(self, db: Session, *, name: str) -> Optional[RetentionPolicy]:
        """
        Get a retention policy by name.

        Args:
            db: Database session
            name: Policy name

        Returns:
            Retention policy object or None if not found
        """
        return db.query(RetentionPolicy).filter(RetentionPolicy.policy_name == name).first()

    def get_by_mode(
            self, db: Session, *, mode: str, skip: int = 0, limit: int = 100
    ) -> List[RetentionPolicy]:
        """
        Get retention policies by mode.

        Args:
            db: Database session
            mode: Policy mode ('governance' or 'compliance')
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of retention policy objects
        """
        return db.query(RetentionPolicy).filter(
            RetentionPolicy.mode == mode
        ).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: RetentionPolicyCreate) -> RetentionPolicy:
        """
        Create a new retention policy.

        Args:
            db: Database session
            obj_in: Retention policy creation schema

        Returns:
            Created retention policy object
        """
        db_obj = RetentionPolicy(
            policy_name=obj_in.policy_name,
            retention_period=obj_in.retention_period,
            mode=obj_in.mode,
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
            db_obj: RetentionPolicy,
            obj_in: Union[RetentionPolicyUpdate, Dict[str, Any]]
    ) -> RetentionPolicy:
        """
        Update a retention policy.

        Args:
            db: Database session
            db_obj: Existing retention policy object
            obj_in: Update schema or dictionary

        Returns:
            Updated retention policy object
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        return super().update(db, db_obj=db_obj, obj_in=update_data)


# Create a singleton instance
retention_policy_crud = CRUDRetentionPolicy(RetentionPolicy)