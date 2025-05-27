# app/crud/tag.py
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional, Union

from backend.app.models import Tag
from backend.app.schemas import TagCreate
from .base import CRUDBase


class CRUDTag(CRUDBase[Tag, TagCreate, Any]):
    """CRUD operations for tag table."""

    def get_by_object(
            self, db: Session, *, object_id: int, skip: int = 0, limit: int = 100
    ) -> List[Tag]:
        """
        Get tags for a specific backup object.

        Args:
            db: Database session
            object_id: Backup object ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of tag objects
        """
        return db.query(Tag).filter(
            Tag.object_id == object_id
        ).offset(skip).limit(limit).all()

    def get_by_key_value(
            self, db: Session, *, key: str, value: str, skip: int = 0, limit: int = 100
    ) -> List[Tag]:
        """
        Get tags by key and value.

        Args:
            db: Database session
            key: Tag key
            value: Tag value
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of tag objects
        """
        return db.query(Tag).filter(
            Tag.key == key,
            Tag.value == value
        ).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: TagCreate) -> Tag:
        """
        Create a new tag record.

        Args:
            db: Database session
            obj_in: Tag creation schema

        Returns:
            Created tag object
        """
        # Check if we already have a tag with the same key for this object
        existing_tag = db.query(Tag).filter(
            Tag.object_id == obj_in.object_id,
            Tag.key == obj_in.key
        ).first()

        # If exists, update the value
        if existing_tag:
            existing_tag.value = obj_in.value
            db.add(existing_tag)
            db.commit()
            db.refresh(existing_tag)
            return existing_tag

        # Otherwise create new tag
        db_obj = Tag(
            object_id=obj_in.object_id,
            key=obj_in.key,
            value=obj_in.value,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_objects_by_tags(
            self, db: Session, *, tags: Dict[str, str], skip: int = 0, limit: int = 100
    ) -> List[int]:
        """
        Get backup object IDs that match all specified tags.

        Args:
            db: Database session
            tags: Dictionary of tag keys and values
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of backup object IDs
        """
        query = db.query(Tag.object_id)

        for key, value in tags.items():
            subquery = db.query(Tag.object_id).filter(
                Tag.key == key,
                Tag.value == value
            ).subquery()

            query = query.filter(Tag.object_id.in_(subquery))

        return query.distinct().offset(skip).limit(limit).all()


# Create a singleton instance
tag_crud = CRUDTag(Tag)