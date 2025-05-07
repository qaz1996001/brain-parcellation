# app/crud/base.py
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.models import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base class for CRUD operations.

    Attributes:
        model: The SQLAlchemy model class
    """

    def __init__(self, model: Type[ModelType]):
        """
        Initialize with SQLAlchemy model.

        Args:
            model: A SQLAlchemy model class
        """
        self.model = model

    def get(self, db: Session, id: int) -> Optional[ModelType]:
        """
        Get a single record by ID.

        Args:
            db: Database session
            id: Primary key ID value

        Returns:
            The model instance or None if not found
        """
        # Use dynamic approach to handle different primary key column names
        # For BackupJob it should use job_id, for File it should use file_id, etc.
        primary_key_column = None
        for column_name in self.model.__table__.columns.keys():
            if column_name.endswith('_id'):
                primary_key_column = column_name
                break

        if primary_key_column:
            return db.query(self.model).filter(getattr(self.model, primary_key_column) == id).first()
        else:
            # Fallback to id if no _id column is found
            return db.query(self.model).filter_by(id=id).first()

    def get_multi(
            self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        """
        Get multiple records with pagination.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of model instances
        """
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """
        Create a new record.

        Args:
            db: Database session
            obj_in: Schema for creating a record

        Returns:
            The created model instance
        """
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
            self,
            db: Session,
            *,
            db_obj: ModelType,
            obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """
        Update a record.

        Args:
            db: Database session
            db_obj: Model instance to update
            obj_in: Schema or dict with updated fields

        Returns:
            The updated model instance
        """
        obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, *, id: int) -> ModelType:
        """
        Delete a record.

        Args:
            db: Database session
            id: Primary key ID value

        Returns:
            The deleted model instance
        """
        # Use dynamic approach to handle different primary key column names
        # Find the primary key column name
        primary_key_column = None
        for column_name in self.model.__table__.columns.keys():
            if column_name.endswith('_id'):
                primary_key_column = column_name
                break

        if primary_key_column:
            obj = db.query(self.model).filter(getattr(self.model, primary_key_column) == id).first()
        else:
            # Fallback to id if no _id column is found
            obj = db.query(self.model).filter_by(id=id).first()

        if obj:
            db.delete(obj)
            db.commit()
        return obj