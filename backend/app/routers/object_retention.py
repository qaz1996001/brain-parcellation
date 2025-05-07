# app/routers/object_retention.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.database import get_db
from app.schemas import ObjectRetentionCreate, ObjectRetentionResponse, ObjectRetentionUpdate
from app.crud import object_retention_crud, backup_object_crud, retention_policy_crud

router = APIRouter()


@router.post("/", response_model=ObjectRetentionResponse, status_code=201)
def create_object_retention(
        retention: ObjectRetentionCreate,
        db: Session = Depends(get_db)
) -> ObjectRetentionResponse:
    """Create a new object retention entry."""
    # Verify that the backup object exists
    obj = backup_object_crud.get(db=db, id=retention.object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Backup object not found")

    # Verify that the retention policy exists
    policy = retention_policy_crud.get(db=db, id=retention.policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Retention policy not found")

    # Check if retention already exists for this object and policy
    existing_retention = object_retention_crud.get_by_object_and_policy(
        db=db,
        object_id=retention.object_id,
        policy_id=retention.policy_id
    )
    if existing_retention:
        raise HTTPException(
            status_code=400,
            detail="Retention already exists for this object and policy"
        )

    return object_retention_crud.create(db=db, obj_in=retention)


@router.get("/", response_model=List[ObjectRetentionResponse])
def read_object_retentions(
        skip: int = 0,
        limit: int = 100,
        object_id: Optional[int] = None,
        policy_id: Optional[int] = None,
        db: Session = Depends(get_db)
) -> List[ObjectRetentionResponse]:
    """Get a list of object retention entries with optional filters."""
    if object_id:
        return object_retention_crud.get_by_object(
            db=db,
            object_id=object_id,
            skip=skip,
            limit=limit
        )
    if policy_id:
        return object_retention_crud.get_by_policy(
            db=db,
            policy_id=policy_id,
            skip=skip,
            limit=limit
        )
    return object_retention_crud.get_multi(db=db, skip=skip, limit=limit)


@router.get("/expired/", response_model=List[ObjectRetentionResponse])
def read_expired_object_retentions(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db)
) -> List[ObjectRetentionResponse]:
    """Get a list of expired object retention entries."""
    return object_retention_crud.get_expired(db=db, skip=skip, limit=limit)


@router.get("/{retention_id}", response_model=ObjectRetentionResponse)
def read_object_retention(
        retention_id: int,
        db: Session = Depends(get_db)
) -> ObjectRetentionResponse:
    """Get a specific object retention entry by ID."""
    retention = object_retention_crud.get(db=db, id=retention_id)
    if retention is None:
        raise HTTPException(status_code=404, detail="Object retention not found")
    return retention


@router.put("/{retention_id}", response_model=ObjectRetentionResponse)
def update_object_retention(
        retention_id: int,
        retention: ObjectRetentionUpdate,
        db: Session = Depends(get_db)
) -> ObjectRetentionResponse:
    """Update a specific object retention entry."""
    db_retention = object_retention_crud.get(db=db, id=retention_id)
    if db_retention is None:
        raise HTTPException(status_code=404, detail="Object retention not found")

    # If the object is locked, check if modifications are allowed
    if db_retention.is_locked and (
            retention.expiry_date is not None or
            retention.is_locked is False
    ):
        # In compliance mode, no changes allowed once locked
        policy = retention_policy_crud.get(db=db, id=db_retention.policy_id)
        if policy and policy.mode == "compliance":
            raise HTTPException(
                status_code=403,
                detail="Cannot modify retention in compliance mode once locked"
            )

    return object_retention_crud.update(db=db, db_obj=db_retention, obj_in=retention)


@router.delete("/{retention_id}", response_model=ObjectRetentionResponse)
def delete_object_retention(
        retention_id: int,
        db: Session = Depends(get_db)
) -> ObjectRetentionResponse:
    """Delete a specific object retention entry."""
    retention = object_retention_crud.get(db=db, id=retention_id)
    if retention is None:
        raise HTTPException(status_code=404, detail="Object retention not found")

    # Check if deletion is allowed based on retention policy mode
    policy = retention_policy_crud.get(db=db, id=retention.policy_id)
    if policy and policy.mode == "compliance" and retention.is_locked:
        raise HTTPException(
            status_code=403,
            detail="Cannot delete retention in compliance mode once locked"
        )

    return object_retention_crud.remove(db=db, id=retention_id)


@router.post("/{retention_id}/apply-legal-hold", response_model=ObjectRetentionResponse)
def apply_legal_hold(
        retention_id: int,
        db: Session = Depends(get_db)
) -> ObjectRetentionResponse:
    """Apply legal hold to an object retention entry."""
    retention = object_retention_crud.get(db=db, id=retention_id)
    if retention is None:
        raise HTTPException(status_code=404, detail="Object retention not found")

    return object_retention_crud.apply_legal_hold(db=db, retention_id=retention_id)


@router.post("/{retention_id}/release-legal-hold", response_model=ObjectRetentionResponse)
def release_legal_hold(
        retention_id: int,
        db: Session = Depends(get_db)
) -> ObjectRetentionResponse:
    """Release legal hold from an object retention entry."""
    retention = object_retention_crud.get(db=db, id=retention_id)
    if retention is None:
        raise HTTPException(status_code=404, detail="Object retention not found")

    return object_retention_crud.release_legal_hold(db=db, retention_id=retention_id)


@router.post("/{retention_id}/lock", response_model=ObjectRetentionResponse)
def lock_retention(
        retention_id: int,
        db: Session = Depends(get_db)
) -> ObjectRetentionResponse:
    """Lock an object retention entry."""
    retention = object_retention_crud.get(db=db, id=retention_id)
    if retention is None:
        raise HTTPException(status_code=404, detail="Object retention not found")

    return object_retention_crud.lock_retention(db=db, retention_id=retention_id)