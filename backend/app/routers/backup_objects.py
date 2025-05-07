# app/routers/backup_objects.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.schemas import BackupObjectCreate, BackupObjectResponse, BackupObjectUpdate
from app.crud import backup_object_crud, file_crud, backup_job_crud, bucket_crud

router = APIRouter()


@router.post("/", response_model=BackupObjectResponse, status_code=201)
def create_backup_object(
        obj: BackupObjectCreate,
        db: Session = Depends(get_db)
) -> BackupObjectResponse:
    """Create a new backup object metadata entry."""
    # Verify that the file exists
    file = file_crud.get(db=db, id=obj.file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify that the job exists
    job = backup_job_crud.get(db=db, id=obj.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Backup job not found")

    # Verify that the bucket exists
    bucket = bucket_crud.get(db=db, id=obj.bucket_id)
    if not bucket:
        raise HTTPException(status_code=404, detail="Bucket not found")

    return backup_object_crud.create(db=db, obj_in=obj)


@router.get("/", response_model=List[BackupObjectResponse])
def read_backup_objects(
        skip: int = 0,
        limit: int = 100,
        job_id: Optional[int] = None,
        bucket_id: Optional[int] = None,
        file_id: Optional[int] = None,
        db: Session = Depends(get_db)
) -> List[BackupObjectResponse]:
    """Get a list of backup object metadata entries with optional filters."""
    if job_id:
        return backup_object_crud.get_by_job(db=db, job_id=job_id, skip=skip, limit=limit)
    if bucket_id:
        return backup_object_crud.get_by_bucket(db=db, bucket_id=bucket_id, skip=skip, limit=limit)
    if file_id:
        return backup_object_crud.get_by_file(db=db, file_id=file_id, skip=skip, limit=limit)
    return backup_object_crud.get_multi(db=db, skip=skip, limit=limit)


@router.get("/{object_id}", response_model=BackupObjectResponse)
def read_backup_object(
        object_id: int,
        db: Session = Depends(get_db)
) -> BackupObjectResponse:
    """Get a specific backup object metadata entry by ID."""
    obj = backup_object_crud.get(db=db, id=object_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Backup object not found")
    return obj


@router.get("/latest/", response_model=List[BackupObjectResponse])
def read_latest_backup_objects(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db)
) -> List[BackupObjectResponse]:
    """Get a list of latest backup object versions."""
    return backup_object_crud.get_latest(db=db, skip=skip, limit=limit)


@router.put("/{object_id}", response_model=BackupObjectResponse)
def update_backup_object(
        object_id: int,
        obj: BackupObjectUpdate,
        db: Session = Depends(get_db)
) -> BackupObjectResponse:
    """Update a specific backup object metadata entry."""
    db_obj = backup_object_crud.get(db=db, id=object_id)
    if db_obj is None:
        raise HTTPException(status_code=404, detail="Backup object not found")
    return backup_object_crud.update(db=db, db_obj=db_obj, obj_in=obj)


@router.delete("/{object_id}", response_model=BackupObjectResponse)
def delete_backup_object(
        object_id: int,
        db: Session = Depends(get_db)
) -> BackupObjectResponse:
    """Delete a specific backup object metadata entry."""
    obj = backup_object_crud.get(db=db, id=object_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Backup object not found")
    return backup_object_crud.remove(db=db, id=object_id)


@router.post("/{object_id}/mark_as_latest", response_model=BackupObjectResponse)
def mark_backup_object_as_latest(
        object_id: int,
        db: Session = Depends(get_db)
) -> BackupObjectResponse:
    """Mark a backup object as the latest version."""
    obj = backup_object_crud.get(db=db, id=object_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Backup object not found")

    return backup_object_crud.mark_as_latest_version(db=db, object_id=object_id)