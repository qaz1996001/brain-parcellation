# app/routers/object_version.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.schemas import ObjectVersionCreate, ObjectVersionResponse
from app.crud import object_version_crud, backup_object_crud

router = APIRouter()


@router.post("/", response_model=ObjectVersionResponse, status_code=201)
def create_object_version(
        version: ObjectVersionCreate,
        db: Session = Depends(get_db)
) -> ObjectVersionResponse:
    """Create a new object version metadata entry."""
    # Verify that the backup object exists
    obj = backup_object_crud.get(db=db, id=version.object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Backup object not found")

    # Verify that the MinIO version ID is not already registered
    existing_version = object_version_crud.get_by_minio_id(db=db, minio_version_id=version.minio_version_id)
    if existing_version:
        raise HTTPException(status_code=400, detail="Version with this MinIO version ID already exists")

    return object_version_crud.create(db=db, obj_in=version)


@router.get("/", response_model=List[ObjectVersionResponse])
def read_object_versions(
        skip: int = 0,
        limit: int = 100,
        object_id: Optional[int] = None,
        db: Session = Depends(get_db)
) -> List[ObjectVersionResponse]:
    """Get a list of object version metadata entries with optional object ID filter."""
    if object_id:
        return object_version_crud.get_by_object(db=db, object_id=object_id, skip=skip, limit=limit)
    return object_version_crud.get_multi(db=db, skip=skip, limit=limit)


@router.get("/{version_id}", response_model=ObjectVersionResponse)
def read_object_version(
        version_id: int,
        db: Session = Depends(get_db)
) -> ObjectVersionResponse:
    """Get a specific object version metadata entry by ID."""
    version = object_version_crud.get(db=db, id=version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Object version not found")
    return version


@router.get("/by-minio-id/{minio_version_id}", response_model=ObjectVersionResponse)
def read_object_version_by_minio_id(
        minio_version_id: str,
        db: Session = Depends(get_db)
) -> ObjectVersionResponse:
    """Get an object version metadata entry by MinIO version ID."""
    version = object_version_crud.get_by_minio_id(db=db, minio_version_id=minio_version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Object version not found")
    return version


@router.delete("/{version_id}", response_model=ObjectVersionResponse)
def delete_object_version(
        version_id: int,
        db: Session = Depends(get_db)
) -> ObjectVersionResponse:
    """Delete a specific object version metadata entry."""
    version = object_version_crud.get(db=db, id=version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Object version not found")
    return object_version_crud.remove(db=db, id=version_id)


@router.post("/{version_id}/mark-as-delete-marker", response_model=ObjectVersionResponse)
def mark_as_delete_marker(
        version_id: int,
        db: Session = Depends(get_db)
) -> ObjectVersionResponse:
    """Mark an object version as a delete marker."""
    version = object_version_crud.get(db=db, id=version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Object version not found")

    return object_version_crud.mark_as_delete_marker(db=db, version_id=version_id)


@router.post("/{version_id}/deactivate", response_model=ObjectVersionResponse)
def deactivate_version(
        version_id: int,
        db: Session = Depends(get_db)
) -> ObjectVersionResponse:
    """Deactivate an object version."""
    version = object_version_crud.get(db=db, id=version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Object version not found")

    return object_version_crud.deactivate_version(db=db, version_id=version_id)