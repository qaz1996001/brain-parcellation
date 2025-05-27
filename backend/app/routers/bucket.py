# app/routers/bucket.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.app.database import get_db
from backend.app.schemas import BucketCreate, BucketResponse, BucketUpdate
from backend.app.crud import bucket_crud

router = APIRouter()


@router.post("/", response_model=BucketResponse, status_code=201)
def create_bucket(
        bucket: BucketCreate,
        db: Session = Depends(get_db)
) -> BucketResponse:
    """Create a new MinIO bucket in the database."""
    db_bucket = bucket_crud.get_by_name(db, name=bucket.bucket_name)
    if db_bucket:
        raise HTTPException(status_code=400, detail="Bucket with this name already exists")
    return bucket_crud.create(db=db, obj_in=bucket)


@router.get("/", response_model=List[BucketResponse])
def read_buckets(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db)
) -> List[BucketResponse]:
    """Get a list of MinIO buckets."""
    return bucket_crud.get_multi(db=db, skip=skip, limit=limit)


@router.get("/{bucket_id}", response_model=BucketResponse)
def read_bucket(
        bucket_id: int,
        db: Session = Depends(get_db)
) -> BucketResponse:
    """Get a specific MinIO bucket by ID."""
    bucket = bucket_crud.get(db=db, id=bucket_id)
    if bucket is None:
        raise HTTPException(status_code=404, detail="Bucket not found")
    return bucket


@router.put("/{bucket_id}", response_model=BucketResponse)
def update_bucket(
        bucket_id: int,
        bucket: BucketUpdate,
        db: Session = Depends(get_db)
) -> BucketResponse:
    """Update a specific MinIO bucket."""
    db_bucket = bucket_crud.get(db=db, id=bucket_id)
    if db_bucket is None:
        raise HTTPException(status_code=404, detail="Bucket not found")

    # Check if name is being updated and if it already exists
    if bucket.bucket_name and bucket.bucket_name != db_bucket.bucket_name:
        existing_bucket = bucket_crud.get_by_name(db, name=bucket.bucket_name)
        if existing_bucket:
            raise HTTPException(status_code=400, detail="Bucket with this name already exists")

    return bucket_crud.update(db=db, db_obj=db_bucket, obj_in=bucket)


@router.delete("/{bucket_id}", response_model=BucketResponse)
def delete_bucket(
        bucket_id: int,
        db: Session = Depends(get_db)
) -> BucketResponse:
    """Delete a specific MinIO bucket."""
    bucket = bucket_crud.get(db=db, id=bucket_id)
    if bucket is None:
        raise HTTPException(status_code=404, detail="Bucket not found")
    return bucket_crud.remove(db=db, id=bucket_id)