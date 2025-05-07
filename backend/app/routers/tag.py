# app/routers/tag.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Dict

from app.database import get_db
from app.schemas import TagCreate, TagResponse
from app.crud import tag_crud, backup_object_crud

router = APIRouter()


@router.post("/", response_model=TagResponse, status_code=201)
def create_tag(
        tag: TagCreate,
        db: Session = Depends(get_db)
) -> TagResponse:
    """Create a new tag for a backup object."""
    # Verify that the backup object exists
    obj = backup_object_crud.get(db=db, id=tag.object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Backup object not found")

    return tag_crud.create(db=db, obj_in=tag)


@router.get("/", response_model=List[TagResponse])
def read_tags(
        skip: int = 0,
        limit: int = 100,
        object_id: Optional[int] = None,
        db: Session = Depends(get_db)
) -> List[TagResponse]:
    """Get a list of tags with optional object ID filter."""
    if object_id:
        return tag_crud.get_by_object(db=db, object_id=object_id, skip=skip, limit=limit)
    return tag_crud.get_multi(db=db, skip=skip, limit=limit)


@router.get("/by-key-value/", response_model=List[TagResponse])
def read_tags_by_key_value(
        key: str,
        value: str,
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db)
) -> List[TagResponse]:
    """Get tags by key and value."""
    return tag_crud.get_by_key_value(db=db, key=key, value=value, skip=skip, limit=limit)


@router.delete("/{tag_id}", response_model=TagResponse)
def delete_tag(
        tag_id: int,
        db: Session = Depends(get_db)
) -> TagResponse:
    """Delete a specific tag."""
    tag = tag_crud.get(db=db, id=tag_id)
    if tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    return tag_crud.remove(db=db, id=tag_id)


@router.post("/search/", response_model=List[int])
def search_objects_by_tags(
        tags: Dict[str, str],
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db)
) -> List[int]:
    """
    Search for backup objects that match all specified tags.

    Args:
        tags: Dictionary of tag keys and values
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List of backup object IDs that match all specified tags
    """
    if not tags:
        raise HTTPException(status_code=400, detail="No tags provided for search")

    return tag_crud.get_objects_by_tags(db=db, tags=tags, skip=skip, limit=limit)