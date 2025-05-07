# app/routers/file.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.schemas import FileCreate, FileResponse, FileUpdate
from app.crud import file_crud

router = APIRouter()

@router.post("/", response_model=FileResponse, status_code=201)
def create_file(
    file: FileCreate,
    db: Session = Depends(get_db)
) -> FileResponse:
    """Create a new file metadata entry."""
    return file_crud.create(db=db, obj_in=file)

@router.get("/", response_model=List[FileResponse])
def read_files(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[FileResponse]:
    """Get a list of file metadata entries."""
    return file_crud.get_multi(db=db, skip=skip, limit=limit)

@router.get("/{file_id}", response_model=FileResponse)
def read_file(
    file_id: int,
    db: Session = Depends(get_db)
) -> FileResponse:
    """Get a specific file metadata entry by ID."""
    file = file_crud.get(db=db, id=file_id)
    if file is None:
        raise HTTPException(status_code=404, detail="File not found")
    return file

@router.get("/by-hash/{file_hash}", response_model=List[FileResponse])
def read_files_by_hash(
    file_hash: str,
    db: Session = Depends(get_db)
) -> List[FileResponse]:
    """Get file metadata entries by content hash."""
    files = file_crud.get_by_hash(db=db, file_hash=file_hash)
    if not files:
        raise HTTPException(status_code=404, detail="No files found with this hash")
    return files

@router.put("/{file_id}", response_model=FileResponse)
def update_file(
    file_id: int,
    file: FileUpdate,
    db: Session = Depends(get_db)
) -> FileResponse:
    """Update a specific file metadata entry."""
    db_file = file_crud.get(db=db, id=file_id)
    if db_file is None:
        raise HTTPException(status_code=404, detail="File not found")
    return file_crud.update(db=db, db_obj=db_file, obj_in=file)

@router.delete("/{file_id}", response_model=FileResponse)
def delete_file(
    file_id: int,
    db: Session = Depends(get_db)
) -> FileResponse:
    """Delete a specific file metadata entry."""
    file = file_crud.get(db=db, id=file_id)
    if file is None:
        raise HTTPException(status_code=404, detail="File not found")
    return file_crud.remove(db=db, id=file_id)

@router.get("/duplicates/", response_model=List[dict])
def get_duplicate_files(
    db: Session = Depends(get_db)
) -> List[dict]:
    """Get information about duplicate files."""
    return file_crud.get_duplicate_files(db=db)