# app/routers/backup_config.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.app.database import get_db
from backend.app.schemas import BackupConfigCreate, BackupConfigResponse, BackupConfigUpdate
from backend.app.crud import backup_config_crud

router = APIRouter()


@router.post("/", response_model=BackupConfigResponse, status_code=201)
def create_backup_config(
        config: BackupConfigCreate,
        db: Session = Depends(get_db)
) -> BackupConfigResponse:
    """Create a new backup configuration."""
    # Check if config with the same key already exists
    existing_config = backup_config_crud.get_by_key(db, config_key=config.config_key)
    if existing_config:
        raise HTTPException(
            status_code=400,
            detail=f"Configuration with key '{config.config_key}' already exists"
        )

    return backup_config_crud.create(db=db, obj_in=config)


@router.get("/", response_model=List[BackupConfigResponse])
def read_backup_configs(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db)
) -> List[BackupConfigResponse]:
    """Get a list of backup configurations."""
    return backup_config_crud.get_multi(db=db, skip=skip, limit=limit)


@router.get("/{config_id}", response_model=BackupConfigResponse)
def read_backup_config_by_id(
        config_id: int,
        db: Session = Depends(get_db)
) -> BackupConfigResponse:
    """Get a specific backup configuration by ID."""
    config = backup_config_crud.get(db=db, id=config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Backup configuration not found")
    return config


@router.get("/key/{config_key}", response_model=BackupConfigResponse)
def read_backup_config_by_key(
        config_key: str,
        db: Session = Depends(get_db)
) -> BackupConfigResponse:
    """Get a specific backup configuration by key."""
    config = backup_config_crud.get_by_key(db=db, config_key=config_key)
    if config is None:
        raise HTTPException(status_code=404, detail="Backup configuration not found")
    return config


@router.put("/{config_id}", response_model=BackupConfigResponse)
def update_backup_config(
        config_id: int,
        config: BackupConfigUpdate,
        db: Session = Depends(get_db)
) -> BackupConfigResponse:
    """Update a specific backup configuration."""
    db_config = backup_config_crud.get(db=db, id=config_id)
    if db_config is None:
        raise HTTPException(status_code=404, detail="Backup configuration not found")

    return backup_config_crud.update(db=db, db_obj=db_config, obj_in=config)


@router.put("/upsert/", response_model=BackupConfigResponse)
def upsert_backup_config(
        config: BackupConfigCreate,
        db: Session = Depends(get_db)
) -> BackupConfigResponse:
    """Upsert a backup configuration."""
    return backup_config_crud.upsert(
        db=db,
        config_key=config.config_key,
        config_value=config.config_value,
        description=config.description
    )


@router.delete("/{config_id}", response_model=BackupConfigResponse)
def delete_backup_config(
        config_id: int,
        db: Session = Depends(get_db)
) -> BackupConfigResponse:
    """Delete a specific backup configuration."""
    config = backup_config_crud.get(db=db, id=config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Backup configuration not found")
    return backup_config_crud.remove(db=db, id=config_id)