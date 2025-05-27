# app/crud/backup_config.py
from sqlalchemy.orm import Session
from typing import Any, Dict, Optional, Union

from backend.app.models import BackupConfig
from backend.app.schemas import BackupConfigCreate, BackupConfigUpdate
from .base import CRUDBase


class CRUDBackupConfig(CRUDBase[BackupConfig, BackupConfigCreate, BackupConfigUpdate]):
    """CRUD operations for backup configuration table."""

    def get_by_key(self, db: Session, *, config_key: str) -> Optional[BackupConfig]:
        """
        Get a backup configuration by key.

        Args:
            db: Database session
            config_key: Configuration key

        Returns:
            BackupConfig object or None if not found
        """
        return db.query(BackupConfig).filter(BackupConfig.config_key == config_key).first()

    def create(self, db: Session, *, obj_in: BackupConfigCreate) -> BackupConfig:
        """
        Create a new backup configuration.

        Args:
            db: Database session
            obj_in: Backup configuration creation schema

        Returns:
            Created backup configuration object
        """
        db_obj = BackupConfig(
            config_key=obj_in.config_key,
            config_value=obj_in.config_value,
            description=obj_in.description
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
            self,
            db: Session,
            *,
            db_obj: BackupConfig,
            obj_in: Union[BackupConfigUpdate, Dict[str, Any]]
    ) -> BackupConfig:
        """
        Update a backup configuration.

        Args:
            db: Database session
            db_obj: Existing backup configuration object
            obj_in: Update schema or dictionary

        Returns:
            Updated backup configuration object
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def upsert(
            self,
            db: Session,
            *,
            config_key: str,
            config_value: str,
            description: Optional[str] = None
    ) -> BackupConfig:
        """
        Upsert (insert or update) a backup configuration.

        Args:
            db: Database session
            config_key: Configuration key
            config_value: Configuration value
            description: Optional description

        Returns:
            Upserted backup configuration object
        """
        existing_config = self.get_by_key(db, config_key=config_key)

        if existing_config:
            # Update existing configuration
            return self.update(
                db,
                db_obj=existing_config,
                obj_in={
                    'config_value': config_value,
                    'description': description or existing_config.description
                }
            )
        else:
            # Create new configuration
            create_data = BackupConfigCreate(
                config_key=config_key,
                config_value=config_value,
                description=description
            )
            return self.create(db, obj_in=create_data)


# Create a singleton instance
backup_config_crud = CRUDBackupConfig(BackupConfig)