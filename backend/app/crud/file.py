# app/crud/file.py
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional, Union

from backend.app.models import File
from backend.app.schemas import FileCreate, FileUpdate
from .base import CRUDBase


class CRUDFile(CRUDBase[File, FileCreate, FileUpdate]):
    """CRUD operations for file table."""

    def get_by_hash(self, db: Session, *, file_hash: str) -> List[File]:
        """
        Get files by content hash.

        Args:
            db: Database session
            file_hash: Content hash value

        Returns:
            List of file objects with matching hash
        """
        return db.query(File).filter(File.original_hash == file_hash).all()

    def get_by_name_and_path(
            self, db: Session, *, file_name: str, file_path: str
    ) -> Optional[File]:
        """
        Get a file by name and path.

        Args:
            db: Database session
            file_name: File name
            file_path: File path

        Returns:
            File object or None if not found
        """
        return db.query(File).filter(
            File.file_name == file_name,
            File.path_nii == file_path
        ).first()

    def create(self, db: Session, *, obj_in: FileCreate) -> File:
        """
        Create a new file record.

        Args:
            db: Database session
            obj_in: File creation schema

        Returns:
            Created file object
        """
        db_obj = File(
            file_path=obj_in.path_nii,
            file_name=obj_in.file_name,
            file_extension=obj_in.file_extension,
            original_size=obj_in.original_size,
            original_hash=obj_in.original_hash,
            source_system=obj_in.source_system,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
            self,
            db: Session,
            *,
            db_obj: File,
            obj_in: Union[FileUpdate, Dict[str, Any]]
    ) -> File:
        """
        Update a file record.

        Args:
            db: Database session
            db_obj: Existing file object
            obj_in: Update schema or dictionary

        Returns:
            Updated file object
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def get_duplicate_files(self, db: Session) -> List[Dict[str, Any]]:
        """
        Get information about duplicate files.

        Args:
            db: Database session

        Returns:
            List of dictionaries with hash and count of duplicates
        """
        # Use the duplicate_files view
        return db.execute("""
            SELECT original_hash, COUNT(*) as duplication_count
            FROM files
            GROUP BY original_hash
            HAVING COUNT(*) > 1
        """).fetchall()


# Create a singleton instance
file_crud = CRUDFile(File)