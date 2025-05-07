# app/services/backup_service.py
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Union
from sqlalchemy.orm import Session

from app.minio_client import MinioClient
from app.crud import (
    bucket_crud,
    backup_job_crud,
    file_crud,
    backup_object_crud,
    backup_config_crud
)
from app.schemas import (
    BackupJobCreate,
    FileCreate,
    BackupObjectCreate
)


class BackupService:
    """
    Service for handling backup operations.

    This service provides methods to:
    - Validate and prepare backup configurations
    - Perform file backup to MinIO
    - Manage backup metadata
    """

    def __init__(self, db: Session, minio_client: MinioClient):
        """
        Initialize the backup service.

        Args:
            db: Database session
            minio_client: MinIO client instance
        """
        self.db = db
        self.minio_client = minio_client
        self._current_configs = {}

    def _load_current_configs(self):
        """
        Load current backup configurations from the database.
        """
        configs = backup_config_crud.get_multi(self.db)
        self._current_configs = {
            config.config_key: config.config_value
            for config in configs
        }

    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a specific configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Reload configs to ensure latest values
        self._load_current_configs()
        return self._current_configs.get(key, default)

    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal representation of the MD5 hash
        """
        md5 = hashlib.md5()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)

        return md5.hexdigest()

    def backup_file(
            self,
            file_path: str,
            source_system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backup a single file to MinIO.

        Args:
            file_path: Full path to the file
            source_system: Optional source system identifier

        Returns:
            Dictionary with backup details
        """
        # Validate file existence
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        # Get default source system if not provided
        if not source_system:
            source_system = self.get_config('default_source_system', 'unknown')

        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)

        # Check for existing file with same hash
        existing_files = file_crud.get_by_hash(self.db, file_hash=file_hash)

        # Prepare file creation data
        file_stat = os.stat(file_path)
        file_create_data = FileCreate(
            file_path=os.path.relpath(file_path),
            file_name=os.path.basename(file_path),
            file_extension=os.path.splitext(file_path)[1][1:].lower(),
            original_size=file_stat.st_size,
            original_hash=file_hash,
            source_system=source_system
        )

        # Create or get existing file record
        if existing_files:
            file_db = existing_files[0]
        else:
            file_db = file_crud.create(self.db, obj_in=file_create_data)

        # Determine target bucket
        target_bucket_name = self.get_config('default_backup_bucket', 'default-backup')

        # Ensure bucket exists
        bucket = bucket_crud.get_by_name(self.db, name=target_bucket_name)
        if not bucket:
            # Create bucket if not exists
            bucket_create_data = {
                'bucket_name': target_bucket_name,
                'versioning_enabled': True,
                'description': 'Default backup bucket'
            }
            bucket = bucket_crud.create(self.db, obj_in=bucket_create_data)

        # Create backup job
        backup_job_data = BackupJobCreate(
            job_name=f"Backup-{os.path.basename(file_path)}",
            source_system=source_system,
            start_time=datetime.utcnow(),
            bucket_id=bucket.bucket_id,
            status='pending',
            total_files=1,
            total_size=file_stat.st_size
        )
        backup_job = backup_job_crud.create(self.db, obj_in=backup_job_data)

        # Upload to MinIO
        object_key = f"{backup_job.job_id}/{os.path.basename(file_path)}"
        upload_success, version_id, etag = self.minio_client.upload_file(
            bucket_name=target_bucket_name,
            object_name=object_key,
            file_path=file_path
        )

        if not upload_success:
            raise RuntimeError(f"Failed to upload file {file_path} to MinIO")

        # Create backup object record
        backup_object_data = BackupObjectCreate(
            file_id=file_db.file_id,
            job_id=backup_job.job_id,
            bucket_id=bucket.bucket_id,
            object_key=object_key,
            storage_size=file_stat.st_size,
            latest_version=True
        )
        backup_object = backup_object_crud.create(self.db, obj_in=backup_object_data)

        # Update backup job status
        backup_job_crud.update(
            self.db,
            db_obj=backup_job,
            obj_in={'status': 'completed', 'end_time': datetime.utcnow()}
        )

        return {
            'file_id': file_db.file_id,
            'backup_job_id': backup_job.job_id,
            'backup_object_id': backup_object.object_id,
            'minio_bucket': target_bucket_name,
            'object_key': object_key,
            'version_id': version_id
        }

    def backup_directory(
            self,
            directory_path: str,
            source_system: Optional[str] = None,
            recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Backup an entire directory to MinIO.

        Args:
            directory_path: Path to the directory to backup
            source_system: Optional source system identifier
            recursive: Whether to backup subdirectories

        Returns:
            Dictionary with backup summary
        """
        # Validate directory existence
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")

        # Get default source system if not provided
        if not source_system:
            source_system = self.get_config('default_source_system', 'unknown')

        # Determine target bucket
        target_bucket_name = self.get_config('default_backup_bucket', 'default-backup')

        # Ensure bucket exists
        bucket = bucket_crud.get_by_name(self.db, name=target_bucket_name)
        if not bucket:
            # Create bucket if not exists
            bucket_create_data = {
                'bucket_name': target_bucket_name,
                'versioning_enabled': True,
                'description': 'Default backup bucket'
            }
            bucket = bucket_crud.create(self.db, obj_in=bucket_create_data)

        # Create backup job
        backup_job_data = BackupJobCreate(
            job_name=f"Backup-{os.path.basename(directory_path)}",
            source_system=source_system,
            start_time=datetime.utcnow(),
            bucket_id=bucket.bucket_id,
            status='pending'
        )
        backup_job = backup_job_crud.create(self.db, obj_in=backup_job_data)

        # Track backup statistics
        total_files = 0
        total_size = 0
        failed_files = []

        # Walk through directory
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)

                try:
                    # Backup individual file
                    file_backup_result = self.backup_file(
                        file_path,
                        source_system=source_system
                    )

                    total_files += 1
                    total_size += os.path.getsize(file_path)

                except Exception as e:
                    # Log failed files
                    failed_files.append({
                        'path': file_path,
                        'error': str(e)
                    })

            # Stop if not recursive
            if not recursive:
                break

        # Update backup job with final statistics
        backup_job_crud.update(
            self.db,
            db_obj=backup_job,
            obj_in={
                'status': 'completed' if not failed_files else 'partial',
                'end_time': datetime.utcnow(),
                'total_files': total_files,
                'total_size': total_size
            }
        )

        return {
            'job_id': backup_job.job_id,
            'total_files': total_files,
            'total_size': total_size,
            'failed_files': failed_files,
            'minio_bucket': target_bucket_name
        }