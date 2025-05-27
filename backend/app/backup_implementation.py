# app/backup_implementation.py
import os
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import shutil
from pathlib import Path
import asyncio
from sqlalchemy.orm import Session

from backend.app.minio_client import MinioClient
from backend.app.models import BackupJob, File, BackupObject, ObjectVersion
from backend.app.crud import (
    file_crud, backup_object_crud, object_version_crud,
    backup_job_crud, tag_crud
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BackupImplementation:
    """Implementation of backup job processing logic."""

    def __init__(self, minio_client: MinioClient):
        """
        Initialize the backup implementation.

        Args:
            minio_client: MinIO client instance
        """
        self.minio_client = minio_client

    async def process_backup_job(self, db: Session, job: BackupJob) -> None:
        """
        Process a backup job.

        Args:
            db: Database session
            job: Backup job object
        """
        logger.info(f"Processing backup job {job.job_id}: {job.job_name}")

        # Ensure the bucket exists
        bucket_name = f"backup-{job.bucket_id}"
        if not self.minio_client.ensure_bucket_exists(bucket_name):
            raise Exception(f"Failed to ensure bucket exists: {bucket_name}")

        # Get the source system path
        source_path = job.source_system
        if not os.path.exists(source_path):
            raise Exception(f"Source path does not exist: {source_path}")

        # Scan source directory for files
        files_to_backup = self._scan_directory(source_path)
        logger.info(f"Found {len(files_to_backup)} files to backup")

        # Update job with total files and size
        total_size = sum(file_info['size'] for file_info in files_to_backup)
        backup_job_crud.update(
            db,
            db_obj=job,
            obj_in={
                "total_files": len(files_to_backup),
                "total_size": total_size
            }
        )

        # Process each file
        processed_files = 0
        processed_size = 0

        for file_info in files_to_backup:
            try:
                await self._process_file(db, job, file_info, bucket_name)
                processed_files += 1
                processed_size += file_info['size']

                # Update job progress periodically
                if processed_files % 10 == 0:
                    backup_job_crud.update(
                        db,
                        db_obj=job,
                        obj_in={
                            "total_files": len(files_to_backup),
                            "total_size": total_size
                        }
                    )

            except Exception as e:
                logger.error(f"Error processing file {file_info['path']}: {e}")
                # Continue with next file

        logger.info(f"Processed {processed_files} files, total size: {processed_size} bytes")

    def _scan_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Scan a directory recursively and get information about files.

        Args:
            directory: Directory path to scan

        Returns:
            List of file information dictionaries
        """
        files = []

        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)

                # Skip symbolic links
                if os.path.islink(file_path):
                    continue

                # Get file information
                try:
                    file_stat = os.stat(file_path)
                    file_hash = self._calculate_file_hash(file_path)

                    files.append({
                        'path': file_path,
                        'rel_path': os.path.relpath(file_path, directory),
                        'name': filename,
                        'extension': os.path.splitext(filename)[1][1:].lower(),
                        'size': file_stat.st_size,
                        'modified_time': datetime.fromtimestamp(file_stat.st_mtime),
                        'hash': file_hash
                    })
                except Exception as e:
                    logger.error(f"Error getting file information for {file_path}: {e}")

        return files

    async def _process_file(
            self,
            db: Session,
            job: BackupJob,
            file_info: Dict[str, Any],
            bucket_name: str
    ) -> None:
        """
        Process a single file for backup.

        Args:
            db: Database session
            job: Backup job object
            file_info: File information dictionary
            bucket_name: MinIO bucket name
        """
        # Check if file exists in database by hash
        existing_files = file_crud.get_by_hash(db, file_hash=file_info['hash'])

        # Create file record if not exists
        if not existing_files:
            file_db = file_crud.create(
                db,
                obj_in={
                    'file_path': file_info['rel_path'],
                    'file_name': file_info['name'],
                    'file_extension': file_info['extension'],
                    'original_size': file_info['size'],
                    'original_hash': file_info['hash'],
                    'source_system': job.source_system
                }
            )
            is_new_file = True
        else:
            # Use the first matching file record
            file_db = existing_files[0]
            is_new_file = False

        # Prepare object key (path in MinIO)
        object_key = f"{job.job_id}/{file_info['rel_path']}"

        # Upload file to MinIO
        success, version_id, etag = self.minio_client.upload_file(
            bucket_name=bucket_name,
            object_name=object_key,
            file_path=file_info['path'],
            metadata={
                'file_hash': file_info['hash'],
                'job_id': str(job.job_id),
                'file_id': str(file_db.file_id)
            }
        )

        if not success or not version_id:
            raise Exception(f"Failed to upload file to MinIO: {file_info['path']}")

        # Create or update backup object record
        existing_object = backup_object_crud.get_by_object_key(
            db,
            object_key=object_key,
            bucket_id=job.bucket_id
        )

        if existing_object and not is_new_file:
            # If object exists and file hash matches, create a new version
            # First, mark existing object as not latest
            backup_object_crud.update(
                db,
                db_obj=existing_object,
                obj_in={'latest_version': False}
            )

            # Create new backup object
            backup_obj = backup_object_crud.create(
                db,
                obj_in={
                    'file_id': file_db.file_id,
                    'job_id': job.job_id,
                    'bucket_id': job.bucket_id,
                    'object_key': object_key,
                    'storage_size': file_info['size'],
                    'latest_version': True,
                    'is_compressed': False,
                    'compression_ratio': None
                }
            )
        else:
            # Create new backup object
            backup_obj = backup_object_crud.create(
                db,
                obj_in={
                    'file_id': file_db.file_id,
                    'job_id': job.job_id,
                    'bucket_id': job.bucket_id,
                    'object_key': object_key,
                    'storage_size': file_info['size'],
                    'latest_version': True,
                    'is_compressed': False,
                    'compression_ratio': None
                }
            )

        # Create object version
        object_version_crud.create(
            db,
            obj_in={
                'object_id': backup_obj.object_id,
                'minio_version_id': version_id,
                'version_number': 1,  # This will be calculated by the create method
                'is_active': True,
                'is_delete_marker': False
            }
        )

        # Add tags
        tag_crud.create(
            db,
            obj_in={
                'object_id': backup_obj.object_id,
                'key': 'backup_date',
                'value': datetime.utcnow().isoformat()
            }
        )

        tag_crud.create(
            db,
            obj_in={
                'object_id': backup_obj.object_id,
                'key': 'source_path',
                'value': file_info['rel_path']
            }
        )

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