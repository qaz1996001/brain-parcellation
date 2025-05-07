# app/minio_client.py
from minio import Minio
from minio.error import S3Error
import io
import hashlib
import os
from typing import Dict, List, Optional, Tuple, BinaryIO, Any
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MinioClient:
    """Client for interacting with MinIO storage."""

    def __init__(
            self,
            endpoint: str = "localhost:9000",
            access_key: str = "minioadmin",
            secret_key: str = "minioadmin",
            secure: bool = False
    ):
        """
        Initialize MinIO client.

        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Use HTTPS if True
        """
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        logger.info(f"Initialized MinIO client for endpoint: {endpoint}")

    def ensure_bucket_exists(self, bucket_name: str) -> bool:
        """
        Ensure that a bucket exists, creating it if necessary.

        Args:
            bucket_name: Name of the bucket

        Returns:
            True if bucket exists or was created successfully, False otherwise
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                logger.info(f"Creating bucket: {bucket_name}")
                self.client.make_bucket(bucket_name)
                # Enable versioning on the bucket
                self.enable_versioning(bucket_name)
            return True
        except S3Error as err:
            logger.error(f"Error ensuring bucket exists: {err}")
            return False

    def enable_versioning(self, bucket_name: str) -> bool:
        """
        Enable versioning on a bucket.

        Args:
            bucket_name: Name of the bucket

        Returns:
            True if versioning was enabled successfully, False otherwise
        """
        try:
            logger.info(f"Enabling versioning on bucket: {bucket_name}")
            self.client.enable_bucket_versioning(bucket_name)
            return True
        except S3Error as err:
            logger.error(f"Error enabling versioning: {err}")
            return False

    def upload_file(
            self,
            bucket_name: str,
            object_name: str,
            file_path: str,
            content_type: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload a file to MinIO.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in MinIO (key)
            file_path: Path to the file to upload
            content_type: Content type of the file
            metadata: Metadata to attach to the object

        Returns:
            Tuple of (success, version_id, etag)
        """
        try:
            if not self.ensure_bucket_exists(bucket_name):
                return False, None, None

            # Default metadata if not provided
            if metadata is None:
                metadata = {}

            # Add timestamp to metadata
            metadata['X-Amz-Meta-Upload-Time'] = datetime.utcnow().isoformat()

            # Add file hash to metadata
            file_hash = self._calculate_file_hash(file_path)
            metadata['X-Amz-Meta-Content-Hash'] = file_hash

            # Add original file name to metadata if object_name is different
            original_file_name = os.path.basename(file_path)
            if original_file_name != os.path.basename(object_name):
                metadata['X-Amz-Meta-Original-Filename'] = original_file_name

            file_size = os.path.getsize(file_path)

            logger.info(f"Uploading file {file_path} to {bucket_name}/{object_name}")

            # Perform the upload
            result = self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
                content_type=content_type,
                metadata=metadata
            )

            logger.info(f"Upload successful: {result.etag}, version: {result.version_id}")
            return True, result.version_id, result.etag

        except S3Error as err:
            logger.error(f"Error uploading file: {err}")
            return False, None, None

    def download_file(
            self,
            bucket_name: str,
            object_name: str,
            file_path: str,
            version_id: Optional[str] = None
    ) -> bool:
        """
        Download a file from MinIO.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in MinIO (key)
            file_path: Path where the file should be saved
            version_id: Optional version ID to download a specific version

        Returns:
            True if download was successful, False otherwise
        """
        try:
            logger.info(f"Downloading {bucket_name}/{object_name} to {file_path}")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Perform the download
            if version_id:
                self.client.fget_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=file_path,
                    version_id=version_id
                )
            else:
                self.client.fget_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=file_path
                )

            logger.info(f"Download successful")
            return True

        except S3Error as err:
            logger.error(f"Error downloading file: {err}")
            return False

    def get_object_metadata(
            self,
            bucket_name: str,
            object_name: str,
            version_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an object.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in MinIO (key)
            version_id: Optional version ID to get metadata for a specific version

        Returns:
            Dictionary of metadata or None if object doesn't exist
        """
        try:
            if version_id:
                obj = self.client.stat_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    version_id=version_id
                )
            else:
                obj = self.client.stat_object(
                    bucket_name=bucket_name,
                    object_name=object_name
                )

            # Extract metadata from object
            metadata = {
                'size': obj.size,
                'etag': obj.etag,
                'last_modified': obj.last_modified,
                'content_type': obj.content_type,
                'version_id': obj.version_id
            }

            # Add custom metadata
            for key, value in obj.metadata.items():
                if key.lower().startswith('x-amz-meta-'):
                    clean_key = key[len('x-amz-meta-'):].lower()
                    metadata[clean_key] = value

            return metadata

        except S3Error as err:
            logger.error(f"Error getting object metadata: {err}")
            return None

    def list_objects(
            self,
            bucket_name: str,
            prefix: str = "",
            recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List objects in a bucket.

        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects
            recursive: Whether to list objects recursively

        Returns:
            List of object information dictionaries
        """
        try:
            objects = []
            for obj in self.client.list_objects(
                    bucket_name=bucket_name,
                    prefix=prefix,
                    recursive=recursive
            ):
                objects.append({
                    'name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag
                })

            return objects

        except S3Error as err:
            logger.error(f"Error listing objects: {err}")
            return []

    def list_object_versions(
            self,
            bucket_name: str,
            object_name: str
    ) -> List[Dict[str, Any]]:
        """
        List all versions of an object.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in MinIO (key)

        Returns:
            List of version information dictionaries
        """
        try:
            versions = []

            # MinIO doesn't have a direct API to list versions of a single object
            # We need to list all versions in the bucket and filter by object name
            for version in self.client.list_objects_v2(
                    bucket_name=bucket_name,
                    prefix=object_name,
                    include_version=True
            ):
                if version.object_name == object_name:
                    versions.append({
                        'version_id': version.version_id,
                        'last_modified': version.last_modified,
                        'etag': version.etag,
                        'size': version.size,
                        'is_latest': version.is_latest
                    })

            return versions

        except S3Error as err:
            logger.error(f"Error listing object versions: {err}")
            return []

    def delete_object(
            self,
            bucket_name: str,
            object_name: str,
            version_id: Optional[str] = None
    ) -> bool:
        """
        Delete an object or a specific version of an object.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in MinIO (key)
            version_id: Optional version ID to delete a specific version

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if version_id:
                self.client.remove_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    version_id=version_id
                )
            else:
                self.client.remove_object(
                    bucket_name=bucket_name,
                    object_name=object_name
                )

            logger.info(f"Deleted object {bucket_name}/{object_name}")
            return True

        except S3Error as err:
            logger.error(f"Error deleting object: {err}")
            return False

    def set_object_retention(
            self,
            bucket_name: str,
            object_name: str,
            retention_mode: str,
            retain_until_date: datetime,
            version_id: Optional[str] = None
    ) -> bool:
        """
        Set retention configuration for an object.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in MinIO (key)
            retention_mode: 'GOVERNANCE' or 'COMPLIANCE'
            retain_until_date: Date until which the object should be retained
            version_id: Optional version ID to set retention for a specific version

        Returns:
            True if setting retention was successful, False otherwise
        """
        try:
            if version_id:
                self.client.set_object_retention(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    retention_mode=retention_mode,
                    retain_until_date=retain_until_date,
                    version_id=version_id
                )
            else:
                self.client.set_object_retention(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    retention_mode=retention_mode,
                    retain_until_date=retain_until_date
                )

            logger.info(f"Set retention for {bucket_name}/{object_name}")
            return True

        except S3Error as err:
            logger.error(f"Error setting object retention: {err}")
            return False

    def set_object_tags(
            self,
            bucket_name: str,
            object_name: str,
            tags: Dict[str, str],
            version_id: Optional[str] = None
    ) -> bool:
        """
        Set tags for an object.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in MinIO (key)
            tags: Dictionary of tag keys and values
            version_id: Optional version ID to set tags for a specific version

        Returns:
            True if setting tags was successful, False otherwise
        """
        try:
            if version_id:
                self.client.set_object_tags(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    tags=tags,
                    version_id=version_id
                )
            else:
                self.client.set_object_tags(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    tags=tags
                )

            logger.info(f"Set tags for {bucket_name}/{object_name}")
            return True

        except S3Error as err:
            logger.error(f"Error setting object tags: {err}")
            return False

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