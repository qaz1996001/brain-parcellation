# MinIO Backup Metadata System

A FastAPI application to manage backup file metadata in MinIO with support for duplicate file detection and version control.

## Features

- Track backup jobs and their status
- Record file metadata and detect duplicates automatically
- Manage object versions in MinIO
- Apply retention policies to backup objects
- Add custom tags to objects
- Background task system for processing backup jobs

## System Architecture

The system consists of:

1. **FastAPI Web Service**: RESTful API for managing backups
2. **PostgreSQL Database**: Stores metadata about buckets, files, objects, versions
3. **MinIO Server**: Object storage for actual backup files
4. **Task Scheduler**: Background process to check and complete pending backup jobs

## Database Schema

The database schema includes the following tables:

- `buckets`: Tracks MinIO buckets
- `backup_jobs`: Records information about backup operations
- `files`: Stores metadata about original files
- `backup_objects`: Contains metadata about objects stored in MinIO
- `object_versions`: Tracks all versions of objects
- `tags`: Supports custom tagging for objects
- `retention_policies`: Defines rules for how long objects should be kept
- `object_retention`: Links objects to retention policies

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/minio-backup-system.git
   cd minio-backup-system
   ```

2. Start the services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Access the API documentation at http://localhost:8000/docs

## API Endpoints

The system provides endpoints for:

- Managing buckets
- Creating and monitoring backup jobs
- Tracking files and objects
- Managing object versions
- Setting tags and retention policies

## Usage Examples

### Create a Backup Job

```bash
curl -X POST "http://localhost:8000/backup-jobs/" \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "Daily Backup",
    "source_system": "/path/to/backup/source",
    "start_time": "2025-05-07T12:00:00",
    "bucket_id": 1
  }'
```

### Check Backup Job Status

```bash
curl -X GET "http://localhost:8000/backup-jobs/1"
```

### Get Latest Versions of Backup Objects

```bash
curl -X GET "http://localhost:8000/backup-objects/latest/"
```

## Configuration

Environment variables for configuration:

- `MINIO_ENDPOINT`: MinIO server endpoint (default: localhost:9000)
- `MINIO_ACCESS_KEY`: MinIO access key (default: minioadmin)
- `MINIO_SECRET_KEY`: MinIO secret key (default: minioadmin)
- `MINIO_SECURE`: Use HTTPS for MinIO (default: false)
- `SQLALCHEMY_DATABASE_URL`: PostgreSQL connection string

## License

This project is licensed under the MIT License - see the LICENSE file for details.