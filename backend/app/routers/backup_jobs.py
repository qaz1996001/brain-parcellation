# app/routers/backup_jobs.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.schemas import BackupJobCreate, BackupJobResponse, BackupJobUpdate
from app.crud import backup_job_crud, bucket_crud
from app.tasks import TaskScheduler

# Get reference to the task scheduler
# In a real implementation, you'd need to access the actual instance from main.py
task_scheduler = TaskScheduler()

router = APIRouter()


@router.post("/", response_model=BackupJobResponse, status_code=201)
def create_backup_job(
        job: BackupJobCreate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
) -> BackupJobResponse:
    """Create a new backup job."""
    # Verify that the bucket exists
    bucket = bucket_crud.get(db=db, id=job.bucket_id)
    if not bucket:
        raise HTTPException(status_code=404, detail="Bucket not found")

    # Create the backup job
    new_job = backup_job_crud.create(db=db, obj_in=job)

    # Add job to the task scheduler
    task_scheduler.add_job(new_job.job_id)

    return new_job


@router.get("/", response_model=List[BackupJobResponse])
def read_backup_jobs(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        db: Session = Depends(get_db)
) -> List[BackupJobResponse]:
    """Get a list of backup jobs with optional status filter."""
    if status:
        return backup_job_crud.get_by_status(db=db, status=status, skip=skip, limit=limit)
    return backup_job_crud.get_multi(db=db, skip=skip, limit=limit)


@router.get("/{job_id}", response_model=BackupJobResponse)
def read_backup_job(
        job_id: int,
        db: Session = Depends(get_db)
) -> BackupJobResponse:
    """Get a specific backup job by ID."""
    job = backup_job_crud.get(db=db, id=job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Backup job not found")
    return job


@router.put("/{job_id}", response_model=BackupJobResponse)
def update_backup_job(
        job_id: int,
        job: BackupJobUpdate,
        db: Session = Depends(get_db)
) -> BackupJobResponse:
    """Update a specific backup job."""
    db_job = backup_job_crud.get(db=db, id=job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="Backup job not found")

    # If bucket_id is being updated, verify that the bucket exists
    if job.bucket_id:
        bucket = bucket_crud.get(db=db, id=job.bucket_id)
        if not bucket:
            raise HTTPException(status_code=404, detail="Bucket not found")

    return backup_job_crud.update(db=db, db_obj=db_job, obj_in=job)


@router.delete("/{job_id}", response_model=BackupJobResponse)
def delete_backup_job(
        job_id: int,
        db: Session = Depends(get_db)
) -> BackupJobResponse:
    """Delete a specific backup job."""
    job = backup_job_crud.get(db=db, id=job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Backup job not found")

    # Remove job from task scheduler if it's pending
    task_scheduler.remove_job(job_id)

    return backup_job_crud.remove(db=db, id=job_id)


@router.post("/{job_id}/retry", response_model=BackupJobResponse)
def retry_backup_job(
        job_id: int,
        db: Session = Depends(get_db)
) -> BackupJobResponse:
    """Retry a failed backup job."""
    job = backup_job_crud.get(db=db, id=job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Backup job not found")

    if job.status != "failed":
        raise HTTPException(status_code=400, detail="Only failed jobs can be retried")

    # Update job status to pending
    updated_job = backup_job_crud.update(
        db=db,
        db_obj=job,
        obj_in={"status": "pending", "error_message": None}
    )

    # Add job back to task scheduler
    task_scheduler.add_job(job_id)

    return updated_job