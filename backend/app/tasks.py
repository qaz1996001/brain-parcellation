# app/tasks.py
import asyncio
import logging
from datetime import datetime
from typing import List, Set, Dict, Any, Optional
from sqlalchemy.orm import Session
import time
import os
from threading import Lock

from backend.app.database import SessionLocal
from backend.app.crud.backup_job import backup_job_crud
from backend.app.minio_client import MinioClient
from backend.app.backup_implementation import BackupImplementation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Task scheduler for managing backup jobs.

    Attributes:
        running: Flag indicating if the scheduler is running
        pending_jobs: Set of pending job IDs
        job_lock: Lock for thread-safe operations on pending_jobs
        interval: Check interval in seconds
    """

    def __init__(self, interval: int = 60):
        """
        Initialize the task scheduler.

        Args:
            interval: Interval in seconds between checks for pending jobs
        """
        self.running: bool = False
        self.pending_jobs: Set[int] = set()
        self.job_lock: Lock = Lock()
        self.interval: int = interval
        self.minio_client: MinioClient = MinioClient(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
        )
        logger.info("Task scheduler initialized")

    async def start(self) -> None:
        """Start the task scheduler."""
        self.running = True
        logger.info("Task scheduler started")

        # Load pending jobs from database
        self._load_pending_jobs()

        while self.running:
            try:
                await self.check_pending_jobs()
                await asyncio.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in task scheduler: {e}")
                await asyncio.sleep(self.interval)

    async def stop(self) -> None:
        """Stop the task scheduler."""
        self.running = False
        logger.info("Task scheduler stopped")

    def add_job(self, job_id: int) -> None:
        """
        Add a job to the pending jobs set.

        Args:
            job_id: ID of the job to add
        """
        with self.job_lock:
            self.pending_jobs.add(job_id)
        logger.info(f"Added job {job_id} to pending jobs")

    def remove_job(self, job_id: int) -> None:
        """
        Remove a job from the pending jobs set.

        Args:
            job_id: ID of the job to remove
        """
        with self.job_lock:
            if job_id in self.pending_jobs:
                self.pending_jobs.remove(job_id)
                logger.info(f"Removed job {job_id} from pending jobs")

    def get_pending_jobs(self) -> List[int]:
        """
        Get a list of pending job IDs.

        Returns:
            List of pending job IDs
        """
        with self.job_lock:
            return list(self.pending_jobs)

    def _load_pending_jobs(self) -> None:
        """Load pending jobs from the database."""
        db = SessionLocal()
        try:
            pending_jobs = backup_job_crud.get_by_status(db, status="pending")
            with self.job_lock:
                self.pending_jobs = set(job.job_id for job in pending_jobs)
            logger.info(f"Loaded {len(self.pending_jobs)} pending jobs from database")
        finally:
            db.close()

    async def check_pending_jobs(self) -> None:
        """Check and process pending jobs."""
        if not self.pending_jobs:
            return

        logger.info(f"Checking {len(self.pending_jobs)} pending jobs")

        # Create a copy of pending jobs to avoid modification during iteration
        with self.job_lock:
            jobs_to_process = list(self.pending_jobs)

        db = SessionLocal()
        try:
            for job_id in jobs_to_process:
                # Use the adjusted get method that handles different primary key column names
                job = backup_job_crud.get(db, id=job_id)

                if not job:
                    logger.warning(f"Job {job_id} not found in database")
                    self.remove_job(job_id)
                    continue

                if job.status != "pending":
                    logger.info(f"Job {job_id} is no longer pending, status: {job.status}")
                    self.remove_job(job_id)
                    continue

                # Process the job
                try:
                    logger.info(f"Processing job {job_id}: {job.job_name}")

                    # Update job status to running
                    backup_job_crud.update(db, db_obj=job, obj_in={"status": "running"})

                    # Process the backup job
                    await self._process_backup_job(db, job)

                    # Update job status to completed
                    end_time = datetime.utcnow()
                    backup_job_crud.update(
                        db,
                        db_obj=job,
                        obj_in={
                            "status": "completed",
                            "end_time": end_time
                        }
                    )

                    logger.info(f"Job {job_id} completed successfully")

                except Exception as e:
                    logger.error(f"Error processing job {job_id}: {e}")

                    # Update job status to failed
                    backup_job_crud.update(
                        db,
                        db_obj=job,
                        obj_in={
                            "status": "failed",
                            "error_message": str(e),
                            "end_time": datetime.utcnow()
                        }
                    )

                # Remove job from pending jobs
                self.remove_job(job_id)

        finally:
            db.close()

    async def _process_backup_job(self, db: Session, job: Any) -> None:
        """
        Process a backup job.

        Args:
            db: Database session
            job: Backup job object
        """
        # Use the backup implementation to process the job
        logger.info(f"Processing backup job {job.job_id}: {job.job_name}")

        # Create a backup implementation instance
        backup_impl = BackupImplementation(self.minio_client)

        # Process the job using the implementation
        await backup_impl.process_backup_job(db, job)

        logger.info(f"Backup job {job.job_id} processing completed")