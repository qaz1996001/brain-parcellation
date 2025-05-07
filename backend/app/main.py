# app/main.py
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

from app.database import engine
from app.models import Base
from app.routers import (
    buckets,
    backup_jobs,
    files,
    backup_objects,
    object_versions,
    tags,
    retention_policies,
    object_retentions
)
from app.tasks import TaskScheduler

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize task scheduler
task_scheduler = TaskScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events for the application.

    Args:
        app: FastAPI application instance
    """
    # Startup event
    asyncio.create_task(task_scheduler.start())
    yield
    # Shutdown event
    await task_scheduler.stop()


app = FastAPI(
    title="MinIO Backup Metadata API",
    description="API for managing backup metadata in MinIO with versioning support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(buckets.router, prefix="/buckets", tags=["buckets"])
app.include_router(backup_jobs.router, prefix="/backup-jobs", tags=["backup jobs"])
app.include_router(files.router, prefix="/files", tags=["files"])
app.include_router(backup_objects.router, prefix="/backup-objects", tags=["backup objects"])
app.include_router(object_versions.router, prefix="/object-versions", tags=["object versions"])
app.include_router(tags.router, prefix="/tags", tags=["tags"])
app.include_router(retention_policies.router, prefix="/retention-policies", tags=["retention policies"])
app.include_router(object_retentions.router, prefix="/object-retentions", tags=["object retentions"])


# Task management endpoints
@app.get("/tasks/pending", response_model=list[int], tags=["tasks"])
def get_pending_tasks() -> list[int]:
    """Get a list of pending task IDs."""
    return task_scheduler.get_pending_jobs()


@app.post("/tasks/manual-check", tags=["tasks"])
async def run_manual_check() -> dict[str, str]:
    """Manually trigger a check for pending backup tasks."""
    await task_scheduler.check_pending_jobs()
    return {"message": "Manual check for pending tasks initiated"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)