# app/main.py
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.app.routers import router
from code_ai import load_dotenv


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events for the application.

    Args:
        app: FastAPI application instance
    """
    # Startup event
    # asyncio.create_task(task_scheduler.start())
    yield
    # Shutdown event
    # await task_scheduler.stop()


app = FastAPI(
    title="SHH AI API",
    description="API for SHH AI",
    version="1.0.0",
    lifespan=lifespan,
    # root_path="/api/v1"
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
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    load_dotenv()
    UPLOAD_DATA_JSON_PORT = int(os.getenv("UPLOAD_DATA_JSON_PORT",8000))
    uvicorn.run("backend.app.main:app", host="0.0.0.0",
                port=UPLOAD_DATA_JSON_PORT, reload=False)
