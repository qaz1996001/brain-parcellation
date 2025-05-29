# app/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import router
from .database import alchemy

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
app.include_router(router, prefix="/api/v1")
alchemy.init_app(app)