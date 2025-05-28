# app/main.py
import os
import orjson
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.app import series

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
app.include_router(series.router, prefix="/series", tags=["series"])


@app.post("/upload_json", tags=["upload_json"])
async def run_manual_check(request: Request) -> dict[str, str]:
    """Manually trigger a check for pending backup tasks."""
    json = await request.json()
    print('request.keys()', request.keys())
    return {"message": orjson.dumps(json)}


if __name__ == "__main__":
    load_dotenv()
    UPLOAD_DATA_JSON_PORT = int(os.getenv("UPLOAD_DATA_JSON_PORT",8000))
    uvicorn.run("backend.app.main:app", host="0.0.0.0",
                port=UPLOAD_DATA_JSON_PORT, reload=False)
