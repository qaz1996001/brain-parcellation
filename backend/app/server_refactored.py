"""
Refactored FastAPI server following best practices.

This module provides a clean, async-first API server with proper
error handling, dependency injection, and configuration management.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import get_settings
from .database import init_database, close_database
from .cache import init_cache, close_cache
from .routers import api_router
from .middleware import (
    SecurityMiddleware,
    LoggingMiddleware,
    PerformanceMiddleware,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events following FastAPI best practices.
    """
    # Startup
    logger.info("Starting application...")
    
    try:
        # Initialize database
        await init_database(settings.database_url)
        logger.info("Database initialized")
        
        # Initialize cache
        await init_cache(settings.redis_url)
        logger.info("Cache initialized")
        
        # Any other startup tasks
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    try:
        # Close database connections
        await close_database()
        logger.info("Database connections closed")
        
        # Close cache connections
        await close_cache()
        logger.info("Cache connections closed")
        
        logger.info("Application shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
    lifespan=lifespan,
)


# Add middleware in correct order (outermost first)
# 1. CORS (needs to be early for preflight requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Process-Time"],
)

# 2. Security headers
app.add_middleware(SecurityMiddleware)

# 3. Logging
app.add_middleware(LoggingMiddleware)

# 4. Performance monitoring
app.add_middleware(PerformanceMiddleware)


# Global exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Invalid input data",
            "details": exc.errors(),
            "path": request.url.path,
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "VALUE_ERROR",
            "message": str(exc),
            "path": request.url.path,
        }
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "NOT_FOUND",
            "message": "The requested resource was not found",
            "path": request.url.path,
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle internal server errors."""
    logger.error(f"Internal error on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "path": request.url.path,
        }
    )


# Root endpoints
@app.get("/", tags=["System"])
async def root() -> Dict[str, Any]:
    """API root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs" if settings.enable_docs else None,
    }


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns the current health status of the application and its dependencies.
    """
    health_status = {
        "status": "healthy",
        "timestamp": settings.get_timestamp(),
        "version": settings.app_version,
        "database": "unknown",
        "redis": "unknown",
    }
    
    # Check database health
    try:
        from .database import check_database_health
        db_healthy = await check_database_health()
        health_status["database"] = "healthy" if db_healthy else "unhealthy"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
    
    # Check Redis health
    try:
        from .cache import check_cache_health
        cache_healthy = await check_cache_health()
        health_status["redis"] = "healthy" if cache_healthy else "unhealthy"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"
    
    # Determine overall health
    if health_status["database"] != "healthy" or health_status["redis"] != "healthy":
        health_status["status"] = "degraded"
    
    return health_status


# Include API routes
app.include_router(api_router, prefix="/api/v1")
