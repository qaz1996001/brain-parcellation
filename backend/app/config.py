"""
Application configuration using Pydantic v2 settings.

Follows best practices for configuration management with type safety
and environment variable support.
"""
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = Field(default="Medical Imaging AI API", description="Application name")
    app_description: str = Field(
        default="API for medical imaging AI processing with clean architecture",
        description="Application description"
    )
    app_version: str = Field(default="2.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    enable_docs: bool = Field(default=True, description="Enable API documentation")
    
    # Database settings
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/medical_ai",
        description="Database connection URL"
    )
    database_pool_size: int = Field(default=20, description="Database connection pool size")
    database_max_overflow: int = Field(default=30, description="Maximum overflow connections")
    database_pool_recycle: int = Field(default=3600, description="Connection recycle time in seconds")
    
    # Redis settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_cache_db: int = Field(default=1, description="Redis database for caching")
    redis_task_db: int = Field(default=2, description="Redis database for task queue")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    # Security settings
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration time")
    
    # File processing settings
    upload_dir: str = Field(default="./uploads", description="Upload directory")
    max_file_size: int = Field(default=500 * 1024 * 1024, description="Maximum file size in bytes (500MB)")
    allowed_file_extensions: List[str] = Field(
        default=[".nii", ".nii.gz", ".dcm", ".dicom"],
        description="Allowed file extensions"
    )
    
    # Processing settings
    default_depth_number: int = Field(default=5, description="Default white matter parcellation depth")
    processing_timeout: int = Field(default=3600, description="Processing timeout in seconds")
    max_concurrent_tasks: int = Field(default=5, description="Maximum concurrent processing tasks")
    
    # Model paths
    synthseg_model_path: str = Field(
        default="/models/synthseg_2.0.h5",
        description="Path to SynthSeg model"
    )
    wmh_model_path: str = Field(
        default="/models/wmh_detector.h5",
        description="Path to WMH detection model"
    )
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="APP_",
    )
    
    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
    
    def get_redis_url(self, db: Optional[int] = None) -> str:
        """Get Redis URL with specific database."""
        if db is None:
            return self.redis_url
        
        # Parse and replace database number
        base_url = self.redis_url.rsplit("/", 1)[0]
        return f"{base_url}/{db}"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to ensure settings are only loaded once.
    """
    return Settings()
