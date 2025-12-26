"""
Configuration dataclasses for backend application.

All configuration is represented as immutable, type-annotated dataclasses
following Knuth's precision principle: explicit types, boundaries, and validation.

Design Pattern: Configuration Centralization
- All environment variable reads concentrated in loader.py
- Immutable frozen dataclasses ensure no runtime modification
- Type hints provide compile-time safety
- Hierarchical structure mirrors system architecture
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class APIConfig:
    """
    API endpoint configuration.

    Contains URLs and credentials for external API communication.
    All URLs should include protocol (http:// or https://).

    Attributes:
        upload_data_url: URL for uploading data (e.g., http://localhost:8000/upload)
    """
    upload_data_url: str


@dataclass(frozen=True)
class PathConfig:
    """
    File system path configuration.

    All paths are represented as pathlib.Path objects for:
    - Type safety (Path vs str)
    - Platform independence (automatic path separator handling)
    - Validation capabilities

    Attributes:
        path_process: Directory for processing temporary files
        path_json: Directory for JSON output files
        path_log: Directory for log files
        path_root: Root directory for application data
        path_rename_dicom: Directory for renamed DICOM files
    """
    path_process: Path
    path_json: Path
    path_log: Path
    path_root: Path
    path_rename_dicom: Path


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database connection configuration.

    Supports PostgreSQL async connections via asyncpg.
    Connection string format: postgresql+asyncpg://user:password@host:port/database

    Attributes:
        connection_string: Full database connection string with credentials
    """
    connection_string: str


@dataclass(frozen=True)
class BackendConfig:
    """
    Root configuration aggregating all backend subsystems.

    This is the single source of truth for backend configuration.
    Immutable by design (frozen=True) to prevent runtime modification.

    Design Principles:
    - Knuth: Each field has explicit type and clear purpose
    - Linus (Data Structure): Good structure makes code naturally simple
    - Immutability: Configuration cannot be modified after creation

    Attributes:
        api: API endpoint configuration
        paths: File system path configuration
        database: Database connection configuration
    """
    api: APIConfig
    paths: PathConfig
    database: DatabaseConfig
