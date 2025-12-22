from advanced_alchemy.extensions.fastapi import (
    AdvancedAlchemy,
    AsyncSessionConfig,
    SQLAlchemyAsyncConfig,
)

# Environment configuration support
from backend.app.config import get_environment, get_config

# Get environment-specific configuration
_ENV = get_environment()
_CONFIG = get_config()

# Database connection string - environment-aware
# In production: use production database
# In testing: use testing database (isolated from production)
_DB_NAME_SUFFIX = "" if _ENV == "production" else "_testing"
_CONNECTION_STRING = f"postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom{_DB_NAME_SUFFIX}"

sqlalchemy_config = SQLAlchemyAsyncConfig(
    # connection_string="sqlite+aiosqlite:///test.sqlite",
    connection_string=_CONNECTION_STRING,
    session_config=AsyncSessionConfig(expire_on_commit=False),
    commit_mode="autocommit",
    create_all=True,
)
alchemy = AdvancedAlchemy(config=sqlalchemy_config,)


def get_db_connection_string() -> str:
    """
    Get the current database connection string for the active environment.

    Returns:
        Database connection string

    Examples:
        >>> import os
        >>> os.environ["ENV"] = "production"
        >>> "dicom_testing" in get_db_connection_string()
        False

        >>> os.environ["ENV"] = "testing"
        >>> "dicom_testing" in get_db_connection_string()
        True
    """
    return _CONNECTION_STRING
