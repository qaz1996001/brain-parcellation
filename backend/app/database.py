from advanced_alchemy.extensions.fastapi import (
    AdvancedAlchemy,
    AsyncSessionConfig,
    SQLAlchemyAsyncConfig,
)
from code_ai import load_dotenv
load_dotenv()
AI_APP_CONNECTION_STRING = os.getenv("AI_APP_CONNECTION_STRING","postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom")

sqlalchemy_config = SQLAlchemyAsyncConfig(
    # connection_string="sqlite+aiosqlite:///test.sqlite",
    connection_string=AI_APP_CONNECTION_STRING,
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
    return AI_APP_CONNECTION_STRING
