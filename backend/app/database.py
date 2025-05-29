from advanced_alchemy.extensions.fastapi import (
    AdvancedAlchemy,
    AsyncSessionConfig,
    SQLAlchemyAsyncConfig,
)

sqlalchemy_config = SQLAlchemyAsyncConfig(
    # connection_string="sqlite+aiosqlite:///test.sqlite",
    connection_string="postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom",
    session_config=AsyncSessionConfig(expire_on_commit=False),
    commit_mode="autocommit",
    create_all=True,
)
alchemy = AdvancedAlchemy(config=sqlalchemy_config,)
