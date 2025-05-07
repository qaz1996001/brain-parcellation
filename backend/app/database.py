# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator

# Database connection URL - you may need to adjust this for your environment
# Format: postgresql://username:password@hostname/database_name
SQLALCHEMY_DATABASE_URL: str = "postgresql://postgres_n:postgres_p@localhost:15433/minio_backup"

# Create SQLAlchemy engine with appropriate connection pool settings
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Function to get database session
def get_db() -> Generator:
    """
    Get a database session.

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()