from typing import Annotated, Generator

from sqlalchemy.orm import Session
from fastapi import Depends

from app.core.db import engine,SessionLocal


# def get_db() -> Generator[Session, None, None]:
#     with Session(engine) as session:
#         yield session

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


SessionDep = Annotated[Session, Depends(get_db)]
