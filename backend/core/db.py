from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.core.config import settings

# from sqlmodel import Session,create_engine,select




# make sure all SQLModel models are imported (app.models) before initializing DB
# otherwise, SQLModel might fail to initialize relationships properly
# for more details: https://github.com/tiangolo/full-stack-fastapi-template/issues/28

# 用create_engine對這個URL_DATABASE建立一個引擎
engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))

# 使用sessionmaker來與資料庫建立一個對話，記得要bind=engine，這才會讓專案和資料庫連結
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 創建SQLAlchemy的一個class，然後在其它地方使用
Base = declarative_base()