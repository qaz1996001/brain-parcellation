import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
# import code_ai
from code_ai.model import TaskModel,Base


if __name__ == '__main__':
    from sqlalchemy import create_engine
    # from sqlalchemy.orm import sessionmaker

    # 用create_engine對這個URL_DATABASE建立一個引擎
    engine = create_engine('postgresql+psycopg2://postgres_n:postgres_p@127.0.0.1:15433/db_name',
                           pool_recycle=3600, pool_size=10, max_overflow=10)
    Base.metadata.create_all(engine)
    # 使用sessionmaker來與資料庫建立一個對話，記得要bind=engine，這才會讓專案和資料庫連結
    # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, )

