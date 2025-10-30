# 資料庫互動模式審查報告

## 審查標準
基於專案的資料庫互動規則進行評估

## 1. 非同步操作審查

### ❌ 違規：硬編碼的連接字串
```python
# backend/app/database.py
sqlalchemy_config = SQLAlchemyAsyncConfig(
    connection_string="postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom",
    session_config=AsyncSessionConfig(expire_on_commit=False),
    commit_mode="autocommit",
    create_all=True,
)
```

**問題分析**：
1. 🔴 **安全風險**：密碼明文存儲
2. 🔴 **環境耦合**：無法在不同環境使用
3. 🔴 **缺少連接池配置**：效能問題
4. 🟡 **autocommit 模式**：可能導致事務問題

### ✅ 正確的資料庫配置
```python
# config/database.py
import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, QueuePool
from pydantic import BaseSettings, SecretStr

class DatabaseSettings(BaseSettings):
    """資料庫設定"""
    # 連接參數
    host: str = "localhost"
    port: int = 5432
    username: str
    password: SecretStr
    database: str
    
    # 連接池設定
    pool_size: int = 20
    max_overflow: int = 30
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    echo: bool = False
    
    class Config:
        env_prefix = "DB_"
        env_file = ".env"
    
    @property
    def async_url(self) -> str:
        """構建非同步連接 URL"""
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.username}:{pwd}@{self.host}:{self.port}/{self.database}"

# 初始化設定
settings = DatabaseSettings()

# 創建非同步引擎
engine = create_async_engine(
    settings.async_url,
    echo=settings.echo,
    pool_size=settings.pool_size,
    max_overflow=settings.max_overflow,
    pool_pre_ping=settings.pool_pre_ping,
    pool_recycle=settings.pool_recycle,
    poolclass=QueuePool,
)

# 創建會話工廠
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# 依賴注入函數
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """取得資料庫會話"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

## 2. CRUD 操作模式審查

### ❌ 當前問題：缺少標準化的 CRUD 模式

### ✅ 建議的 CRUD 基礎類別
```python
# crud/base.py
from typing import Generic, TypeVar, Type, Optional, List, Any, Dict
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeMeta

ModelType = TypeVar("ModelType", bound=DeclarativeMeta)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """基礎 CRUD 操作類別"""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def get(
        self, db: AsyncSession, id: Any
    ) -> Optional[ModelType]:
        """根據 ID 取得單一記錄"""
        result = await db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_multi(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """取得多筆記錄"""
        query = select(self.model)
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.where(getattr(self.model, key) == value)
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    
    async def create(
        self, db: AsyncSession, *, obj_in: CreateSchemaType
    ) -> ModelType:
        """創建記錄"""
        db_obj = self.model(**obj_in.dict())
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: ModelType,
        obj_in: UpdateSchemaType
    ) -> ModelType:
        """更新記錄"""
        update_data = obj_in.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def delete(
        self, db: AsyncSession, *, id: Any
    ) -> bool:
        """刪除記錄"""
        result = await db.execute(
            select(self.model).where(self.model.id == id)
        )
        db_obj = result.scalar_one_or_none()
        
        if not db_obj:
            return False
        
        await db.delete(db_obj)
        await db.commit()
        return True
    
    async def count(
        self,
        db: AsyncSession,
        *,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """計算記錄數量"""
        query = select(func.count()).select_from(self.model)
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.where(getattr(self.model, key) == value)
        
        result = await db.execute(query)
        return result.scalar()
```

## 3. 事務管理審查

### ❌ 當前問題：缺少明確的事務管理

### ✅ 建議的事務管理模式
```python
# services/transaction.py
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

class TransactionManager:
    """事務管理器"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """明確的事務上下文"""
        async with self.session.begin():
            yield self.session
    
    @asynccontextmanager
    async def savepoint(self) -> AsyncGenerator[AsyncSession, None]:
        """保存點管理"""
        async with self.session.begin_nested():
            yield self.session

# 使用範例
async def transfer_study(
    db: AsyncSession,
    study_id: int,
    from_user_id: int,
    to_user_id: int
) -> bool:
    """轉移 Study 所有權 - 事務範例"""
    manager = TransactionManager(db)
    
    async with manager.transaction():
        # 檢查 study 存在
        study = await get_study_by_id(db, study_id)
        if not study:
            raise ValueError("Study 不存在")
        
        # 檢查權限
        if study.owner_id != from_user_id:
            raise ValueError("無權轉移此 Study")
        
        # 更新所有權
        study.owner_id = to_user_id
        
        # 記錄轉移日誌
        log = TransferLog(
            study_id=study_id,
            from_user_id=from_user_id,
            to_user_id=to_user_id,
            transferred_at=datetime.utcnow()
        )
        db.add(log)
        
        # 事務會自動提交或回滾
    
    return True
```

## 4. 查詢最佳化審查

### ❌ 發現的 N+1 查詢問題
```python
# 問題代碼（推測）
studies = await get_all_studies(db)
for study in studies:
    # N+1 問題：每個 study 都會觸發新查詢
    study.series = await get_series_by_study_id(db, study.id)
```

### ✅ 使用預載入解決 N+1
```python
from sqlalchemy.orm import selectinload, joinedload

async def get_studies_with_series(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100
) -> List[Study]:
    """預載入 series 避免 N+1"""
    result = await db.execute(
        select(Study)
        .options(selectinload(Study.series))  # 預載入關聯
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().unique().all()

# 更複雜的預載入
async def get_study_details(
    db: AsyncSession,
    study_id: int
) -> Optional[Study]:
    """取得完整的 Study 詳情"""
    result = await db.execute(
        select(Study)
        .options(
            selectinload(Study.series).selectinload(Series.instances),
            selectinload(Study.patient),
            selectinload(Study.reports)
        )
        .where(Study.id == study_id)
    )
    return result.scalar_one_or_none()
```

## 5. 連接池監控

### ❌ 缺少連接池監控機制

### ✅ 建議的監控實作
```python
# monitoring/database.py
from fastapi import APIRouter
from sqlalchemy.pool import QueuePool
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health/database")
async def database_health(db: AsyncSession = Depends(get_db)):
    """資料庫健康檢查"""
    try:
        # 執行簡單查詢
        await db.execute(text("SELECT 1"))
        
        # 取得連接池狀態
        pool = db.bind.pool
        if isinstance(pool, QueuePool):
            pool_status = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total": pool.checkedout() + pool.checkedin()
            }
        else:
            pool_status = {"type": "non-queue-pool"}
        
        return {
            "status": "healthy",
            "pool": pool_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# 定期記錄連接池狀態
async def log_pool_status():
    """記錄連接池狀態"""
    while True:
        await asyncio.sleep(60)  # 每分鐘記錄一次
        
        pool = engine.pool
        if isinstance(pool, QueuePool):
            logger.info(
                "Connection pool status",
                extra={
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "total": pool.checkedout() + pool.checkedin()
                }
            )
```

## 6. 批次操作最佳化

### ✅ 建議的批次操作模式
```python
# crud/batch.py
from typing import List, TypeVar, Type
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

T = TypeVar("T")

class BatchOperations:
    """批次操作工具"""
    
    @staticmethod
    async def bulk_insert(
        db: AsyncSession,
        model: Type[T],
        data: List[dict]
    ) -> None:
        """批次插入"""
        if not data:
            return
        
        await db.execute(insert(model), data)
        await db.commit()
    
    @staticmethod
    async def bulk_upsert(
        db: AsyncSession,
        model: Type[T],
        data: List[dict],
        index_elements: List[str]
    ) -> None:
        """批次插入或更新（PostgreSQL）"""
        if not data:
            return
        
        stmt = pg_insert(model).values(data)
        stmt = stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_={
                key: stmt.excluded[key]
                for key in data[0].keys()
                if key not in index_elements
            }
        )
        
        await db.execute(stmt)
        await db.commit()
```

## 資料庫互動評分

| 類別 | 評分 | 說明 |
|-----|------|------|
| 非同步操作 | 6/10 | 使用非同步但配置不當 |
| 連接池管理 | 2/10 | 缺少適當的連接池配置 |
| 事務管理 | 3/10 | 缺少明確的事務控制 |
| 查詢最佳化 | 3/10 | 可能存在 N+1 問題 |
| 錯誤處理 | 2/10 | 缺少資料庫特定錯誤處理 |
| 監控 | 1/10 | 沒有連接池監控 |

## 總體評分：2.8/10

## 改進建議

### 立即修復（P0）
1. 移除硬編碼的資料庫密碼
2. 實施環境變數配置
3. 添加基本錯誤處理

### 短期改進（P1）
1. 實施標準化 CRUD 模式
2. 添加連接池配置
3. 解決 N+1 查詢問題

### 長期優化（P2）
1. 實施完整的事務管理
2. 添加連接池監控
3. 實施批次操作最佳化
