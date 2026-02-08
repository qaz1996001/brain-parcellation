# SDS - 軟體設計規格書 (Software Design Specification)

## Brain Parcellation System - Database Schema & Queue Redesign

> 版本: 2.0 | 日期: 2026-02-08
> 依據: Martin Fowler AI Programming Guide v2

---

## 1. 架構概覽

### 1.1 分層架構 (Fowler PoEAA)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                          │
│          FastAPI Routers (REST API Endpoints)                   │
│    /api/v1/studies  /api/v1/tasks  /api/v1/pipelines           │
├─────────────────────────────────────────────────────────────────┤
│                      Domain Layer                               │
│   Services: StudyService, TaskService, PipelineService          │
│   State Machine: StudyStateMachine, TaskStateMachine            │
│   Events: EventPublisher (Observer Pattern)                     │
├─────────────────────────────────────────────────────────────────┤
│                    Data Source Layer                             │
│   Repositories: StudyRepository, TaskRepository, etc.           │
│   Models: Study, Series, Pipeline, Task, TaskResult, EventLog   │
│   ORM: SQLAlchemy 2.0 (Async)                                  │
├─────────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                          │
│   Queue: Funboost (RabbitMQ Broker)                             │
│   Database: PostgreSQL + AsyncPG                                │
│   Cache: Redis (Heartbeat, RPC Results)                         │
│   Storage: NFS/Local Filesystem (DICOM, NIfTI files)            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 設計模式應用

| 模式 | 應用位置 | 目的 |
|------|---------|------|
| **Repository** (Data Mapper) | Data Source Layer | 資料存取抽象，隔離 ORM 細節 |
| **Factory Method** | PipelineFactory | 根據 InferenceEnum 動態建立推論管線 |
| **Strategy** | InferenceStrategy | 各推論演算法可互換 (Aneurysm, WMH, CMB...) |
| **State** | StudyStateMachine, TaskStateMachine | 明確管理狀態轉換 |
| **Observer** | EventPublisher | 狀態變更時通知多個監聽者 (DB Log, Queue, Notification) |
| **Command** | TaskCommand | 將推論請求封裝為物件，支援佇列化與重試 |
| **Chain of Responsibility** | PreprocessChain | SynthSeg 前處理管線 (resample → synthseg → process) |
| **Facade** | InferenceFacade | 簡化推論子系統的呼叫介面 |
| **Decorator** | @with_retry, @with_db_session | 橫切關注點 (重試、資料庫 Session 管理) |

---

## 2. 資料庫 Schema 設計

### 2.1 ER 圖 (Entity-Relationship)

```
┌──────────────┐       ┌──────────────┐       ┌──────────────────┐
│    study     │       │    series    │       │    pipeline      │
├──────────────┤       ├──────────────┤       ├──────────────────┤
│ id (UUID) PK │──┐    │ id (UUID) PK │       │ id (UUID) PK     │
│ study_uid    │  │    │ study_id FK  │──┐    │ name (unique)    │
│ study_id     │  │    │ series_uid   │  │    │ display_name     │
│ status       │  └──1:N│ series_type  │  │    │ required_series  │
│ dicom_path   │       │ status       │  │    │ cmd_template     │
│ nifti_path   │       │ dicom_path   │  │    │ active           │
│ created_at   │       │ nifti_path   │  │    │ created_at       │
│ updated_at   │       │ created_at   │  │    └──────────────────┘
└──────────────┘       │ updated_at   │  │             │
       │               └──────────────┘  │             │
       │                                 │             │
       │    ┌────────────────────────────┘             │
       │    │                                          │
       │    │    ┌──────────────────┐                   │
       │    │    │      task        │                   │
       │    │    ├──────────────────┤                   │
       │    │    │ id (UUID) PK     │                   │
       └──1:N───│ study_id FK      │                   │
            │   │ pipeline_id FK   │───────────────N:1──┘
            │   │ status           │
            │   │ priority         │
            │   │ input_files JSON │
            │   │ output_files JSON│
            │   │ params JSON      │
            │   │ queued_at        │
            │   │ started_at       │
            │   │ completed_at     │
            │   │ created_at       │
            │   │ updated_at       │
            │   └──────────────────┘
            │            │
            │            │ 1:N
            │            ▼
            │   ┌──────────────────┐
            │   │   task_result    │
            │   ├──────────────────┤
            │   │ id (UUID) PK     │
            │   │ task_id FK       │
            │   │ attempt_number   │
            │   │ success          │
            │   │ stdout TEXT      │
            │   │ stderr TEXT      │
            │   │ exception TEXT   │
            │   │ time_cost FLOAT  │
            │   │ host_name        │
            │   │ process_id       │
            │   │ funboost_task_id │
            │   │ created_at       │
            │   └──────────────────┘
            │
            │   ┌──────────────────┐
            │   │   event_log      │
            │   ├──────────────────┤
            │   │ id (UUID) PK     │
            │   │ entity_type      │  ← 'study' | 'series' | 'task'
            │   │ entity_id        │
            │   │ from_status      │
            │   │ to_status        │
            │   │ operator         │  ← 'system' | 'user' | 'scheduler'
            │   │ metadata JSON    │
            │   │ created_at       │
            │   └──────────────────┘
```

### 2.2 完整 Schema 定義

#### 2.2.1 `study` 表

```sql
CREATE TABLE study (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_uid   VARCHAR(128) NOT NULL UNIQUE,  -- Orthanc Study ID
    study_id    VARCHAR(128),                  -- Rename ID (e.g., 14914694_20220905_MR_21109050071)
    status      VARCHAR(32) NOT NULL DEFAULT 'NEW',
    dicom_path  TEXT,                          -- Raw DICOM 路徑
    nifti_path  TEXT,                          -- 轉換後 NIfTI 路徑
    metadata    JSONB,                         -- 額外 DICOM metadata
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 索引
CREATE INDEX idx_study_study_uid ON study(study_uid);
CREATE INDEX idx_study_study_id ON study(study_id);
CREATE INDEX idx_study_status ON study(status);
CREATE INDEX idx_study_created_at ON study(created_at);

-- Study 狀態: NEW → TRANSFERRING → TRANSFERRED → CONVERTING → CONVERTED
--            → INFERENCE_READY → INFERENCE_RUNNING → INFERENCE_COMPLETE
--            → RESULTS_SENT | FAILED
```

#### 2.2.2 `series` 表

```sql
CREATE TABLE series (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id    UUID NOT NULL REFERENCES study(id) ON DELETE CASCADE,
    series_uid  VARCHAR(128) NOT NULL,         -- Orthanc Series ID
    series_type VARCHAR(64),                   -- e.g., T1BRAVO_AXI, T2FLAIR_AXI, MRA_BRAIN
    status      VARCHAR(32) NOT NULL DEFAULT 'NEW',
    dicom_path  TEXT,
    nifti_path  TEXT,
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(study_id, series_uid)
);

CREATE INDEX idx_series_study_id ON series(study_id);
CREATE INDEX idx_series_series_uid ON series(series_uid);
CREATE INDEX idx_series_series_type ON series(series_type);
CREATE INDEX idx_series_status ON series(status);

-- Series 狀態: NEW → TRANSFERRING → TRANSFERRED → CONVERTING → CONVERTED
--             → CONVERSION_SKIPPED | FAILED
```

#### 2.2.3 `pipeline` 表

```sql
CREATE TABLE pipeline (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(64) NOT NULL UNIQUE,   -- e.g., 'Aneurysm', 'WMH', 'CMB'
    display_name    VARCHAR(128),                   -- 顯示名稱
    description     TEXT,
    required_series JSONB NOT NULL,                 -- 所需 Series 類型組合
                                                    -- e.g., [["MRA_BRAIN"]] 或 [["SWAN","T1BRAVO_AXI"]]
    cmd_template    TEXT,                            -- 推論命令模板
    output_formats  JSONB,                           -- 輸出檔案格式定義
    queue_name      VARCHAR(128) NOT NULL,           -- Funboost queue name
    concurrent_mode VARCHAR(32) DEFAULT 'SOLO',      -- SOLO | THREADING
    qps             INTEGER DEFAULT 1,
    max_retry       INTEGER DEFAULT 3,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 預設資料
-- INSERT: Aneurysm, WMH, WMH_PVS, CMB, DWI, Infarct, Area, SynthSeg
```

#### 2.2.4 `task` 表

```sql
CREATE TABLE task (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id      UUID NOT NULL REFERENCES study(id) ON DELETE CASCADE,
    pipeline_id   UUID NOT NULL REFERENCES pipeline(id),
    status        VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    priority      INTEGER NOT NULL DEFAULT 0,       -- 0=normal, 1=high, -1=low
    input_files   JSONB,                             -- 輸入檔案路徑列表
    output_files  JSONB,                             -- 預期輸出檔案路徑列表
    params        JSONB,                             -- 執行參數
    error_message TEXT,                              -- 最近一次錯誤訊息
    attempt_count INTEGER NOT NULL DEFAULT 0,        -- 已嘗試次數
    queued_at     TIMESTAMPTZ,                       -- 放入佇列時間
    started_at    TIMESTAMPTZ,                       -- 開始執行時間
    completed_at  TIMESTAMPTZ,                       -- 完成時間
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_task_study_id ON task(study_id);
CREATE INDEX idx_task_pipeline_id ON task(pipeline_id);
CREATE INDEX idx_task_status ON task(status);
CREATE INDEX idx_task_priority_status ON task(priority DESC, status);
CREATE INDEX idx_task_created_at ON task(created_at);

-- Task 狀態: PENDING → QUEUED → RUNNING → COMPLETED | FAILED
--           FAILED → QUEUED (rerun)
```

#### 2.2.5 `task_result` 表

```sql
CREATE TABLE task_result (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id          UUID NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    attempt_number   INTEGER NOT NULL,
    success          BOOLEAN NOT NULL,
    stdout           TEXT,
    stderr           TEXT,
    exception        TEXT,
    time_cost        FLOAT,                         -- 執行時間 (秒)
    host_name        VARCHAR(128),
    process_id       INTEGER,
    funboost_task_id VARCHAR(128),                  -- Funboost 的 task_id
    result_data      JSONB,                         -- 結構化結果 (如 JSON metrics)
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_task_result_task_id ON task_result(task_id);
```

#### 2.2.6 `event_log` 表 (Append-Only)

```sql
CREATE TABLE event_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(32) NOT NULL,              -- 'study' | 'series' | 'task'
    entity_id   UUID NOT NULL,
    from_status VARCHAR(32),
    to_status   VARCHAR(32) NOT NULL,
    operator    VARCHAR(64) NOT NULL DEFAULT 'system',  -- 'system' | 'user' | 'scheduler'
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_event_log_entity ON event_log(entity_type, entity_id);
CREATE INDEX idx_event_log_created_at ON event_log(created_at);

-- 此表為 append-only，不允許 UPDATE 或 DELETE (透過 application layer 控制)
```

### 2.3 SQLAlchemy Model 設計

```python
# backend/app/models/base.py
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """統一的 Base class - Fowler 模組化原則"""
    pass


class TimestampMixin:
    """時間戳 Mixin - 避免重複定義"""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False
    )
```

```python
# backend/app/models/study.py
import enum
from uuid import uuid4
from sqlalchemy import String, Text, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship


class StudyStatus(str, enum.Enum):
    NEW              = "NEW"
    TRANSFERRING     = "TRANSFERRING"
    TRANSFERRED      = "TRANSFERRED"
    CONVERTING       = "CONVERTING"
    CONVERTED        = "CONVERTED"
    INFERENCE_READY  = "INFERENCE_READY"
    INFERENCE_RUNNING= "INFERENCE_RUNNING"
    INFERENCE_COMPLETE="INFERENCE_COMPLETE"
    RESULTS_SENT     = "RESULTS_SENT"
    FAILED           = "FAILED"


class Study(Base, TimestampMixin):
    __tablename__ = "study"

    id: Mapped[str]         = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    study_uid: Mapped[str]  = mapped_column(String(128), unique=True, nullable=False, index=True)
    study_id: Mapped[str]   = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str]     = mapped_column(String(32), nullable=False, default=StudyStatus.NEW)
    dicom_path: Mapped[str] = mapped_column(Text, nullable=True)
    nifti_path: Mapped[str] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=True)

    # Relationships
    series_list = relationship("Series", back_populates="study", cascade="all, delete-orphan")
    tasks       = relationship("Task", back_populates="study", cascade="all, delete-orphan")
```

```python
# backend/app/models/series.py
import enum

class SeriesStatus(str, enum.Enum):
    NEW                = "NEW"
    TRANSFERRING       = "TRANSFERRING"
    TRANSFERRED        = "TRANSFERRED"
    CONVERTING         = "CONVERTING"
    CONVERTED          = "CONVERTED"
    CONVERSION_SKIPPED = "CONVERSION_SKIPPED"
    FAILED             = "FAILED"


class Series(Base, TimestampMixin):
    __tablename__ = "series"

    id: Mapped[str]          = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    study_id: Mapped[str]    = mapped_column(UUID(as_uuid=False), ForeignKey("study.id", ondelete="CASCADE"), nullable=False)
    series_uid: Mapped[str]  = mapped_column(String(128), nullable=False, index=True)
    series_type: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str]      = mapped_column(String(32), nullable=False, default=SeriesStatus.NEW)
    dicom_path: Mapped[str]  = mapped_column(Text, nullable=True)
    nifti_path: Mapped[str]  = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict]  = mapped_column("metadata", JSONB, nullable=True)

    # Relationships
    study = relationship("Study", back_populates="series_list")

    __table_args__ = (
        UniqueConstraint("study_id", "series_uid", name="uq_series_study_series"),
    )
```

```python
# backend/app/models/pipeline.py
class Pipeline(Base, TimestampMixin):
    __tablename__ = "pipeline"

    id: Mapped[str]              = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str]            = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str]    = mapped_column(String(128), nullable=True)
    description: Mapped[str]     = mapped_column(Text, nullable=True)
    required_series: Mapped[dict]= mapped_column(JSONB, nullable=False)
    cmd_template: Mapped[str]    = mapped_column(Text, nullable=True)
    output_formats: Mapped[dict] = mapped_column(JSONB, nullable=True)
    queue_name: Mapped[str]      = mapped_column(String(128), nullable=False)
    concurrent_mode: Mapped[str] = mapped_column(String(32), default="SOLO")
    qps: Mapped[int]             = mapped_column(default=1)
    max_retry: Mapped[int]       = mapped_column(default=3)
    active: Mapped[bool]         = mapped_column(default=True)

    tasks = relationship("Task", back_populates="pipeline")
```

```python
# backend/app/models/task.py
import enum

class TaskStatus(str, enum.Enum):
    PENDING   = "PENDING"
    QUEUED    = "QUEUED"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"


class Task(Base, TimestampMixin):
    __tablename__ = "task"

    id: Mapped[str]            = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    study_id: Mapped[str]      = mapped_column(UUID(as_uuid=False), ForeignKey("study.id", ondelete="CASCADE"), nullable=False)
    pipeline_id: Mapped[str]   = mapped_column(UUID(as_uuid=False), ForeignKey("pipeline.id"), nullable=False)
    status: Mapped[str]        = mapped_column(String(32), nullable=False, default=TaskStatus.PENDING)
    priority: Mapped[int]      = mapped_column(default=0)
    input_files: Mapped[dict]  = mapped_column(JSONB, nullable=True)
    output_files: Mapped[dict] = mapped_column(JSONB, nullable=True)
    params: Mapped[dict]       = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    attempt_count: Mapped[int] = mapped_column(default=0)
    queued_at: Mapped[datetime]    = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime]   = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    study    = relationship("Study", back_populates="tasks")
    pipeline = relationship("Pipeline", back_populates="tasks")
    results  = relationship("TaskResult", back_populates="task", cascade="all, delete-orphan")
```

```python
# backend/app/models/task_result.py
class TaskResult(Base):
    __tablename__ = "task_result"

    id: Mapped[str]               = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    task_id: Mapped[str]          = mapped_column(UUID(as_uuid=False), ForeignKey("task.id", ondelete="CASCADE"), nullable=False)
    attempt_number: Mapped[int]   = mapped_column(nullable=False)
    success: Mapped[bool]         = mapped_column(nullable=False)
    stdout: Mapped[str]           = mapped_column(Text, nullable=True)
    stderr: Mapped[str]           = mapped_column(Text, nullable=True)
    exception: Mapped[str]        = mapped_column(Text, nullable=True)
    time_cost: Mapped[float]      = mapped_column(nullable=True)
    host_name: Mapped[str]        = mapped_column(String(128), nullable=True)
    process_id: Mapped[int]       = mapped_column(nullable=True)
    funboost_task_id: Mapped[str] = mapped_column(String(128), nullable=True)
    result_data: Mapped[dict]     = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())

    task = relationship("Task", back_populates="results")
```

```python
# backend/app/models/event_log.py
class EventLog(Base):
    __tablename__ = "event_log"

    id: Mapped[str]            = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    entity_type: Mapped[str]   = mapped_column(String(32), nullable=False)  # 'study' | 'series' | 'task'
    entity_id: Mapped[str]     = mapped_column(UUID(as_uuid=False), nullable=False)
    from_status: Mapped[str]   = mapped_column(String(32), nullable=True)
    to_status: Mapped[str]     = mapped_column(String(32), nullable=False)
    operator: Mapped[str]      = mapped_column(String(64), nullable=False, default="system")
    metadata_: Mapped[dict]    = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

### 2.4 新舊 Schema 對照

| 舊 Schema | 新 Schema | 改善 |
|-----------|-----------|------|
| `dcop_conf_bt` (tool_id + ope_no) | `pipeline` + Python Enum | 明確定義，型別安全 |
| `dcop_event_bt` (扁平大表) | `study` + `series` + `task` + `event_log` | 正規化，職責分離 |
| `FunboostConsumeResult` | `task_result` | 統一管理，關聯到 task |
| `RawDicomToNiiInference` | `study.dicom_path` + `study.nifti_path` | 合併至 study 實體 |
| `VsPrimaryKey` (字串拼接) | UUID 主鍵 | 唯一性保障 |
| `ope_no` 數值比較 | Enum 狀態 + 狀態機 | 安全的狀態轉換 |
| JSON `params_data` | 結構化欄位 + 必要時 JSONB | 可查詢，可索引 |

---

## 3. 狀態機設計 (State Pattern)

### 3.1 Study 狀態機

```
                    ┌─────────┐
                    │   NEW   │
                    └────┬────┘
                         │ trigger: DICOM received
                         ▼
                 ┌───────────────┐
                 │ TRANSFERRING  │
                 └───────┬───────┘
                         │ trigger: all series transferred
                         ▼
                 ┌───────────────┐
                 │ TRANSFERRED   │
                 └───────┬───────┘
                         │ trigger: start DICOM→NIfTI
                         ▼
                 ┌───────────────┐
                 │  CONVERTING   │
                 └───────┬───────┘
                         │ trigger: conversion complete
                         ▼
                 ┌───────────────┐
                 │  CONVERTED    │
                 └───────┬───────┘
                         │ trigger: check available pipelines
                         ▼
              ┌────────────────────┐
              │  INFERENCE_READY   │
              └────────┬───────────┘
                       │ trigger: tasks pushed to queue
                       ▼
            ┌─────────────────────┐
            │  INFERENCE_RUNNING  │
            └────────┬────────────┘
                     │ trigger: all tasks done
                     ▼
          ┌────────────────────────┐
          │  INFERENCE_COMPLETE    │
          └────────┬───────────────┘
                   │ trigger: results uploaded to PACS
                   ▼
            ┌──────────────┐
            │ RESULTS_SENT │
            └──────────────┘

    ※ 任何狀態 ──(error)──→ FAILED
    ※ FAILED ──(rerun)──→ NEW (重新開始)
```

### 3.2 Task 狀態機

```
    ┌─────────┐
    │ PENDING │    建立後等待排程
    └────┬────┘
         │ trigger: pushed to Funboost queue
         ▼
    ┌─────────┐
    │ QUEUED  │    已在 RabbitMQ 佇列中
    └────┬────┘
         │ trigger: consumer picks up
         ▼
    ┌─────────┐
    │ RUNNING │    推論執行中
    └────┬────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌───────┐  ┌────────┐
│COMPLETED│  │ FAILED │
└───────┘  └───┬────┘
               │ trigger: manual rerun or auto-retry
               ▼
           ┌─────────┐
           │ QUEUED  │  (重新加入佇列)
           └─────────┘
```

### 3.3 Python 狀態機實作

```python
# backend/app/domain/state_machine.py
from typing import Dict, Set

class StateMachine:
    """通用有限狀態機 - State Pattern"""

    def __init__(self, transitions: Dict[str, Set[str]]):
        self._transitions = transitions

    def can_transition(self, from_status: str, to_status: str) -> bool:
        allowed = self._transitions.get(from_status, set())
        return to_status in allowed

    def validate_transition(self, from_status: str, to_status: str) -> None:
        if not self.can_transition(from_status, to_status):
            raise ValueError(
                f"Invalid state transition: {from_status} → {to_status}"
            )


STUDY_STATE_MACHINE = StateMachine({
    "NEW":               {"TRANSFERRING", "FAILED"},
    "TRANSFERRING":      {"TRANSFERRED", "FAILED"},
    "TRANSFERRED":       {"CONVERTING", "FAILED"},
    "CONVERTING":        {"CONVERTED", "FAILED"},
    "CONVERTED":         {"INFERENCE_READY", "FAILED"},
    "INFERENCE_READY":   {"INFERENCE_RUNNING", "FAILED"},
    "INFERENCE_RUNNING": {"INFERENCE_COMPLETE", "FAILED"},
    "INFERENCE_COMPLETE":{"RESULTS_SENT", "FAILED"},
    "FAILED":            {"NEW"},  # rerun
})

TASK_STATE_MACHINE = StateMachine({
    "PENDING":   {"QUEUED", "FAILED"},
    "QUEUED":    {"RUNNING", "FAILED"},
    "RUNNING":   {"COMPLETED", "FAILED"},
    "FAILED":    {"QUEUED"},  # retry/rerun
})
```

---

## 4. Repository Pattern 設計

```python
# backend/app/repositories/base.py
from typing import TypeVar, Generic, Type, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

T = TypeVar("T")

class BaseRepository(Generic[T]):
    """Repository 基底類別 - Fowler Data Mapper Pattern"""

    def __init__(self, session: AsyncSession, model_class: Type[T]):
        self._session = session
        self._model_class = model_class

    async def get_by_id(self, id: str) -> Optional[T]:
        return await self._session.get(self._model_class, id)

    async def list_all(self, offset: int = 0, limit: int = 50) -> List[T]:
        stmt = select(self._model_class).offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def add(self, entity: T) -> T:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def delete(self, entity: T) -> None:
        await self._session.delete(entity)


# backend/app/repositories/study_repository.py
class StudyRepository(BaseRepository[Study]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, Study)

    async def get_by_study_uid(self, study_uid: str) -> Optional[Study]:
        stmt = select(Study).where(Study.study_uid == study_uid)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_status(self, status: str, offset=0, limit=50) -> List[Study]:
        stmt = (select(Study)
                .where(Study.status == status)
                .order_by(Study.created_at.desc())
                .offset(offset).limit(limit))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


# backend/app/repositories/task_repository.py
class TaskRepository(BaseRepository[Task]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, Task)

    async def list_by_study(self, study_id: str) -> List[Task]:
        stmt = select(Task).where(Task.study_id == study_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_status(self, status: str, offset=0, limit=50) -> List[Task]:
        stmt = (select(Task)
                .where(Task.status == status)
                .order_by(Task.priority.desc(), Task.created_at)
                .offset(offset).limit(limit))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_pending_tasks_for_study(self, study_id: str) -> List[Task]:
        stmt = (select(Task)
                .where(Task.study_id == study_id, Task.status == TaskStatus.PENDING)
                .order_by(Task.priority.desc()))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
```

---

## 5. Domain Service 設計

### 5.1 StudyService

```python
# backend/app/services/study_service.py
class StudyService:
    """Study 領域服務 - 管理 Study 生命週期"""

    def __init__(self, study_repo: StudyRepository,
                       series_repo: SeriesRepository,
                       task_repo: TaskRepository,
                       pipeline_repo: PipelineRepository,
                       event_publisher: EventPublisher):
        self._study_repo = study_repo
        self._series_repo = series_repo
        self._task_repo = task_repo
        self._pipeline_repo = pipeline_repo
        self._events = event_publisher

    async def register_study(self, study_uid: str, dicom_path: str) -> Study:
        """收到新 DICOM Study 時呼叫"""
        study = Study(study_uid=study_uid, dicom_path=dicom_path)
        study = await self._study_repo.add(study)
        await self._events.publish("study.created", study)
        return study

    async def transition_status(self, study_id: str, new_status: str) -> Study:
        """狀態轉換 - 透過狀態機驗證"""
        study = await self._study_repo.get_by_id(study_id)
        STUDY_STATE_MACHINE.validate_transition(study.status, new_status)
        old_status = study.status
        study.status = new_status
        await self._events.publish("study.status_changed", study,
                                    from_status=old_status, to_status=new_status)
        return study

    async def create_inference_tasks(self, study_id: str) -> List[Task]:
        """根據 Study 的 Series 自動建立推論任務"""
        study = await self._study_repo.get_by_id(study_id)
        series_list = await self._series_repo.list_by_study(study_id)
        pipelines = await self._pipeline_repo.list_active()

        tasks = []
        for pipeline in pipelines:
            if self._can_run_pipeline(pipeline, series_list):
                input_files = self._resolve_input_files(pipeline, series_list)
                task = Task(
                    study_id=study_id,
                    pipeline_id=pipeline.id,
                    input_files=input_files,
                    params={"nifti_path": study.nifti_path, "dicom_path": study.dicom_path}
                )
                task = await self._task_repo.add(task)
                tasks.append(task)
        return tasks

    def _can_run_pipeline(self, pipeline: Pipeline, series_list: List[Series]) -> bool:
        """檢查是否有足夠的 Series 執行該 Pipeline"""
        available_types = {s.series_type for s in series_list if s.status == "CONVERTED"}
        for required_combo in pipeline.required_series:
            if all(rt in available_types for rt in required_combo):
                return True
        return False

    def _resolve_input_files(self, pipeline: Pipeline, series_list: List[Series]) -> dict:
        """解析推論任務所需的輸入檔案"""
        # ... 根據 pipeline.required_series 與 series.nifti_path 組合
        pass
```

### 5.2 TaskService

```python
# backend/app/services/task_service.py
class TaskService:
    """Task 領域服務 - 管理推論任務生命週期"""

    def __init__(self, task_repo: TaskRepository,
                       result_repo: TaskResultRepository,
                       event_publisher: EventPublisher,
                       queue_publisher: QueuePublisher):
        self._task_repo = task_repo
        self._result_repo = result_repo
        self._events = event_publisher
        self._queue = queue_publisher

    async def enqueue_task(self, task_id: str) -> Task:
        """將任務推入 Funboost 佇列"""
        task = await self._task_repo.get_by_id(task_id)
        TASK_STATE_MACHINE.validate_transition(task.status, TaskStatus.QUEUED)
        task.status = TaskStatus.QUEUED
        task.queued_at = datetime.utcnow()
        self._queue.push(task)
        await self._events.publish("task.queued", task)
        return task

    async def record_result(self, task_id: str, result: TaskResult) -> Task:
        """記錄任務執行結果 - 由 Funboost 回調函數呼叫"""
        task = await self._task_repo.get_by_id(task_id)
        result.task_id = task_id
        result.attempt_number = task.attempt_count + 1
        task.attempt_count += 1
        await self._result_repo.add(result)

        if result.success:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
        else:
            task.status = TaskStatus.FAILED
            task.error_message = result.exception
        await self._events.publish("task.finished", task)
        return task

    async def rerun_task(self, task_id: str) -> Task:
        """重新執行失敗任務"""
        task = await self._task_repo.get_by_id(task_id)
        TASK_STATE_MACHINE.validate_transition(task.status, TaskStatus.QUEUED)
        task.status = TaskStatus.QUEUED
        task.error_message = None
        task.queued_at = datetime.utcnow()
        self._queue.push(task)
        await self._events.publish("task.rerun", task)
        return task
```

---

## 6. Funboost 佇列消費者設計

### 6.1 統一配置

```python
# code_ai/task/config.py
from funboost import BoosterParams, BrokerEnum, ConcurrentModeEnum


class BaseBoosterParams(BoosterParams):
    """統一的 Funboost 消費者參數 - Singleton 配置"""
    broker_kind: str = BrokerEnum.RABBITMQ_AMQPSTORM
    is_send_consumer_hearbeat_to_redis: bool = True
    is_using_rpc_mode: bool = True
    rpc_result_expire_seconds: int = 1800
    max_retry_times: int = 3
    retry_interval: int = 20
    is_push_to_dlx_queue_when_retry_max_times: bool = True


class InferenceBoosterParams(BaseBoosterParams):
    """推論任務消費者 - GPU 獨占 (SOLO mode)"""
    concurrent_mode: str = ConcurrentModeEnum.SOLO
    concurrent_num: int = 1
    qps: int = 1


class IOBoosterParams(BaseBoosterParams):
    """I/O 密集任務消費者 - 多執行緒"""
    concurrent_mode: str = ConcurrentModeEnum.THREADING
    concurrent_num: int = 10
    qps: int = 5
```

### 6.2 Task Consumer (消費者) 與 DB 整合

```python
# code_ai/task/consumers.py
from funboost import Booster, fct
from code_ai.task.config import InferenceBoosterParams, IOBoosterParams


def save_task_result_to_db(function_result_status):
    """
    Funboost 回調函數 - 將任務結果寫入資料庫
    取代舊的 HTTP POST 回報機制 (Observer Pattern)
    """
    from backend.app.models.task_result import TaskResult
    from backend.app.database import get_sync_session

    status_dict = function_result_status.get_status_dict()
    task_db_id = status_dict.get("params", {}).get("task_db_id")

    if not task_db_id:
        return

    with get_sync_session() as session:
        task_result = TaskResult(
            task_id=task_db_id,
            attempt_number=status_dict.get("run_times", 1),
            success=status_dict.get("success", False),
            stdout=str(status_dict.get("result", "")),
            stderr="",
            exception=status_dict.get("exception", None),
            time_cost=status_dict.get("time_cost", 0),
            host_name=status_dict.get("host_name", ""),
            process_id=status_dict.get("process_id", 0),
            funboost_task_id=status_dict.get("task_id", ""),
        )
        session.add(task_result)

        # 更新 task 狀態
        task = session.get(Task, task_db_id)
        if task:
            task.status = "COMPLETED" if status_dict["success"] else "FAILED"
            task.attempt_count = status_dict.get("run_times", 1)
            if status_dict["success"]:
                task.completed_at = datetime.utcnow()
            else:
                task.error_message = status_dict.get("exception", "Unknown error")

        session.commit()


@Booster(InferenceBoosterParams(
    queue_name="task_pipeline_inference_queue",
    user_custom_record_process_info_func=save_task_result_to_db,
))
def task_pipeline_inference(task_db_id: str, nifti_study_path: str,
                            dicom_study_path: str, study_uid: str, study_id: str):
    """
    推論任務消費者 - Command Pattern
    每個任務對應資料庫中的一筆 Task 記錄
    """
    import pathlib
    import subprocess
    from code_ai.utils.inference import build_inference_cmd

    inference_cmd = build_inference_cmd(
        pathlib.Path(nifti_study_path),
        pathlib.Path(dicom_study_path)
    )

    results = []
    for item in inference_cmd.cmd_items:
        process = subprocess.Popen(
            args=item.cmd_str, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        results.append({
            "name": item.name,
            "cmd": item.cmd_str,
            "returncode": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
        })
    return results
```

### 6.3 Queue Publisher (發佈者)

```python
# code_ai/task/publisher.py
class QueuePublisher:
    """佇列發佈者 - Facade Pattern"""

    @staticmethod
    def push_inference_task(task: Task) -> None:
        """將推論任務推入 Funboost 佇列"""
        task_pipeline_inference.push({
            "task_db_id": str(task.id),
            "nifti_study_path": task.params["nifti_path"],
            "dicom_study_path": task.params["dicom_path"],
            "study_uid": task.study.study_uid,
            "study_id": task.study.study_id,
        })

    @staticmethod
    def push_synthseg_task(params: dict) -> None:
        """將 SynthSeg 前處理任務推入佇列"""
        from code_ai.task.consumers import task_synthseg
        task_synthseg.push(params)
```

### 6.4 Event Publisher (Observer Pattern)

```python
# backend/app/domain/events.py
from typing import Callable, Dict, List


class EventPublisher:
    """事件發佈者 - Observer Pattern"""

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, listener: Callable):
        self._listeners.setdefault(event_type, []).append(listener)

    async def publish(self, event_type: str, entity, **kwargs):
        for listener in self._listeners.get(event_type, []):
            await listener(entity, **kwargs)


# 註冊 Listener
async def log_event_to_db(entity, **kwargs):
    """寫入 event_log 表"""
    # ...

async def trigger_next_step(entity, **kwargs):
    """觸發下一步處理 (如 study.converted → create inference tasks)"""
    # ...
```

---

## 7. 目錄結構重組

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # 入口
│   ├── server.py                    # FastAPI app 建立
│   ├── database.py                  # DB 連線配置 (統一)
│   │
│   ├── models/                      # Data Source Layer
│   │   ├── __init__.py              # 匯出所有 Model
│   │   ├── base.py                  # Base, TimestampMixin
│   │   ├── study.py                 # Study + StudyStatus
│   │   ├── series.py                # Series + SeriesStatus
│   │   ├── pipeline.py              # Pipeline
│   │   ├── task.py                  # Task + TaskStatus
│   │   ├── task_result.py           # TaskResult
│   │   └── event_log.py             # EventLog
│   │
│   ├── repositories/                # Data Source Layer
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseRepository[T]
│   │   ├── study_repository.py
│   │   ├── series_repository.py
│   │   ├── task_repository.py
│   │   ├── pipeline_repository.py
│   │   └── event_log_repository.py
│   │
│   ├── services/                    # Domain Layer
│   │   ├── __init__.py
│   │   ├── study_service.py
│   │   ├── task_service.py
│   │   └── pipeline_service.py
│   │
│   ├── domain/                      # Domain Layer
│   │   ├── __init__.py
│   │   ├── state_machine.py         # StateMachine + 定義
│   │   └── events.py                # EventPublisher
│   │
│   ├── schemas/                     # Presentation Layer (Pydantic)
│   │   ├── __init__.py
│   │   ├── study_schemas.py
│   │   ├── task_schemas.py
│   │   └── pipeline_schemas.py
│   │
│   ├── routers/                     # Presentation Layer (API)
│   │   ├── __init__.py
│   │   ├── study_router.py
│   │   ├── task_router.py
│   │   ├── pipeline_router.py
│   │   ├── series_router.py
│   │   └── dashboard_router.py
│   │
│   └── dependencies.py              # FastAPI DI (依賴注入)

code_ai/
├── task/
│   ├── config.py                    # 統一 BoosterParams
│   ├── consumers.py                 # Funboost 消費者函數
│   ├── publisher.py                 # QueuePublisher
│   └── workflow.py                  # Chain of Responsibility (保留)
├── pipeline/                        # 推論腳本 (不變)
├── scheduler/                       # 排程器 (簡化)
└── utils/                           # 工具函數 (不變)
```

---

## 8. 資料庫遷移策略

### 8.1 遷移步驟

| 步驟 | 動作 | 風險 |
|------|------|------|
| 1 | 建立新 Schema (study, series, pipeline, task, task_result, event_log) | 低 |
| 2 | 寫遷移腳本：`dcop_event_bt` → `study` + `series` + `event_log` | 中 |
| 3 | 寫遷移腳本：`dcop_conf_bt` → `pipeline` | 低 |
| 4 | 寫遷移腳本：`funboost_consume_results` → `task_result` | 中 |
| 5 | 雙寫期：新舊 Schema 同時寫入，驗證一致性 | 低 |
| 6 | 切換讀取源到新 Schema | 中 |
| 7 | 停止舊 Schema 寫入，移除舊表 | 低 |

### 8.2 Alembic 遷移

建議使用 Alembic 進行 Schema 遷移管理，保留完整遷移歷史。

---
