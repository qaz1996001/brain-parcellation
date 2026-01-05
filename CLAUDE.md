# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brain parcellation system for medical image analysis with AI inference pipelines. The system processes DICOM medical images through various AI models (aneurysm detection, WMH segmentation, brain parcellation) and generates DICOM-SEG results.

**Tech Stack**: Python 3.10+, FastAPI, TensorFlow, RabbitMQ (funboost), Redis, PostgreSQL, Docker

## Build & Run Commands

### Environment Setup
```bash
# Install dependencies with uv (preferred)
pip install uv
uv sync

# Or system-wide install
uv pip install -r pyproject.toml --system

# Activate conda environment (if using conda)
conda activate tf_2_14
```

### Run Services
```bash
# Backend API server (FastAPI)
export PYTHONPATH=$(pwd) && python3 backend/app/main.py

# Task worker (funboost/RabbitMQ consumer)
export PYTHONPATH=$(pwd) && python3 funboost_cli_user.py

# Run with specific commands via funboost CLI
python funboost_cli_user.py consume <queue_name>
python funboost_cli_user.py push <queue_name> --x=1 --y=2
```

### Infrastructure (Docker Compose)
```bash
# Start RabbitMQ, Redis, PostgreSQL
docker-compose up -d

# Stop services
docker-compose down
```

### Pipeline Examples
```bash
# MRA_BRAIN aneurysm detection
export PYTHONPATH=$(pwd) && python3 code_ai/pipeline/pipeline_aneurysm_tensorflow.py \
  --ID 14914694_20220905_MR_21109050071 \
  --Inputs /path/to/MRA_BRAIN.nii.gz \
  --Output_folder /path/to/output \
  --InputsDicomDir /path/to/dicom

# WMH segmentation
export PYTHONPATH=$(pwd) && python3 code_ai/pipeline/pipeline_synthseg_wmh_tensorflow.py \
  --ID <study_id> \
  --Inputs /path/to/T2FLAIR_AXI.nii.gz \
  --Output_folder /path/to/output \
  --InputsDicomDir /path/to/dicom
```

### Testing
Tests use pytest (when available):
```bash
pytest tests/contract/
pytest tests/unit/
```

## High-Level Architecture

### Three-Layer System Design

**1. Backend Layer** (`backend/app/`)
- FastAPI application serving REST APIs
- Handles incoming requests, dispatches tasks to workers
- Module-based routing: `sync/`, `listen/`, `study/`, `series/`, `rerun/`
- Each module has: `routers.py`, `service.py`, `schemas.py`, `deps.py`

**2. Worker Layer** (`code_ai/task/`)
- Funboost-powered distributed task workers consuming from RabbitMQ
- Task functions decorated with `@boost` from funboost
- Main tasks:
  - `task_pipeline_inference`: GPU inference for AI models
  - `task_subprocess_inference`: Subprocess execution wrapper
  - `task_dicom2nii`: DICOM to NIfTI conversion

**3. Pipeline Layer** (`code_ai/pipeline/`)
- AI model inference pipelines (TensorFlow/Keras)
- Core pipelines:
  - `pipeline_aneurysm_tensorflow.py`: MRA brain aneurysm detection
  - `pipeline_synthseg_tensorflow.py`: Brain parcellation
  - `pipeline_wmh_tensorflow.py`: White matter hyperintensity
  - `pipeline_cmb_tensorflow.py`: Cerebral microbleed detection
- Utilities in `code_ai/SynthSeg/`, `code_ai/utils/`

### Configuration Architecture (Pure Function Refactoring)

**Configuration Centralization Pattern** (Knuth & Linus principles):
- **SINGLE SOURCE**: `backend/app/config/loader.py` is the ONLY place reading environment variables
- **Pure Functions**: Task functions accept paths as parameters, not from environment
- **Immutable Config**: `BackendConfig`, `PathConfig`, `APIConfig` models (Pydantic-based)
- **Fail-Safe Production**: Missing env vars → uses defaults, logs warnings
- **Strict Testing**: `fail_safe=False` mode raises errors for missing config

```python
# Configuration loading (centralized)
from backend.app.config.loader import load_backend_config_from_env
config = load_backend_config_from_env(fail_safe=True)  # Production
config = load_backend_config_from_env(fail_safe=False) # Testing/CI

# Task parameter injection pattern
from backend.app.config.task_paths import get_task_execution_paths
task_params = {"path_params": get_task_execution_paths(config)}
```

### Dual Deployment GPU Architecture

Single GPU worker serves multiple environments (production + testing):
- Backend passes environment-specific paths as task parameters
- Worker executes in context specified by dispatcher (not worker's `.env`)
- Enables 50% GPU resource optimization vs dedicated workers per environment

### Data Flow

```
DICOM Input → Backend API → RabbitMQ → Worker → AI Pipeline → DICOM-SEG Output
                 ↓                                    ↓
              PostgreSQL                          Upload to Orthanc/Platform
```

1. **Input**: DICOM files received via sync/listen endpoints
2. **Dispatch**: Backend creates task with parameters → pushes to RabbitMQ queue
3. **Process**: Worker consumes task → runs AI pipeline → generates results
4. **Output**: DICOM-SEG + JSON metadata uploaded to Orthanc/platform
5. **Tracking**: PostgreSQL stores task status and results

### Key Design Patterns

**Module Structure** (backend services):
```
backend/app/<module>/
  ├── routers.py      # FastAPI route handlers
  ├── service.py      # Business logic, task dispatch
  ├── schemas.py      # Pydantic request/response models
  ├── deps.py         # Dependency injection
  └── urls.py         # URL constants
```

**Task Dispatch Pattern**:
```python
# Service layer (backend/app/*/service.py)
from code_ai.task.task_pipeline import task_pipeline_inference
from backend.app.config.task_paths import get_task_execution_paths

async def dispatch_inference(study_id: str, config: BackendConfig):
    task_params = {
        "study_id": study_id,
        "path_params": get_task_execution_paths(config)
    }
    task_pipeline_inference.push(task_params)
```

**Pure Function AI Pipelines**:
```python
# Pipeline functions accept explicit parameters
def pipeline_aneurysm(
    input_file: str,
    output_folder: str,
    dicom_dir: str,
    gpu_n: int = 0
) -> dict:
    # Deterministic, testable, no side effects from environment
    ...
```

## Environment Variables

Critical variables (see `.env` for full list):

**Paths** (configured per deployment environment):
- `PATH_ROOT`: Base directory for all processing
- `PATH_RAW_DICOM`: Raw DICOM input directory
- `PATH_RENAME_DICOM`: Processed DICOM output
- `PATH_RENAME_NIFTI`: NIfTI conversion output
- `PATH_JSON`, `PATH_LOG`, `PATH_PROCESS`: Supporting directories

**Infrastructure**:
- `RABBITMQ_HOST`, `RABBITMQ_PORT`: Message queue for task distribution
- `REDIS_HOST`, `REDIS_PORT`: Cache and session storage
- `AI_APP_CONNECTION_STRING`: PostgreSQL database URI

**API Configuration**:
- `UPLOAD_DATA_API_URL`: Platform API for result upload
- `UPLOAD_DATA_DICOM_SEG_URL`: Orthanc PACS server URL

**Application**:
- `AI_APP_PORT`, `AI_APP_TITLE`, `AI_APP_VERSION`: FastAPI app config
- `GPU_N`: GPU device index for inference

## OpenSpec Integration

This project uses **OpenSpec** for change management and compliance tracking:
- Changes documented in `openspec/changes/<change-id>/`
- Each change has: `proposal.md`, `design.md`, `tasks.md`, `specs/*/spec.md`
- Validation: `openspec validate <change-id>`
- Apply changes: `/openspec:apply <change-id>`

**Current Active Changes**:
- `refactor-to-pure-functions`: Configuration centralization pattern
- `parameterize-task-pipeline-paths`: Dual deployment GPU support
- `add-environment-support`: Environment-aware configuration

## Code Style & Conventions

**Python Naming**:
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

**Import Organization**:
```python
# Standard library
import os
from pathlib import Path

# Third-party
from fastapi import APIRouter
import pandas as pd

# Local imports
from backend.app.config import load_backend_config_from_env
from code_ai.task import task_pipeline_inference
```

**Logging Pattern**:
```python
import logging
logger = logging.getLogger(__name__)

# Per-task log files are managed by funboost
logger.info(f"Processing study {study_id}")
```

## Important Constraints

**GPU Resources**:
- Limited GPU availability requires careful resource management
- Dual deployment architecture maximizes GPU utilization
- TensorFlow models require CUDA 11.x compatibility

**GPU Mutual Exclusion Pattern** (CRITICAL for new GPU tasks):
- All GPU inference tasks MUST use `task_pipeline_inference` as the unified entry point
- Single queue with `qps=1` provides natural GPU mutual exclusion
- NO external coordination needed (no Redis locks, no DB semaphores)
- Data structure determines behavior: `'series_uids' in func_params` → Series Level, otherwise → Study Level

```python
# CORRECT: Use unified entry point for GPU tasks
from code_ai.task.task_pipeline import task_pipeline_inference

# Study Level (original behavior)
task_pipeline_inference.push({
    'study_uid': '...',
    'nifti_study_path': '/path/to/study',
    # NO 'series_uids' key
})

# Series Level (new behavior)
task_pipeline_inference.push({
    'series_uids': ['series_a', 'series_b'],  # Key presence determines level
    'study_uid': '...',
    'model_id': 'aneurysm_v1',
})

# WRONG: Do NOT create separate GPU queues
# This would cause GPU conflicts!
```

**Why qps=1?**
- funboost's `qps=1` ensures only one task executes at a time
- First-in-first-out scheduling, fair for both Study and Series Level
- Simple, reliable, fewer failure points than distributed locks

**DICOM Standards**:
- DICOM-SEG output must conform to medical imaging standards
- Metadata schemas in `code_ai/pipeline/dicomseg/schema/`

**Medical Device Compliance**:
- IEC 62304 software lifecycle documentation in `docs/resource/meta-framework/regulations/`
- RFC 2119 requirement keywords (MUST, SHALL, SHOULD)
- Architecture Decision Records in `docs/adr/`

**Backward Compatibility**:
- Environment fallback pattern maintains compatibility during migration
- Task functions check parameters first, then fall back to environment variables

## Development Workflow

**File Organization**:
- Tests: `tests/contract/`, `tests/unit/`
- Documentation: `docs/` (architecture, ADRs, deployment guides)
- Claude-specific docs: Place reports/analyses in `claudedocs/` (not repository root)

**Configuration Changes**:
1. Update models in `backend/app/config/models.py`
2. Update loader in `backend/app/config/loader.py`
3. Write tests in `tests/contract/test_config_loading.py`
4. Update `.env` example

**Adding New AI Pipeline**:
1. Create pipeline script in `code_ai/pipeline/pipeline_<name>_tensorflow.py`
2. Add DICOM-SEG schema in `code_ai/pipeline/dicomseg/schema/<name>.py`
3. **IMPORTANT**: Reuse `task_pipeline_inference` for GPU tasks (see GPU Mutual Exclusion Pattern above)
4. Update service layer to dispatch via `task_pipeline_inference.push()`
5. Document in `docs/API_REFERENCE.md`

> **Warning**: Do NOT create separate GPU task queues. All GPU inference MUST go through `task_pipeline_inference` to ensure proper GPU mutual exclusion via `qps=1`.

## External Dependencies

**Medical Imaging**:
- pydicom: DICOM file manipulation
- pydicom-seg: DICOM-SEG creation
- nibabel: NIfTI file handling
- SimpleITK: Image processing
- pyorthanc: Orthanc PACS integration

**AI/ML**:
- TensorFlow 2.14 (via conda environment)
- NumPy, SciKit-Image: Numerical processing
- brainextractor: Brain extraction utilities

**Task Distribution**:
- funboost: Distributed task framework (wraps Celery/RabbitMQ)
- celery: Background task execution
- redis: Result backend and caching

**Web Framework**:
- FastAPI: REST API framework
- uvicorn: ASGI server
- SQLAlchemy 2.0: Async ORM for PostgreSQL
- asyncpg: PostgreSQL async driver