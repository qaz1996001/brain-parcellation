# Project Context

## Purpose

Brain parcellation system for medical image analysis with AI inference pipelines. The system processes DICOM medical images through various AI models (aneurysm detection, WMH segmentation, brain parcellation, cerebral microbleed detection) and generates DICOM-SEG results for clinical use.

**Core Capabilities**:
- DICOM image ingestion and preprocessing
- AI-powered medical image analysis (TensorFlow/Keras models)
- DICOM-SEG output generation compliant with medical imaging standards
- Distributed task processing for GPU inference workloads
- Integration with Orthanc PACS and external platforms

## Tech Stack

**Backend Framework**:
- Python 3.10+
- FastAPI (REST API framework)
- uvicorn (ASGI server)
- SQLAlchemy 2.0 (async ORM)
- asyncpg (PostgreSQL async driver)

**Task Distribution**:
- funboost 48.4 (distributed task framework)
- Celery 5.4 (background task execution)
- RabbitMQ (message queue)
- Redis (result backend and caching)

**AI/ML Stack**:
- TensorFlow 2.14 (via conda environment tf_2_14)
- CUDA 11.x (GPU acceleration)
- NumPy, SciKit-Image (numerical processing)
- brainextractor (brain extraction utilities)

**Medical Imaging**:
- pydicom (DICOM file manipulation)
- pydicom-seg (DICOM-SEG creation)
- nibabel (NIfTI file handling)
- SimpleITK (image processing)
- pyorthanc (Orthanc PACS integration)

**Infrastructure**:
- Docker, Docker Compose
- PostgreSQL (task and study tracking)
- Redis (session and cache storage)

## Project Conventions

### Code Style

**Python Naming**:
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private functions: `_leading_underscore`

**Import Organization**:
```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
from fastapi import APIRouter
import pandas as pd

# 3. Local imports
from backend.app.config import load_backend_config_from_env
from code_ai.task import task_pipeline_inference
```

**Docstring Style**: Google-style docstrings with type hints

**Logging Pattern**:
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Processing study {study_id}")
```

### Architecture Patterns

**Three-Layer System Design**:

1. **Backend Layer** (`backend/app/`)
   - FastAPI application serving REST APIs
   - Module-based routing: `sync/`, `listen/`, `study/`, `series/`, `rerun/`, `inference/`
   - Each module structure:
     ```
     backend/app/<module>/
       ├── routers.py      # FastAPI route handlers
       ├── service.py      # Business logic, task dispatch
       ├── schemas.py      # Pydantic request/response models
       ├── deps.py         # Dependency injection
       └── urls.py         # URL constants
     ```

2. **Worker Layer** (`code_ai/task/`)
   - Funboost-powered distributed task workers consuming from RabbitMQ
   - Task functions decorated with `@boost` from funboost
   - Key tasks: `task_pipeline_inference`, `task_subprocess_inference`, `task_dicom2nii`

3. **Pipeline Layer** (`code_ai/pipeline/`)
   - AI model inference pipelines (TensorFlow/Keras)
   - Pure function design: accept explicit parameters, no side effects from environment
   - DICOM-SEG output generation with schema definitions

**Configuration Pattern** (Pure Function Architecture):
- **Single Source of Truth**: `backend/app/config/loader.py` reads all environment variables
- **Immutable Config**: `BackendConfig`, `PathConfig`, `APIConfig` dataclasses (frozen=True)
- **Parameter Injection**: Task functions receive paths as parameters, not from environment
- **Fail-Safe Mode**: Production uses defaults for missing vars; testing raises errors

**Data Flow**:
```
DICOM Input → Backend API → RabbitMQ → Worker → AI Pipeline → DICOM-SEG Output
                 ↓                                    ↓
              PostgreSQL                          Upload to Orthanc/Platform
```

### Testing Strategy

**Test Structure**:
```
tests/
├── contract/           # Contract tests (API, config, integration)
│   ├── test_config_loading.py
│   ├── test_pipeline_base.py
│   └── test_sync_service.py
└── unit/               # Unit tests (isolated logic)
```

**Testing Approach**:
- pytest as test framework
- Contract tests validate interfaces between layers
- Configuration tests use `fail_safe=False` for strict validation
- Mock external dependencies (RabbitMQ, Redis, PostgreSQL)

**Run Tests**:
```bash
pytest tests/contract/
pytest tests/unit/
```

### Git Workflow

**Branch Strategy**:
- Feature branches for all work: `feature/<name>`, `fix/<name>`
- Main/master branch protected
- Create branches before making changes

**Commit Conventions**:
- Descriptive commit messages focusing on "why" not "what"
- Incremental commits with meaningful messages
- Verify changes with `git diff` before committing

**OpenSpec Integration**:
- Changes documented in `openspec/changes/<change-id>/`
- Each change: `proposal.md`, `design.md`, `tasks.md`, `specs/*/spec.md`
- Validate with `openspec validate <change-id>`

## Domain Context

**Medical Imaging Terminology**:
- **DICOM**: Digital Imaging and Communications in Medicine standard
- **DICOM-SEG**: Segmentation objects in DICOM format
- **NIfTI**: Neuroimaging Informatics Technology Initiative format
- **Orthanc**: Open-source PACS (Picture Archiving and Communication System)
- **Study**: Collection of medical images from single patient visit
- **Series**: Subset of images within a study (e.g., T1, T2, FLAIR sequences)

**AI Pipeline Types**:
- `pipeline_aneurysm_tensorflow.py`: MRA brain aneurysm detection
- `pipeline_synthseg_tensorflow.py`: Brain parcellation (SynthSeg)
- `pipeline_wmh_tensorflow.py`: White matter hyperintensity segmentation
- `pipeline_cmb_tensorflow.py`: Cerebral microbleed detection
- `pipeline_synthseg_wmh_tensorflow.py`: Combined parcellation + WMH
- `pipeline_infarct_tensorflow.py`: Infarct detection

**Processing Flow**:
1. DICOM files received via sync/listen endpoints
2. Backend creates task with environment-specific parameters
3. Worker consumes task from RabbitMQ queue
4. AI pipeline processes images using GPU
5. Results saved as DICOM-SEG + JSON metadata
6. Upload to Orthanc PACS and platform API

## Important Constraints

**GPU Resources**:
- Limited GPU availability requires careful resource management
- Dual deployment architecture: single GPU worker serves multiple environments
- TensorFlow models require CUDA 11.x compatibility
- GPU device selection via `GPU_N` environment variable

**DICOM Standards Compliance**:
- DICOM-SEG output must conform to medical imaging standards
- Metadata schemas defined in `code_ai/pipeline/dicomseg/schema/`
- Proper DICOM tags and patient information handling

**Medical Device Compliance**:
- IEC 62304 software lifecycle documentation
- RFC 2119 requirement keywords (MUST, SHALL, SHOULD)
- Architecture Decision Records in `docs/adr/`
- Traceability from requirements to implementation

**Configuration Constraints**:
- All environment variables centralized in `backend/app/config/loader.py`
- Production mode: fail-safe with defaults and warnings
- Testing mode: strict validation, no defaults
- Path parameters passed explicitly to tasks (not read from worker environment)

**Backward Compatibility**:
- Environment fallback pattern during migration phases
- Task functions check parameters first, then fall back to environment variables
- Gradual rollout with feature flags support

## External Dependencies

**Medical Imaging Libraries**:
- pydicom: DICOM file read/write
- pydicom-seg: DICOM-SEG object creation
- nibabel: NIfTI file handling
- SimpleITK: Advanced image processing
- pyorthanc: Orthanc PACS REST API client
- brainextractor: Brain mask extraction

**AI/ML Libraries**:
- TensorFlow 2.14: Deep learning framework
- NumPy: Numerical computations
- SciKit-Image: Image processing algorithms
- OpenCV: Computer vision utilities

**Task Distribution**:
- funboost: Distributed task framework (RabbitMQ/Redis backends)
- Celery: Task queue management
- Redis: Result backend, caching, session storage

**Web Framework**:
- FastAPI: Modern async REST API framework
- uvicorn: ASGI server
- Pydantic: Data validation and serialization
- SQLAlchemy 2.0: Async ORM for PostgreSQL
- asyncpg: PostgreSQL async driver
- httpx: Async HTTP client

**External Systems**:
- Orthanc PACS: Medical image storage (DICOM-SEG upload)
- Platform API: Result notification and data upload
- RabbitMQ: Message queue for task distribution
- PostgreSQL: Task and study tracking database
- Redis: Cache and session management
