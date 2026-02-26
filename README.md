# SHH AI

Medical imaging DICOM processing and AI inference platform for brain MRI analysis. The system bridges DICOM archives (Orthanc), format conversion (NIFTI), and deep learning inference pipelines with comprehensive state tracking and retry capabilities.

## Architecture Overview

```
                    ┌─────────────┐
                    │   Orthanc   │  DICOM Archive (PACS)
                    │  :4242/8042 │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Backend   │  FastAPI REST API (:8000)
                    │  /api/v1/*  │  Event-driven state machine
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼───┐  ┌──────▼──┐  ┌──────▼───┐
       │ RabbitMQ │  │  Redis  │  │PostgreSQL│
       │  :5672   │  │  :6379  │  │  :5432   │
       └──────┬───┘  └─────────┘  └──────────┘
              │
       ┌──────▼──────┐
       │   Worker    │  funboost task consumer
       │  code_ai/*  │  AI inference pipelines
       └─────────────┘
```

## Project Structure

```
.
├── backend/                  # FastAPI backend service
│   └── app/
│       ├── main.py           # Uvicorn entry point
│       ├── server.py         # FastAPI app, CORS, Redis cache, lifespan
│       ├── database.py       # SQLAlchemy async + Advanced Alchemy (PostgreSQL)
│       ├── service.py        # Base repository service, session management
│       ├── config/
│       │   ├── constants.py  # Status enums (StudyStatus, SeriesStatus, InferenceTaskStatus)
│       │   └── deps.py       # Dynamic filter/pagination dependency injection
│       ├── sync/             # Core DICOM sync module (event-sourced state machine)
│       │   ├── model.py      # DCOPEventModel, DCOPConfModel, StudyPrevLinkModel
│       │   ├── schemas.py    # Pydantic DTOs, DCOPStatus enum, OrthancID validator
│       │   ├── service.py    # DCOPEventDicomService (orchestrates full workflow)
│       │   └── routers.py    # POST /sync/study, GET /sync/ope_no, etc.
│       ├── series/           # DICOM sequence classification & orientation analysis
│       │   ├── schemas.py    # SeriesResponse, series type dictionaries
│       │   ├── routers.py    # POST /series/dicom/analyze/by-path, by-upload
│       │   └── deps.py       # ConvertManager, ImageOrientationStrategy providers
│       └── rerun/            # Study rerun/retry module
│           ├── service.py    # ReRunStudyService
│           └── routers.py    # POST /rerun/study/by-uid, by-rename_id
│
├── code_ai/                  # AI inference engine & processing pipelines
│   ├── __init__.py           # Environment config loader
│   ├── task/                 # Task queue framework (funboost)
│   │   ├── params.py         # Booster queue params (RabbitMQ, concurrency)
│   │   ├── task_pipeline.py  # task_pipeline_inference() - main entry point
│   │   └── schema/           # Input/output parameter schemas
│   ├── pipeline/             # Inference pipelines
│   │   ├── __init__.py       # PipelineConfig - command builder registry
│   │   ├── pipeline_cmb_tensorflow.py       # Cerebral Microbleed detection
│   │   ├── pipeline_aneurysm_tensorflow.py  # Aneurysm detection (MRA)
│   │   ├── pipeline_infarct_tensorflow.py   # Acute infarction detection (DWI)
│   │   ├── pipeline_wmh_tensorflow.py       # White Matter Hyperintensity
│   │   ├── pipeline_synthseg_tensorflow.py  # SynthSeg brain segmentation
│   │   ├── main.py           # Multi-algorithm CLI processing
│   │   ├── followup/         # Baseline vs follow-up comparison
│   │   ├── dicomseg/         # DICOM-SEG export
│   │   ├── rdx/              # Radiomics data extraction
│   │   └── upload/           # Results upload to API
│   ├── dicom2nii/            # DICOM to NIfTI conversion
│   │   ├── main.py           # Conversion entry point
│   │   └── convert/          # Series renaming (MR/CT), NIfTI conversion
│   ├── utils/
│   │   ├── inference/        # Command builder + config.yaml task mapping
│   │   ├── parcellation/     # Brain region parcellation (CerebroParcellation, NumPy)
│   │   ├── gpu_env.py        # GPU configuration (pynvml)
│   │   └── database.py       # Database utilities
│   ├── SynthSeg/             # Brain tissue segmentation (TensorFlow)
│   ├── ext/                  # External ML libs (VoxelMorph/Neuron, lab2im)
│   ├── utils_synthseg.py     # SynthSeg wrapper (33-label, 5-class)
│   └── utils_parcellation.py # White matter parcellation, CMB/DWI region extraction
│
├── funboost_cli_user.py      # Task worker entry point (funboost consumer)
├── funboost_config.py        # Message broker connection config
├── docker-compose.yml        # Infrastructure services
├── pyproject.toml            # Python dependencies (uv)
├── scripts/                  # Utility & diagnostic scripts
├── docs/                     # Documentation
└── tests/                    # Test suites
```

## Processing Workflow

```
1. Raw DICOM ──► Rename DICOM
   DCOPEventDicomService.get_series_info
   → dicom_to_nii.process_dir
   → post_ope_no_task
   → check_study_series_transfer_complete

2. Rename DICOM ──► Rename NIfTI
   check_study_series_transfer_complete
   → dicom_to_nii.dicom_2_nii_file
   → study_series_nifti_tool
   → check_study_series_conversion_complete

3. Rename NIfTI ──► Pipeline Inference
   check_study_series_conversion_complete
   → task_pipeline_inference (via RabbitMQ)
   → study_series_inference_nifti_tool
   → check_study_series_inference_complete

4. Inference Results ──► Upload
   check_study_upload_complete
```

State transitions tracked via `DCOPEventModel` (event sourcing):

```
Study:     NEW → TRANSFERRING → TRANSFER_COMPLETE → CONVERTING → CONVERSION_COMPLETE
           → INFERENCE_READY → QUEUED → RUNNING → COMPLETE → RESULTS_SENT
```

## Supported Inference Tasks

| Task | Input | Description |
|------|-------|-------------|
| **SynthSeg** | T1/T2 | Brain tissue segmentation (33 labels / 5 classes) |
| **Aneurysm** | MRA_BRAIN | Aneurysm detection from MRA sequences |
| **CMB** | SWI/SWAN | Cerebral microbleed detection (with follow-up comparison) |
| **Infarct** | DWI | Acute infarction detection |
| **WMH** | T2FLAIR | White matter hyperintensity segmentation |
| **Area** | T1 + SynthSeg | Brain area/volume quantification |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Web Framework | FastAPI + Uvicorn |
| ORM | SQLAlchemy 2.0 async + Advanced Alchemy |
| Database | PostgreSQL 16 (asyncpg) |
| Cache | Redis 7 (fastapi-cache2) |
| Task Queue | funboost + RabbitMQ |
| DICOM Archive | Orthanc (with MinIO S3 storage) |
| Deep Learning | TensorFlow 2.14, Keras |
| Medical Imaging | nibabel, SimpleITK, pydicom, pyorthanc |
| Data | pandas, NumPy 1.26, scikit-image |

## Quick Start

### Prerequisites

- Python >= 3.10 (conda `tf_2_14` environment recommended)
- Docker & Docker Compose (for infrastructure services)
- GPU with TensorFlow support (for inference)

### 1. Start Infrastructure

```bash
docker compose up -d
```

This starts: PostgreSQL, Redis, RabbitMQ, MinIO, Orthanc.

### 2. Install Dependencies

```bash
pip install uv
git clone <repo-url> && cd brain-parcellation
uv sync
```

Or with system Python:

```bash
uv pip install -r pyproject.toml --system
```

### 3. Configure Environment

Create a `.env` file with required variables:

```env
# Database
POSTGRES_DB=dicom
POSTGRES_USER=postgres_n
POSTGRES_PASSWORD=postgres_p
POSTGRES_PORT=15433

# Redis
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# RabbitMQ
RABBITMQ_USER=guest
RABBITMQ_PASS=guest
RABBITMQ_HOST=127.0.0.1
RABBITMQ_PORT=5672
RABBITMQ_VIRTUAL_HOST=/

# Processing paths
PATH_PROCESS=/path/to/processing
PATH_JSON=/path/to/json
PATH_LOG=/path/to/logs
UPLOAD_DATA_API_URL=http://127.0.0.1:8000/api/v1
```

### 4. Run Services

```bash
# Terminal 1: Backend API
export PYTHONPATH=$(pwd) && python3 backend/app/main.py

# Terminal 2: Task Worker (inference consumer)
export PYTHONPATH=$(pwd) && python3 funboost_cli_user.py
```

## API Endpoints

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/sync/study` | Submit DICOM study for processing |
| POST | `/sync/study/transfer/complete` | Check transfer completion |
| POST | `/sync/study/convert` | Check conversion completion |
| POST | `/sync/nifti_tool` | Receive NIFTI tool completion report |
| GET | `/sync/ope_no` | Query event log (with search/pagination) |
| GET | `/sync/cache` | List inference task cache |
| DELETE | `/sync/cache` | Clear study cache |
| POST | `/series/dicom/analyze/by-path` | Analyze DICOM by file path |
| POST | `/series/dicom/analyze/by-upload` | Analyze uploaded DICOM files |
| GET | `/series/types` | List supported series types |
| POST | `/rerun/study/by-uid` | Rerun study by UID |
| POST | `/rerun/study/by-rename_id` | Rerun study by rename ID |

## Running Individual Pipelines

```bash
# Aneurysm detection
export PYTHONPATH=$(pwd) && python3 code_ai/pipeline/pipeline_aneurysm_tensorflow.py \
  --ID <study_id> \
  --Inputs /path/to/MRA_BRAIN.nii.gz \
  --Output_folder /path/to/output \
  --InputsDicomDir /path/to/dicom/MRA_BRAIN

# WMH detection
export PYTHONPATH=$(pwd) && python3 code_ai/pipeline/pipeline_wmh_tensorflow.py \
  --ID <study_id> \
  --Inputs /path/to/T2FLAIR_AXI.nii.gz /path/to/WMH_PVS.nii.gz /path/to/synthseg5.nii.gz \
  --Output_folder /path/to/output \
  --InputsDicomDir /path/to/dicom/T2FLAIR_AXI
```
