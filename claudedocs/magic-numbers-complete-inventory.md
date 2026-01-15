# Complete Magic Numbers & Strings Inventory

## Summary Statistics

| Category | Count |
|----------|-------|
| Magic Numbers | 150+ |
| Magic Strings | 80+ |
| Hardcoded Credentials | 1 (CRITICAL) |
| Hardcoded User Paths | 1 (CRITICAL) |
| Functions > 24 lines | 40+ |
| Functions > 100 lines | 10+ |

---

## Critical Security Issues

### 1. Hardcoded Database Credentials
**File**: `backend/app/config/loader.py`
```python
"postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom"
```

### 2. Hardcoded User Path
**File**: `code_ai/utils_synthsegOnnx.py`
```python
"/home/seanho/fsl/bin/flirt"
```

---

## Magic Numbers by Category

### HTTP Timeouts (seconds)
| Value | Count | Files |
|-------|-------|-------|
| `180` | 10+ | sync, study, inference service.py |
| `300` | 8+ | task_dicom2nii.py, scheduler, orthanc |
| `600` | 5+ | task_pipeline.py subprocess timeout |
| `3600` | 3+ | async result timeout |

### Redis TTL (seconds)
| Value | Meaning | Files |
|-------|---------|-------|
| `21600` | 6 hours | inference, sync, study |
| `1800` | 30 minutes | RPC result expire |

### Queue Processing Rates (qps)
| Value | Purpose | Files |
|-------|---------|-------|
| `1` | GPU mutex | task_pipeline.py |
| `5` | HTTP posting | task_dicom2nii.py |
| `10` | Standard processing | multiple |
| `100` | High throughput | process_instances |

### File Sizes (bytes)
| Value | Purpose | Files |
|-------|---------|-------|
| `500` | Min NIfTI size | task_dicom2nii.py (duplicated) |

### Pagination
| Value | Purpose | Files |
|-------|---------|-------|
| `10` | Default limit | rerun |
| `20` | Batch size | sync |
| `50` | Page size | sync |

### Neural Network Architecture
| Value | Purpose | Files |
|-------|---------|-------|
| `5` | UNet levels | synthseg, predict.py |
| `24` | UNet features | synthseg |
| `2` | Conv per level | synthseg |
| `3` | Conv kernel | synthseg |
| `192` | Input/crop size | synthseg (typo: intput_size) |
| `128` | Min padding | synthseg, parcellation |

### Image Processing
| Value | Purpose | Files |
|-------|---------|-------|
| `0.5` | Gaussian sigma | synthseg |
| `0.25` | Posterior threshold | synthseg |
| `0.1` | Mask threshold | parcellation |
| `0.9` | Completeness threshold | scheduler |
| `0.05` | Resolution tolerance | predict.py |

### Medical Imaging Labels
| Value | Purpose | Files |
|-------|---------|-------|
| `16` | Brain stem (FreeSurfer) | parcellation |
| `3`, `42` | Left/right cortex | parcellation |
| `301` | Brain stem output | parcellation |

### Validation
| Value | Purpose | Files |
|-------|---------|-------|
| `36` | UUID length | task_pipeline.py |
| `4` | UUID hyphen count | task_pipeline.py |
| `999999` | Sentinel "infinity" | parcellation (6x) |

---

## Magic Strings by Category

### Tool Identifiers
| String | Count | Should Be |
|--------|-------|-----------|
| `"DICOM_TOOL"` | 4+ | `tools.dicom` |
| `"NIFTI_TOOL"` | 4+ | `tools.nifti` |
| `"INFERENCE_TOOL"` | 3+ | `tools.inference` |
| `"SERIES_INFERENCE_TOOL"` | 2+ | `tools.series_inference` |

### Queue Names
| String | Count |
|--------|-------|
| `"task_pipeline_inference_queue"` | 5+ |
| `"dicom_2_nii_file_queue"` | 2 |
| `"dicom_2_nii_series_queue"` | 2 |
| `"add_raw_dicom_to_nii_inference_queue"` | 2 |

### Status Codes
| String | Purpose |
|--------|---------|
| `"100.020"` | Pending |
| `"200.100"` | Running |
| `"200.150"` | Complete |
| `"300.055"` | Failed |

### File Extensions
| String | Count |
|--------|-------|
| `".nii.gz"` | 10+ |
| `".nii"` | 5+ |
| `".json"` | 5+ |
| `"_david.nii.gz"` | 3 |

### Model Files
| String | File |
|--------|------|
| `"synthseg_2.0.h5"` | utils_synthseg.py |
| `"synthseg_robust_2.0.h5"` | utils_synthseg.py |
| `"synthseg_parc_2.0.h5"` | utils_synthseg.py |

### Legacy UUID Mappings (SHOULD BE IN CONFIG)
| UUID | Model |
|------|-------|
| `48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6` | CMB |
| `924d1538-597c-41d6-bc27-4b0b359111cf` | Aneurysm |
| `7e94d381-3f5d-46b6-b440-e5d44ebc48d2` | WMH |
| `97abe75d-34de-4e91-80c2-ce74b6c70438` | Infarct |

---

## Environment Variables (Keep as Env)

These should remain as environment variables but be documented:

| Variable | Module |
|----------|--------|
| `PATH_RAW_DICOM` | scheduler |
| `PATH_RENAME_DICOM` | scheduler |
| `PATH_RENAME_NIFTI` | scheduler |
| `PATH_SYNTHSEG` | code_ai config |
| `GPU_N` | code_ai config |
| `UPLOAD_DATA_API_URL` | multiple |
| `UPLOAD_DATA_DICOM_SEG_URL` | task_dicom2nii |
| `AI_APP_PORT` | backend config |
| `AI_APP_CONNECTION_STRING` | backend config |

---

## DICOM Tags (Consider Constants)

| Tag | Purpose |
|-----|---------|
| `(0x20, 0x13)` | Instance Number |
| `(0x20, 0x1002)` | Images In Acquisition |
| `(0x08, 0x60)` | Modality |
| `(0x10, 0x20)` | Patient ID |
| `(0x08, 0x50)` | Accession Number |
| `(0x08, 0x20)` | Study Date |
| `(0x0020, 0x000E)` | Series Instance UID |

---

## Typos Found

| Location | Wrong | Correct |
|----------|-------|---------|
| utils_synthseg*.py | `intput_size` | `input_size` |
| sync/service.py | `"stydy"` | `"study"` |
| scheduler_database.py | `delete_old_date` | `delete_old_data` |
| scheduler_check_add_task.py | `clinet` | `client` |
| rerun/service.py | `flage` | `flag` |
| task/schema/ | `intput_params.py` | `input_params.py` |
