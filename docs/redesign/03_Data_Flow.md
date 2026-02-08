# Data Flow - 資料流程圖

## Brain Parcellation System - Redesign

> 版本: 2.0 | 日期: 2026-02-08

---

## 1. 系統層級資料流 (Context DFD - Level 0)

```
                           ┌─────────────┐
                           │   Orthanc   │
                           │ PACS Server │
                           └──────┬──────┘
                      DICOM files │ ▲ DICOM-SEG results
                                  ▼ │
┌──────────┐        ┌──────────────────────────────┐        ┌──────────────┐
│  Medical │        │                              │        │   Dashboard  │
│  Scanner ├──DICOM─▶  Brain Parcellation System   ├──JSON──▶   / Client  │
│  (MRI)   │        │                              │        │   Web UI     │
└──────────┘        └──────────────────────────────┘        └──────────────┘
                              │           ▲
                    Task msgs │           │ Results
                              ▼           │
                         ┌─────────────┐
                         │  RabbitMQ   │
                         │   Broker    │
                         └─────────────┘
```

---

## 2. 主要資料流 (Level 1 DFD)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  External                                                               │
│  ┌──────────┐                                                          │
│  │ Orthanc  │                                                          │
│  │  /PACS   │                                                          │
│  └────┬─────┘                                                          │
│       │ DICOM Study                                                    │
│       ▼                                                                 │
│  ┌─────────────────┐     ┌──────────────┐     ┌───────────────┐        │
│  │  1.0 Receive &  │     │              │     │               │        │
│  │  Register Study ├────▶│  [study] DB  │◀───▶│  2.0 Convert  │        │
│  │  (Scheduler)    │     │  [series] DB │     │  DICOM→NIfTI  │        │
│  └─────────────────┘     └──────┬───────┘     └───────┬───────┘        │
│                                 │                     │                 │
│                                 │                     │ NIfTI files     │
│                                 │                     ▼                 │
│                          ┌──────┴───────┐     ┌───────────────┐        │
│                          │              │     │ 3.0 Resolve   │        │
│                          │  [task] DB   │◀────│ Pipelines &   │        │
│                          │              │     │ Create Tasks  │        │
│                          └──────┬───────┘     └───────────────┘        │
│                                 │                                       │
│                                 │ task params                           │
│                                 ▼                                       │
│                          ┌─────────────┐                               │
│                          │  RabbitMQ   │                               │
│                          │  Queue      │                               │
│                          └──────┬──────┘                               │
│                                 │                                       │
│                                 ▼                                       │
│                   ┌────────────────────────┐                           │
│                   │  4.0 Funboost Consumer │                           │
│                   │  (Inference Execution) │                           │
│                   └────────┬───────────────┘                           │
│                            │                                            │
│              ┌─────────────┼──────────────┐                            │
│              ▼             ▼              ▼                             │
│        ┌──────────┐ ┌──────────┐ ┌──────────┐                         │
│        │ Pipeline │ │ Pipeline │ │ Pipeline │  ... (7 pipelines)      │
│        │ Aneurysm │ │   WMH    │ │   CMB    │                         │
│        └────┬─────┘ └────┬─────┘ └────┬─────┘                         │
│             │            │            │                                 │
│             └────────────┼────────────┘                                │
│                          │ result files (NIfTI, JSON)                  │
│                          ▼                                              │
│               ┌────────────────────┐     ┌──────────────────┐          │
│               │ 5.0 Post-process & │     │                  │          │
│               │ Record Results     ├────▶│ [task_result] DB │          │
│               └────────┬───────────┘     │ [event_log] DB   │          │
│                        │                 └──────────────────┘          │
│                        │ DICOM-SEG                                     │
│                        ▼                                                │
│               ┌────────────────────┐                                   │
│               │ 6.0 Upload Results │                                   │
│               │ to PACS (Orthanc)  │                                   │
│               └────────────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 各處理程序的資料流細節

### 3.1 Process 1.0 - Receive & Register Study

```
輸入:
  ├── Raw DICOM directory (from Orthanc/PACS)
  └── DICOM metadata (study_uid, patient info)

處理:
  ├── Scheduler 偵測新 DICOM 目錄
  ├── 解析 DICOM header 取得 study_uid
  ├── 建立 Study 記錄 (status=NEW)
  ├── 逐一建立 Series 記錄
  └── DICOM rename (標準化檔名)

輸出:
  ├── study record → PostgreSQL [study] table
  ├── series records → PostgreSQL [series] table
  ├── event_log entry (study NEW→TRANSFERRING)
  └── Renamed DICOM files → filesystem
```

### 3.2 Process 2.0 - Convert DICOM→NIfTI

```
輸入:
  ├── Renamed DICOM directory (from Process 1.0)
  └── Series records (from DB)

處理:
  ├── 對每個 Series 執行 dcm2niix 轉換
  ├── 更新 Series status (CONVERTING → CONVERTED)
  ├── 跳過無法轉換的 Series (→ CONVERSION_SKIPPED)
  └── 所有 Series 完成後，更新 Study status → CONVERTED

輸出:
  ├── NIfTI files → filesystem (nifti_path)
  ├── Updated series.nifti_path → DB
  ├── Updated series.status → DB
  ├── Updated study.nifti_path → DB
  └── event_log entries
```

### 3.3 Process 3.0 - Resolve Pipelines & Create Tasks

```
輸入:
  ├── Study record (status=CONVERTED)
  ├── Series records (with series_type)
  └── Pipeline definitions (active pipelines)

處理:
  ├── 掃描 Study 的所有 CONVERTED Series
  ├── 對每個 active Pipeline:
  │   ├── 檢查 required_series 是否滿足
  │   ├── 滿足 → 建立 Task 記錄 (status=PENDING)
  │   └── 不滿足 → 跳過
  ├── 更新 Study status → INFERENCE_READY
  └── 將所有 PENDING Task 推入 RabbitMQ 佇列

輸出:
  ├── Task records → PostgreSQL [task] table
  ├── Task messages → RabbitMQ queues
  ├── Updated study.status → DB
  └── event_log entries (task PENDING→QUEUED)
```

### 3.4 Process 4.0 - Funboost Consumer (Inference)

```
輸入:
  ├── Task message from RabbitMQ (task_db_id, paths, params)
  └── NIfTI input files (from filesystem)

處理:
  ├── Consumer 從佇列取出 task message
  ├── 更新 Task status → RUNNING
  ├── 根據 Pipeline 類型執行對應推論:
  │   ├── SynthSeg 前處理 (if needed)
  │   │   ├── Resample → SynthSeg → Process → Save
  │   │   └── Chain of Responsibility pattern
  │   └── 推論 subprocess 執行
  │       ├── python3 pipeline_aneurysm_tensorflow.py --args
  │       ├── python3 pipeline_wmh_tensorflow.py --args
  │       └── ... (根據 pipeline type)
  └── Funboost 回調 → save_task_result_to_db()

輸出:
  ├── Result NIfTI files → filesystem (output_files)
  ├── Result JSON files → filesystem
  ├── TaskResult record → PostgreSQL [task_result] table
  ├── Updated task.status → DB (COMPLETED | FAILED)
  └── event_log entries
```

### 3.5 Process 5.0 - Post-process & Record Results

```
輸入:
  ├── Inference output files (NIfTI, JSON)
  ├── Task record (status=COMPLETED)
  └── Original DICOM series data

處理:
  ├── Resample back to original image space
  ├── Generate DICOM-SEG from NIfTI results
  ├── 彙整 Study 下所有 Task 狀態
  ├── 所有 Task COMPLETED → Study status → INFERENCE_COMPLETE
  └── 任何 Task FAILED → Study status → FAILED (partial)

輸出:
  ├── DICOM-SEG files → filesystem
  ├── Updated study.status → DB
  └── event_log entries
```

### 3.6 Process 6.0 - Upload Results to PACS

```
輸入:
  ├── DICOM-SEG result files
  └── Study record (status=INFERENCE_COMPLETE)

處理:
  ├── Upload DICOM-SEG to Orthanc via REST API
  └── 更新 Study status → RESULTS_SENT

輸出:
  ├── DICOM-SEG → Orthanc PACS
  ├── Updated study.status → DB
  └── event_log entry (INFERENCE_COMPLETE → RESULTS_SENT)
```

---

## 4. 資料儲存 (Data Stores)

### 4.1 PostgreSQL 資料庫

| 表 | 讀取方 | 寫入方 | 資料量預估 |
|----|--------|--------|-----------|
| `study` | API, Scheduler, Consumer | Scheduler, API | ~100/天 |
| `series` | API, Consumer | Scheduler | ~500/天 (5 series/study) |
| `pipeline` | Consumer, API | Admin (seed data) | 8 筆 (固定) |
| `task` | API, Consumer | Service Layer | ~300/天 (3 tasks/study) |
| `task_result` | API | Funboost Callback | ~300/天 |
| `event_log` | API (審計) | 所有 Service | ~2000/天 |

### 4.2 Filesystem 資料

```
PATH_RAW_DICOM/           ← 原始 DICOM (輸入)
  └── {patient_id}/
      └── {study_date}/

PATH_RENAME_DICOM/        ← 重新命名後的 DICOM
  └── {study_id}/
      ├── T1BRAVO_AXI/
      ├── T2FLAIR_AXI/
      ├── MRA_BRAIN/
      └── ...

PATH_RENAME_NIFTI/        ← NIfTI 轉換結果
  └── {study_id}/
      ├── T1BRAVO_AXI.nii.gz
      ├── T2FLAIR_AXI.nii.gz
      └── ...

PATH_PROCESS/             ← 推論中間結果與最終結果
  └── {study_id}/
      ├── Deep_cmd_tools/    (命令 JSON)
      ├── Pred_Aneurysm.nii.gz
      ├── Pred_WMH.nii.gz
      ├── Pred_CMB.nii.gz
      └── ...
```

### 4.3 RabbitMQ 佇列

| Queue Name | Producer | Consumer | 用途 |
|-----------|----------|----------|------|
| `task_pipeline_inference_queue` | TaskService | Funboost Consumer (GPU) | 主要推論任務 |
| `resample_task_queue` | Workflow | Funboost Consumer | 影像 resample |
| `synthseg_task_queue` | Workflow | Funboost Consumer (GPU) | SynthSeg 分割 |
| `process_synthseg_task_queue` | Workflow | Funboost Consumer | SynthSeg 後處理 |
| `task_subprocess_queue` | Misc | Funboost Consumer | 通用 subprocess |
| `*.dlx` (Dead Letter) | RabbitMQ | Manual retry | 失敗任務暫存 |

---

## 5. 資料流與設計模式對應

| Data Flow | 設計模式 | 說明 |
|-----------|---------|------|
| DICOM → Study/Series records | **Repository** | 資料存取透過 StudyRepository 抽象 |
| Study → 決定哪些 Pipeline 可執行 | **Strategy** | 每個 Pipeline 的 required_series 匹配邏輯 |
| Task → RabbitMQ message | **Command** | Task 封裝為可序列化的佇列訊息 |
| RabbitMQ → Consumer → subprocess | **Factory Method** | 根據 pipeline type 建立對應的推論命令 |
| Resample → SynthSeg → Process → Save | **Chain of Responsibility** | 前處理管線的鏈式處理 |
| 狀態變更 → event_log | **Observer** | EventPublisher 通知多個 listener |
| Task status transitions | **State** | StateMachine 驗證合法轉換 |
| API → Service → Repository → DB | **Facade / Layered** | 分層架構隱藏內部複雜度 |

---
