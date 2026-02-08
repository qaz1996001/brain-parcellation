# Process Flow - 處理流程圖

## Brain Parcellation System - Redesign

> 版本: 2.0 | 日期: 2026-02-08

---

## 1. 端到端主流程 (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MAIN PROCESS FLOW                               │
│                                                                     │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────────┐   │
│  │ DICOM   │    │ Convert │    │ Resolve  │    │   Inference  │   │
│  │ Receive ├───▶│ to NIfTI├───▶│ Pipeline ├───▶│   Execute    │   │
│  │ & Scan  │    │         │    │ & Tasks  │    │  (Funboost)  │   │
│  └─────────┘    └─────────┘    └──────────┘    └──────┬───────┘   │
│                                                        │           │
│                                                        ▼           │
│                                        ┌──────────────────────┐   │
│  ┌──────────────┐    ┌────────────┐    │   Post-process &    │   │
│  │  Upload to   │◀───│  Generate  │◀───│   Record Results    │   │
│  │  PACS/Orthanc│    │  DICOM-SEG │    │                     │   │
│  └──────────────┘    └────────────┘    └──────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 流程 A: DICOM 接收與註冊

```
                    START
                      │
                      ▼
            ┌──────────────────┐
            │ Scheduler 輪詢   │
            │ PATH_RAW_DICOM   │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐     ┌─────┐
            │ 偵測到新的 DICOM  │─NO──▶│ Wait│──→ (loop)
            │ 目錄？            │     └─────┘
            └────────┬─────────┘
                     │ YES
                     ▼
            ┌──────────────────┐
            │ 解析 DICOM Header│
            │ 取得 study_uid   │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐     ┌──────────────┐
            │ DB: Study 已存在？│─YES─▶│ 跳過 (已處理)│
            └────────┬─────────┘     └──────────────┘
                     │ NO
                     ▼
        ┌────────────────────────┐
        │ DB: INSERT study       │
        │ status = NEW           │
        │ EventLog: NEW          │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ 掃描 DICOM 子目錄      │
        │ 識別各 Series 類型     │
        │ (T1, T2, DWI, MRA...) │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ 對每個 Series:         │
        │  ├─ INSERT series      │
        │  ├─ Rename DICOM files │
        │  └─ status = NEW       │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ Study status:          │
        │ NEW → TRANSFERRING     │
        │    → TRANSFERRED       │
        │ EventLog: TRANSFERRED  │
        └────────────┬───────────┘
                     │
                     ▼
              觸發 流程 B
```

---

## 3. 流程 B: DICOM → NIfTI 轉換

```
         FROM 流程 A (Study TRANSFERRED)
                     │
                     ▼
        ┌────────────────────────┐
        │ Study status:          │
        │ TRANSFERRED→CONVERTING │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ 取得 Study 下所有      │
        │ Series (status=NEW)    │
        └────────────┬───────────┘
                     │
                     ▼
           ┌─────────────────┐
           │ FOR each Series │◀─────────────────┐
           └────────┬────────┘                   │
                    │                            │
                    ▼                            │
         ┌───────────────────┐                   │
         │ Series status:    │                   │
         │ NEW → CONVERTING  │                   │
         └────────┬──────────┘                   │
                  │                              │
                  ▼                              │
         ┌───────────────────┐                   │
         │ 執行 dcm2niix     │                   │
         │ DICOM → NIfTI     │                   │
         └────────┬──────────┘                   │
                  │                              │
             ┌────┴────┐                         │
             ▼         ▼                         │
        ┌────────┐ ┌────────────┐                │
        │ 成功   │ │ 失敗/跳過  │                │
        │        │ │            │                │
        │ status:│ │ status:    │                │
        │CONVERTED│CONVERSION_ │                │
        │        │ │SKIPPED     │                │
        └────┬───┘ └─────┬──────┘                │
             │           │                       │
             └─────┬─────┘                       │
                   │                             │
                   ▼                             │
         ┌───────────────────┐                   │
         │ 更新 series record│                   │
         │ nifti_path = path │                   │
         └────────┬──────────┘                   │
                  │                              │
                  ▼                              │
         ┌───────────────────┐     YES           │
         │ 還有未處理 Series?├───────────────────┘
         └────────┬──────────┘
                  │ NO
                  ▼
        ┌────────────────────────┐
        │ Study status:          │
        │ CONVERTING → CONVERTED │
        │ study.nifti_path = dir │
        └────────────┬───────────┘
                     │
                     ▼
              觸發 流程 C
```

---

## 4. 流程 C: Pipeline 解析與 Task 建立

```
         FROM 流程 B (Study CONVERTED)
                     │
                     ▼
        ┌────────────────────────┐
        │ 載入所有 active        │
        │ Pipeline 定義          │
        │ (from DB pipeline表)   │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ 取得 Study 的所有      │
        │ CONVERTED Series       │
        │ available_types = set  │
        └────────────┬───────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │ FOR each Pipeline   │◀─────────────────┐
           └────────┬────────────┘                   │
                    │                                │
                    ▼                                │
         ┌───────────────────────┐                   │
         │ Pipeline.required_    │                   │
         │ series 是否滿足?      │                   │
         │                       │                   │
         │ e.g., Aneurysm 需要   │                   │
         │ [MRA_BRAIN]           │                   │
         │                       │                   │
         │ CMB 需要              │                   │
         │ [SWAN + T1BRAVO_AXI]  │                   │
         └────────┬──────────────┘                   │
                  │                                  │
             ┌────┴────┐                             │
             ▼         ▼                             │
        ┌────────┐ ┌────────┐                        │
        │ 滿足   │ │ 不滿足 │──── 跳過 ─────────────┤
        └────┬───┘ └────────┘                        │
             │                                       │
             ▼                                       │
   ┌──────────────────────┐                          │
   │ 解析 input_files:    │                          │
   │ 從 series.nifti_path │                          │
   │ 組合輸入檔案路徑     │                          │
   └──────────┬───────────┘                          │
              │                                      │
              ▼                                      │
   ┌──────────────────────┐                          │
   │ DB: INSERT task      │                          │
   │ status = PENDING     │                          │
   │ input_files = [...]  │                          │
   │ pipeline_id = FK     │                          │
   └──────────┬───────────┘                          │
              │                                      │
              ▼                                      │
   ┌───────────────────────┐     YES                 │
   │ 還有未處理 Pipeline?  ├─────────────────────────┘
   └──────────┬────────────┘
              │ NO
              ▼
   ┌──────────────────────┐
   │ Study status:         │
   │ CONVERTED →           │
   │ INFERENCE_READY       │
   └──────────┬────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ FOR each PENDING Task│
   │  ├─ Push to RabbitMQ │
   │  ├─ task.status =    │
   │  │  QUEUED            │
   │  └─ task.queued_at = │
   │     now()             │
   └──────────┬────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Study status:         │
   │ INFERENCE_READY →     │
   │ INFERENCE_RUNNING     │
   └──────────┬────────────┘
              │
              ▼
        觸發 流程 D
```

---

## 5. 流程 D: Funboost 消費者推論執行

```
         FROM RabbitMQ Queue
                │
                ▼
    ┌───────────────────────┐
    │ Funboost Consumer     │
    │ 從佇列取出 task msg   │
    │ {task_db_id, paths}   │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ DB: Task status       │
    │ QUEUED → RUNNING      │
    │ task.started_at=now() │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ Pipeline 需要          │         ┌──────────────────────────┐
    │ SynthSeg 前處理?      │──YES───▶│ SynthSeg Pre-processing  │
    │ (Area, WMH_PVS, CMB, │         │ Chain of Responsibility  │
    │  DWI)                 │         │                          │
    └───────────┬───────────┘         │  Resample                │
                │ NO                  │    ↓                     │
                │                     │  SynthSeg (GPU)          │
                │                     │    ↓                     │
                │                     │  Process SynthSeg        │
                │                     │    ↓                     │
                │                     │  Save Files              │
                │                     │    ↓                     │
                │                     │  Post-process            │
                │                     │    ↓                     │
                │              ┌──────│  Resample to Original    │
                │              │      └──────────────────────────┘
                │              │
                ▼              ▼
    ┌───────────────────────────────┐
    │ build_inference_cmd()         │
    │ 建立推論命令                   │
    │                               │
    │ e.g., python3                 │
    │   pipeline_aneurysm_tf.py    │
    │   --ID {study_id}             │
    │   --Inputs {nifti_paths}      │
    │   --Output_folder {out_dir}   │
    │   --InputsDicomDir {dicom}    │
    └───────────┬───────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │ FOR each cmd_item:    │◀─────────────────┐
    │ subprocess.Popen()    │                   │
    │ 執行推論 (GPU)        │                   │
    └───────────┬───────────┘                   │
                │                               │
           ┌────┴────┐                          │
           ▼         ▼                          │
      ┌────────┐ ┌────────┐                     │
      │ 成功   │ │ 失敗   │                     │
      │ rc=0   │ │ rc!=0  │                     │
      └───┬────┘ └───┬────┘                     │
          │          │                          │
          └────┬─────┘                          │
               │                                │
               ▼                                │
      ┌────────────────────┐     YES            │
      │ 還有 cmd_item?     ├────────────────────┘
      └────────┬───────────┘
               │ NO
               ▼
    ┌───────────────────────────────┐
    │ Funboost 回調:                │
    │ save_task_result_to_db()     │
    │                               │
    │ INSERT task_result            │
    │  ├─ attempt_number            │
    │  ├─ success/failure           │
    │  ├─ stdout/stderr             │
    │  ├─ time_cost                 │
    │  └─ funboost_task_id          │
    │                               │
    │ UPDATE task                   │
    │  ├─ status = COMPLETED|FAILED │
    │  └─ completed_at = now()      │
    └───────────┬───────────────────┘
                │
           ┌────┴────┐
           ▼         ▼
    ┌──────────┐ ┌──────────────────┐
    │ COMPLETED│ │      FAILED      │
    │          │ │                  │
    │          │ │ attempt < max?   │
    │          │ │  YES → auto retry│
    │          │ │  NO → DLX queue  │
    └────┬─────┘ └───────┬──────────┘
         │               │
         └───────┬───────┘
                 │
                 ▼
    ┌───────────────────────────────┐
    │ 檢查 Study 下所有 Task       │
    │                               │
    │ ALL COMPLETED?                │
    │  → Study: INFERENCE_COMPLETE │
    │                               │
    │ ANY FAILED?                   │
    │  → Study: FAILED (可 rerun)  │
    │                               │
    │ STILL RUNNING?                │
    │  → 等待其他 Task              │
    └───────────┬───────────────────┘
                │
                ▼
          觸發 流程 E
```

---

## 6. 流程 E: 結果上傳

```
         FROM 流程 D (Study INFERENCE_COMPLETE)
                     │
                     ▼
        ┌────────────────────────┐
        │ 收集所有推論結果        │
        │ - Pred_*.nii.gz       │
        │ - Pred_*.json         │
        │ - synthseg_*.nii.gz   │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ 將 NIfTI 結果轉為      │
        │ DICOM-SEG 格式         │
        │ (upload_dicom_seg.py)  │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ 上傳 DICOM-SEG 到      │
        │ Orthanc PACS Server    │
        │ via REST API           │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ Study status:          │
        │ INFERENCE_COMPLETE     │
        │ → RESULTS_SENT         │
        │                        │
        │ EventLog: RESULTS_SENT │
        └────────────┬───────────┘
                     │
                     ▼
                    END
```

---

## 7. 流程 F: 失敗重試 (Rerun)

```
         User/API 觸發 Rerun
                │
                ▼
    ┌───────────────────────┐
    │ POST /api/v1/tasks/   │
    │ {task_id}/rerun       │
    │                       │
    │ 或                    │
    │                       │
    │ POST /api/v1/studies/ │
    │ {study_id}/rerun      │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ 驗證狀態機：           │
    │ FAILED → QUEUED       │
    │ (allowed)             │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ Task:                 │
    │ status = QUEUED       │
    │ error_message = null  │
    │ queued_at = now()     │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ 重新推入 RabbitMQ     │
    │ 佇列                  │
    │                       │
    │ EventLog:             │
    │ FAILED → QUEUED       │
    │ operator = "user"     │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ 如果是 Study rerun:   │
    │ Study status:         │
    │ FAILED → NEW          │
    │ 重新執行全部流程      │
    └───────────────────────┘
```

---

## 8. 系統交互序列圖

### 8.1 正常流程序列

```
Scheduler       API Server       PostgreSQL      RabbitMQ      Funboost        GPU Pipeline
   │                │                │              │          Consumer            │
   │  detect DICOM  │                │              │              │               │
   ├───────────────▶│                │              │              │               │
   │                │  INSERT study  │              │              │               │
   │                ├───────────────▶│              │              │               │
   │                │  INSERT series │              │              │               │
   │                ├───────────────▶│              │              │               │
   │                │                │              │              │               │
   │  convert NIfTI │                │              │              │               │
   ├───────────────▶│                │              │              │               │
   │                │ UPDATE series  │              │              │               │
   │                ├───────────────▶│              │              │               │
   │                │ UPDATE study   │              │              │               │
   │                ├───────────────▶│              │              │               │
   │                │                │              │              │               │
   │                │ resolve tasks  │              │              │               │
   │                ├────┐           │              │              │               │
   │                │    │           │              │              │               │
   │                │◀───┘           │              │              │               │
   │                │ INSERT tasks   │              │              │               │
   │                ├───────────────▶│              │              │               │
   │                │                │              │              │               │
   │                │ push task msg  │              │              │               │
   │                ├──────────────────────────────▶│              │               │
   │                │                │              │              │               │
   │                │                │              │  consume msg │               │
   │                │                │              ├─────────────▶│               │
   │                │                │              │              │               │
   │                │                │  UPDATE task │              │               │
   │                │                │  → RUNNING   │              │               │
   │                │                │◀─────────────┤              │               │
   │                │                │              │              │               │
   │                │                │              │   subprocess │               │
   │                │                │              │   inference  │               │
   │                │                │              │  ───────────▶│               │
   │                │                │              │              │  result files │
   │                │                │              │  ◀───────────│               │
   │                │                │              │              │               │
   │                │                │ INSERT result│              │               │
   │                │                │◀─────────────┤              │               │
   │                │                │ UPDATE task  │              │               │
   │                │                │ → COMPLETED  │              │               │
   │                │                │◀─────────────┤              │               │
   │                │                │              │              │               │
   │                │                │ UPDATE study │              │               │
   │                │                │ → COMPLETE   │              │               │
   │                │                │◀─────────────┤              │               │
```

---

## 9. Funboost 消費者部署拓撲

```
┌────────────────────────────────────────────────────────────────┐
│                      Production Deployment                      │
│                                                                  │
│  ┌──────────────────┐     ┌──────────────────┐                 │
│  │  API Server      │     │  PostgreSQL      │                 │
│  │  (FastAPI)       │────▶│  Database        │                 │
│  │  :8000           │     │  :5432           │                 │
│  └──────────────────┘     └──────────────────┘                 │
│           │                        ▲                            │
│           │                        │                            │
│           ▼                        │                            │
│  ┌──────────────────┐              │                            │
│  │  RabbitMQ        │              │                            │
│  │  Broker          │              │                            │
│  │  :5672 / :15672  │              │                            │
│  └────────┬─────────┘              │                            │
│           │                        │                            │
│     ┌─────┼─────────┐             │                            │
│     │     │         │             │                            │
│     ▼     ▼         ▼             │                            │
│  ┌──────┐ ┌──────┐ ┌──────┐      │                            │
│  │GPU-1 │ │GPU-2 │ │IO    │      │                            │
│  │Worker│ │Worker│ │Worker│──────┘                            │
│  │      │ │      │ │      │                                    │
│  │Infer │ │Synth │ │DICOM │  (直接寫 DB，不再透過 HTTP POST)   │
│  │ence  │ │Seg   │ │Conv  │                                    │
│  └──────┘ └──────┘ └──────┘                                    │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │  Redis           │                                           │
│  │  (Heartbeat/RPC) │                                           │
│  │  :6379           │                                           │
│  └──────────────────┘                                           │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │  Scheduler       │   (定期掃描新 DICOM，觸發處理流程)         │
│  │  (Cron/Loop)     │                                           │
│  └──────────────────┘                                           │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## 10. 錯誤處理流程

```
          任何步驟發生錯誤
                │
                ▼
    ┌───────────────────────┐
    │ 錯誤類型判斷           │
    └───────────┬───────────┘
                │
    ┌───────────┼───────────────────┐
    ▼           ▼                   ▼
┌────────┐ ┌──────────┐     ┌─────────────┐
│暫時性   │ │永久性     │     │ 資源不足    │
│(網路)   │ │(資料錯誤) │     │ (GPU OOM)   │
└───┬────┘ └─────┬────┘     └──────┬──────┘
    │            │                 │
    ▼            ▼                 ▼
┌────────┐ ┌──────────┐     ┌─────────────┐
│自動重試 │ │標記FAILED│     │ 標記 FAILED  │
│(3 次)  │ │不再重試   │     │ 記錄 OOM    │
│間隔 20s│ │通知管理員 │     │ 待手動處理   │
└───┬────┘ └──────────┘     └─────────────┘
    │
    ▼
┌──────────────────┐
│ 重試成功?         │
│ YES → COMPLETED  │
│ NO  → 送入 DLX   │
│       (死信佇列)  │
└──────────────────┘
```

---

## 11. 新舊流程對比摘要

| 環節 | 舊流程 | 新流程 | 改善 |
|------|--------|--------|------|
| 資料追蹤 | 單一 event 表 + ope_no 比較 | 正規化 study/series/task 表 + 狀態機 | 查詢效率、資料完整性 |
| 狀態管理 | 字串拼接 PK + 數值比較 | UUID PK + Enum 狀態 + StateMachine 驗證 | 型別安全、轉換規則明確 |
| 任務回報 | HTTP POST 回 API server | Funboost 回調直接寫 DB | 減少延遲、消除網路依賴 |
| 任務依賴 | Chain of Responsibility 阻塞 | 保留 Chain 模式但改用非同步 | 資源利用率 |
| Pipeline 定義 | 硬編碼在 Python 字典 | DB pipeline 表 + YAML config | 動態配置，開放封閉原則 |
| 錯誤處理 | 無標準化 | StateMachine + DLX + EventLog 審計 | 可追溯、可恢復 |
| 審計日誌 | 無 | event_log (append-only) | 完整歷史記錄 |

---
