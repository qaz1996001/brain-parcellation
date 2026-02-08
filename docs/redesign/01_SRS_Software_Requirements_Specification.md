# SRS - 軟體需求規格書 (Software Requirements Specification)

## Brain Parcellation System - Database Schema & Queue Redesign

> 版本: 2.0 | 日期: 2026-02-08
> 依據: Martin Fowler AI Programming Guide v2

---

## 1. 文件目的

本 SRS 定義 Brain Parcellation System 在資料庫 Schema 重新設計、後端程式重構、以及 Funboost 佇列消費者整合的完整需求規格。

---

## 2. 系統概述

### 2.1 現有系統 (As-Is)

Brain Parcellation System 是一個醫學影像分析系統，處理 DICOM 醫學影像，透過多種 AI 推論管線 (Aneurysm、WMH、CMB、Infarct、Area、DWI) 進行腦部分割與疾病偵測。

**現有架構組件：**
- **FastAPI 後端**: 提供 REST API (series 分析、sync 事件、rerun 管線)
- **PostgreSQL 資料庫**: 儲存事件追蹤 (`dcop_event_bt`, `dcop_conf_bt`)
- **Funboost 任務佇列**: 透過 RabbitMQ 分派推論任務
- **AI 推論管線**: 多個 TensorFlow-based 推論腳本 (subprocess 執行)

### 2.2 現有問題分析

| # | 問題 | 影響 | Fowler 原則違反 |
|---|------|------|----------------|
| 1 | `dcop_event_bt` 單一扁平表混合 study/series 事件 | 查詢複雜、需 stored procedure | **模組化** - 職責不分離 |
| 2 | `VsPrimaryKey` 使用字串拼接作為主鍵 | 無法保證唯一性、難以索引 | **資料管理** - 不良的 key 設計 |
| 3 | 無外鍵關聯 (study → series → task) | 資料一致性無保障 | **資料管理** - 缺乏參照完整性 |
| 4 | 狀態碼用 `ope_no` 數值字串比較 | 邏輯脆弱、難以擴展 | **明確狀態** - 無狀態機 |
| 5 | `params_data`/`result_data` 全部塞 JSON | 無法有效查詢與索引 | **分層架構** - 資料結構不明確 |
| 6 | Backend 用 `advanced_alchemy`、Funboost 用 plain SQLAlchemy | 兩套 ORM 配置不統一 | **架構模組化** - 基礎設施不一致 |
| 7 | Task worker 透過 HTTP POST 回報狀態 | 緊耦合、增加延遲、可能遺失事件 | **鬆耦合** - 應使用 Observer/Event |
| 8 | Chain of Responsibility 中 `.get()` 阻塞等待 | 資源浪費、無法平行處理 | **敏捷開發** - 不支援非同步 |
| 9 | 無重試/錯誤恢復的標準化機制 | 失敗任務難以追蹤與重啟 | **測試自動化** - 缺乏可靠性 |
| 10 | Model 散佈在多個檔案、不同 Base class | 維護困難 | **單一職責** - 程式碼組織混亂 |

---

## 3. 功能需求 (Functional Requirements)

### 3.1 資料庫 Schema 重設計

#### FR-DB-001: Study 管理
- 系統**必須**提供獨立的 Study 實體表，記錄每個 DICOM Study 的基本資訊
- 每個 Study 以 UUID 為主鍵，並記錄 `study_uid` (Orthanc ID)、`study_id` (Rename ID)
- Study 必須有明確的狀態欄位，使用有限狀態機管理生命週期

#### FR-DB-002: Series 管理
- 系統**必須**提供獨立的 Series 實體表，與 Study 建立一對多外鍵關聯
- 每個 Series 記錄 `series_uid`、`series_type` (MRA_BRAIN, T1BRAVO_AXI, etc.)
- Series 有獨立的狀態欄位追蹤轉換進度

#### FR-DB-003: Pipeline (推論管線) 管理
- 系統**必須**提供 Pipeline 定義表，描述可用的推論任務類型
- Pipeline 定義包含：名稱 (Aneurysm, WMH, CMB...)、所需 series 類型、執行命令模板
- Pipeline 可啟用/停用

#### FR-DB-004: Task (任務實例) 管理
- 系統**必須**提供 Task 實體表，記錄每次推論任務的執行
- Task 關聯到 Study 和 Pipeline
- Task 必須有獨立的狀態機: `PENDING → QUEUED → RUNNING → COMPLETED / FAILED`
- Task 記錄輸入檔案路徑、輸出檔案路徑、執行參數

#### FR-DB-005: TaskResult (任務結果) 管理
- 系統**必須**提供 TaskResult 表，儲存每個 Task 的詳細執行結果
- 記錄：執行時間、成功/失敗、stdout/stderr、例外資訊
- 與 Funboost 的 `FunctionResultStatus` 整合

#### FR-DB-006: Event Log (事件日誌)
- 系統**必須**提供不可變 (append-only) 的 EventLog 表
- 記錄所有狀態變更事件，支援審計追蹤
- 每筆事件記錄：entity_type、entity_id、from_status、to_status、timestamp、operator

#### FR-DB-007: 狀態機定義
- 系統**必須**提供 StatusDefinition 配置表 (取代 `dcop_conf_bt`)
- 以 Enum 定義合法狀態及轉換規則
- 支援 study-level 和 task-level 兩套獨立狀態機

### 3.2 後端 API 重設計

#### FR-API-001: Study CRUD API
- `GET /api/v1/studies` - 查詢 Study 列表 (支援篩選、分頁、排序)
- `GET /api/v1/studies/{study_id}` - 取得 Study 詳情 (含 series 列表)
- `POST /api/v1/studies` - 建立新 Study (接收 DICOM 傳入通知)
- `PATCH /api/v1/studies/{study_id}/status` - 更新 Study 狀態

#### FR-API-002: Series API
- `GET /api/v1/studies/{study_id}/series` - 取得 Study 下所有 Series
- `POST /api/v1/series/analyze` - 分析 DICOM Series 類型 (保留現有功能)

#### FR-API-003: Task API
- `GET /api/v1/tasks` - 查詢所有任務 (支援狀態篩選)
- `GET /api/v1/tasks/{task_id}` - 取得任務詳情與結果
- `POST /api/v1/tasks` - 提交新推論任務到佇列
- `POST /api/v1/tasks/{task_id}/rerun` - 重新執行失敗任務

#### FR-API-004: Pipeline API
- `GET /api/v1/pipelines` - 取得可用推論管線列表
- `GET /api/v1/pipelines/{pipeline_name}/tasks` - 取得特定管線的任務列表

#### FR-API-005: Dashboard / 統計 API
- `GET /api/v1/dashboard/stats` - 取得任務統計 (pending/running/completed/failed 計數)
- `GET /api/v1/dashboard/studies/recent` - 近期 Study 處理進度

### 3.3 Funboost 佇列消費者

#### FR-QUEUE-001: 統一 Broker 配置
- 所有任務**必須**使用統一的 RabbitMQ broker
- 配置集中管理，避免散佈在各 task 檔案

#### FR-QUEUE-002: Task 狀態同步
- 任務狀態變更**必須**直接寫入資料庫 (取代 HTTP POST 回報)
- 使用 `user_custom_record_process_info_func` 回調函數同步狀態

#### FR-QUEUE-003: 任務鏈管理
- 支援 DAG (有向無環圖) 形式的任務依賴
- DICOM 傳入 → DICOM 轉 NIfTI → SynthSeg 前處理 → 推論 → 後處理 → 上傳結果
- 每個步驟失敗不應阻塞其他獨立任務

#### FR-QUEUE-004: 重試與死信佇列
- 失敗任務自動重試 (最多 3 次，間隔 20 秒)
- 超過重試上限的任務移入死信佇列 (DLX)
- 死信佇列任務可透過 API 手動重新提交

#### FR-QUEUE-005: 消費者健康監控
- 消費者**必須**發送心跳到 Redis
- 系統可查詢活躍消費者數量與狀態

---

## 4. 非功能需求 (Non-Functional Requirements)

### 4.1 效能 (Performance)
- **NFR-PERF-001**: API 回應時間 < 200ms (不含推論計算)
- **NFR-PERF-002**: 資料庫查詢需有適當索引，避免 full table scan
- **NFR-PERF-003**: 佇列消費者 QPS 依 GPU 資源動態調整

### 4.2 可靠性 (Reliability)
- **NFR-REL-001**: 所有狀態變更必須是原子操作 (transaction)
- **NFR-REL-002**: 推論任務失敗不影響其他任務執行
- **NFR-REL-003**: 系統重啟後可從中斷點恢復未完成任務

### 4.3 可維護性 (Maintainability)
- **NFR-MAINT-001**: 遵循分層架構 (Presentation → Domain → Data Source)
- **NFR-MAINT-002**: 所有 Model 使用統一的 Base class 和命名規範
- **NFR-MAINT-003**: 新增推論管線只需新增配置，不需修改核心程式碼 (開放封閉原則)

### 4.4 可擴展性 (Scalability)
- **NFR-SCALE-001**: 支援多台 GPU 機器的水平擴展 (多個 Funboost consumer)
- **NFR-SCALE-002**: 資料庫 Schema 支援未來新增推論類型

### 4.5 可追溯性 (Traceability)
- **NFR-TRACE-001**: EventLog 保留完整的狀態變更歷史
- **NFR-TRACE-002**: 每個任務可追溯到原始 DICOM、中間產物、最終結果

---

## 5. 約束條件 (Constraints)

| 約束 | 說明 |
|------|------|
| 技術棧 | Python 3.10+, FastAPI, SQLAlchemy 2.0, PostgreSQL, Funboost, RabbitMQ |
| 相容性 | 必須能處理現有的 DICOM Series 類型 (T1, T2, FLAIR, DWI, MRA, SWAN) |
| 推論引擎 | 推論仍以 subprocess 呼叫 TensorFlow 腳本，不改變推論核心邏輯 |
| 部署 | Docker Compose 部署 (PostgreSQL, RabbitMQ, Redis) |
| 向後相容 | 現有的推論管線腳本 (`pipeline_*.py`) 保持不變，僅重構呼叫層 |

---

## 6. 詞彙表 (Glossary)

| 術語 | 定義 |
|------|------|
| **Study** | 一次 MRI 掃描的完整資料集，包含多個 Series |
| **Series** | Study 中的單一影像序列 (如 T1, T2, DWI) |
| **Pipeline** | 一種推論任務類型的定義 (如 Aneurysm Detection) |
| **Task** | Pipeline 的一次具體執行實例 |
| **Funboost** | Python 分散式任務框架，支援多種 Message Broker |
| **SynthSeg** | 腦部分割模型，用於前處理 |
| **NIfTI** | Neuroimaging Informatics Technology Initiative 格式 |
| **DICOM** | Digital Imaging and Communications in Medicine 格式 |
| **Orthanc** | 開源 DICOM/PACS 伺服器 |
| **DLX** | Dead Letter Exchange，RabbitMQ 死信佇列 |

---
