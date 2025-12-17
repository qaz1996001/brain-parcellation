# @sync 模組架構概覽

## 🏗️ 分層架構

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Routes (routers.py)           │
│  ├─ POST /sync/study         → 排程新 Study 同步       │
│  ├─ POST /sync/ope_no        → 批次寫入事件            │
│  ├─ POST /sync/nifti_tool    → 接收外部工具回報        │
│  ├─ GET  /sync/cache         → 查詢推論快取            │
│  └─ GET  /sync/query/*       → 各種狀態查詢            │
└──────────────────────────────┬──────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────┐
│              Service Layer (service.py)                 │
│  ├─ DCOPEventDicomService                              │
│  │  ├─ schedule_new_studies()                           │
│  │  ├─ check_study_series_transfer_complete()           │
│  │  ├─ check_study_series_conversion_complete()         │
│  │  ├─ study_series_nifti_tool()                        │
│  │  └─ Various query methods...                         │
│  └─ BaseRepositoryService                              │
└──────────────────────────────┬──────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────┐
│              Database Models (model.py)                 │
│  ├─ DCOPConfModel      (設定映射表)                    │
│  ├─ DCOPEventModel     (事件紀錄表)                    │
│  └─ StudyPrevLinkModel (時序鏈結表)                   │
└──────────────────────────────┬──────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────┐
│                  SQLAlchemy ORM                         │
│              PostgreSQL / MySQL Database                │
└─────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│         Validation Layer (schemas.py - Pydantic)        │
│  ├─ PostStudyRequest              (API 請求驗證)       │
│  ├─ DCOPEventRequest              (事件驗證)           │
│  ├─ DCOPEventNIFTITOOLRequest     (外部工具驗證)       │
│  ├─ StydySeriesOpeNoStatus        (查詢結果驗證)       │
│  ├─ DCOPStatus (列舉)              (狀態碼定義)        │
│  └─ validate_orthanc_id()          (UID 驗證器)        │
└─────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│         External Integration Points                      │
│  ├─ Orthanc API (DICOM 伺服器)                          │
│  ├─ NIFTI_TOOL (轉檔工具)                               │
│  ├─ Inference Engine (推論服務)                         │
│  └─ Redis (快取層)                                      │
└──────────────────────────────────────────────────────────┘
```

## 📊 狀態轉遷流程

### 完整生命週期

```
1️⃣  STUDY_NEW
    ↓
2️⃣  STUDY_TRANSFERRING
    ↓
3️⃣  SERIES_TRANSFERRING (多個 Series 並行)
    ├─ Series 1: SERIES_NEW → TRANSFERRING → TRANSFER_COMPLETE
    ├─ Series 2: SERIES_NEW → TRANSFERRING → TRANSFER_COMPLETE
    └─ Series N: SERIES_NEW → TRANSFERRING → TRANSFER_COMPLETE
    ↓
4️⃣  [CheckPoint] STUDY_TRANSFER_COMPLETE
    ↓ (自動轉遷)
5️⃣  STUDY_CONVERTING
    ↓
6️⃣  SERIES_CONVERTING (多個 Series 並行)
    ├─ Series 1: CONVERTING → CONVERSION_COMPLETE
    ├─ Series 2: CONVERTING → CONVERSION_SKIP (不支援格式)
    └─ Series N: CONVERTING → CONVERSION_COMPLETE
    ↓
7️⃣  [CheckPoint] STUDY_CONVERSION_COMPLETE
    ↓ (自動轉遷)
8️⃣  STUDY_INFERENCE_READY → QUEUED → RUNNING
    ↓
9️⃣  STUDY_INFERENCE_COMPLETE
    ↓
🔟  STUDY_RESULTS_SENT (完成)
```

### 錯誤路徑

```
任何階段失敗
    ↓
STUDY_INFERENCE_FAILED (終止狀態)
    ↓
[可選重試]
    ↓
STUDY_NEW_RE / STUDY_TRANSFERRING_RE / ...
    ↓
重新進入主流程
```

## 🔄 API 調用流程

### Scenario 1: 新 Study 到達

```
外部系統
    │
    ├─→ POST /sync/study
    │   └─ PostStudyRequest(study_uid="abc-123")
    │
Client ↓ (立即返回)
    │
背景任務
    ├─ schedule_new_studies()
    │  └─ 建立 STUDY_NEW 事件
    │
    ├─ dicom_tool_get_series_info()
    │  └─ 從 Orthanc 擷取 Series 列表
    │     └─ 建立 SERIES_NEW 事件
    │
    └─ [可選] link_prev_study()
       └─ 建立時序鏈結
```

### Scenario 2: 傳輸完成檢查

```
Orthanc/DICOM Tool
    │
    └─→ POST /sync/ope_no
        └─ [DCOPEventRequest, ...]  (多個 Series 完成事件)
    │
Client ↓ (立即返回)
    │
背景任務
    ├─ post_ope_no_task()
    │  └─ 寫入所有事件到資料庫
    │
    └─ 檢查邏輯
       └─ 若所有 Series 已 TRANSFER_COMPLETE
          └─ 呼叫 check_study_series_transfer_complete()
             └─ 建立 STUDY_CONVERTING 事件
```

### Scenario 3: NIFTI 轉檔完成

```
NIFTI_TOOL 伺服器
    │
    └─→ POST /sync/nifti_tool
        └─ [DCOPEventNIFTITOOLRequest, ...]
    │
Client ↓ (立即返回)
    │
背景任務
    ├─ study_series_nifti_tool()
    │  ├─ 反向查詢 ope_no 取得 Study/Series
    │  ├─ 建立 SERIES_CONVERSION_COMPLETE 事件
    │  └─ 儲存結果資料 (output_path 等)
    │
    └─ 檢查邏輯
       └─ 若該 Study 的所有 Series 已完成轉檔
          └─ 建立 STUDY_CONVERSION_COMPLETE 事件
             └─ 排程推論工作 (入隊)
```

## 🗂️ 檔案結構

```
backend/app/sync/
├── __init__.py              (模組初始化)
├── model.py                 (資料庫模型) ★ 已文檔化
├── schemas.py               (Pydantic 驗證) ★ 已文檔化
├── routers.py               (FastAPI 路由) ★ 已文檔化
├── service.py               (業務邏輯服務)
├── urls.py                  (URL 常數定義)
├── deps.py                  (依賴注入)
├── settings.py              (設定管理)
└── DOCUMENTATION_UPDATE.md  (本次更新說明) ★ 新增
```

## 📝 核心概念

### 1. Checkpoint API 設計
- `POST /sync/study/transfer` 
- `POST /sync/study/convert`
  
這兩個端點是 "推動狀態轉遷" 的關鍵，它們：
1. 檢查所有 Series/Study 是否達到某個條件
2. 若達成，自動進入下一階段
3. 排程後續工作

### 2. 鬆散耦合的外部工具集成
NIFTI_TOOL 通過最小 payload 回報：
```json
{
    "ope_no": "200.195",
    "tool_id": "NIFTI_TOOL",
    "result_data": {...}
}
```
系統自動根據 `ope_no` 反向查詢找到對應的 Study/Series

### 3. 事件驅動的狀態機
- 每個狀態轉遷都產生一條事件紀錄
- 完整的審計日誌
- 支援狀態重放和故障排查

## 🔐 安全性設計

### Orthanc ID 驗證
```python
# 自動驗證 40 碼十六進位格式
study_uid: OrthancID = "ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30"
```

### OpeNo 驗證
```python
# 自動驗證 xxx.xxx 格式
ope_no: OpeNo = "100.095"
```

## 📈 效能優化

### 索引策略
```sql
-- dcop_event_bt 表索引
CREATE INDEX idx_study_uid ON dcop_event_bt(study_uid);
CREATE INDEX idx_ope_no ON dcop_event_bt(ope_no);
CREATE INDEX idx_tool_id ON dcop_event_bt(tool_id);
CREATE INDEX idx_create_time ON dcop_event_bt(create_time DESC);
```

### 快取層（Redis）
- 推論任務隊列快取
- 格式：`{prefix}:{study_uid},{study_id}`
- 支援按 Study 清除快取

## 🧪 測試重點

1. **狀態轉遷**: 驗證各 checkpoint 的邏輯
2. **異步工作**: 驗證後台任務的正確性
3. **資料一致性**: 驗證事件紀錄的完整性
4. **外部整合**: 驗證 Orthanc、NIFTI_TOOL 的互動

## 🚨 常見問題

### Q: 為什麼使用 "推動" 而非自動轉遷？
A: 提供控制點。支援：
- 手動檢查
- 條件觸發
- 外部系統協調

### Q: 為什麼需要 `study_prev_link` 表？
A: 支援追蹤患者的系列掃描：基線→隨訪→對比研究

### Q: 為什麼 NIFTI_TOOL 只提供 ope_no？
A: 最小化耦合。外部工具無需知道系統內部架構

---

**架構設計原則**: Good Taste (消除特殊情況、資料結構驅動、扁平邏輯)
**狀態碼標準**: 3 位十進位 xxx.xxx 格式
**事件驅動**: 所有狀態轉遷都可審計

