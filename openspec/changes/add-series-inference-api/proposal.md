# 變更提案: 新增 Series 級別推論 API

## 為什麼需要這個變更

目前系統僅支援 study 級別的推論，會一次處理 study 中的所有 series。這在多模型場景中造成不靈活：
- 不同模型需要針對同一 study 中的特定 series 執行
- 使用者想要觸發特定 series 的推論，而不是處理整個 study
- 推論結果需要按模型追蹤，而非按 study 追蹤

radax server 需要一個 API 來觸發 study 內特定 series 的推論，並指定模型，支援批次請求和部分處理（當某些 series 尚未準備好時）。

## ⚠️ 關鍵約束：不可修改既有程式碼

**使用者需求**：
> "設計不可以影響舊的使用空間，意思就是 sync、task_pipeline_inference 都不可以有大改動，儘可以程式碼重構"
>
> "新的設計復用舊的 → OK"
>
> "大改舊的設計 → NG"

**設計原則**：
- ✅ 建立新模組 `backend/app/inference/`（與既有 `sync/` 平行）
- ✅ 建立新的 `task_series_inference.py`（與既有 `task_pipeline.py` 分離）
- ✅ 對既有檔案僅做最小修改（import/export）
- ✅ 單一 process 循序消費兩個 queue（不修改既有程式碼）
- ❌ 不可建立統一 queue（會需要重構既有程式碼）
- ❌ 不可修改 `sync/service.py` 或 `task_pipeline.py`

## 變更內容

- **新增 API 模組**: `backend/app/inference/` 提供 series 級別推論端點
- **批次請求支援**: 單一 API 呼叫可接受多個 study 請求
- **部分處理**: 立即處理已準備好的 series，在回應中記錄被拒絕的 series
- **模型選擇**: 支援 model_name + model_version 或 model_id (UUID)
- **Redis 快取**: 在 24 小時內防止重複的推論請求
- **新的 Task Queue**: `series_inference_queue` 用於 series 級別 GPU 推論
- **狀態追蹤**: 新增 DCOP 狀態碼用於 series 推論生命週期
- **回呼整合**: 符合 radax 需求的成功/失敗回報

### 破壞性變更
無 - 這是與既有 study 級別推論並行運作的新功能。

## 影響範圍

### 受影響的規格
- **新增**: `series-inference` 能力（series 級別推論編排）

### 受影響的程式碼
**新建檔案** (7 個檔案):
- `backend/app/inference/routers.py` - FastAPI 端點
- `backend/app/inference/service.py` - 業務邏輯服務
- `backend/app/inference/schemas.py` - 請求/回應模型
- `backend/app/inference/model.py` - 資料庫模型
- `backend/app/inference/deps.py` - 依賴注入
- `backend/app/inference/urls.py` - URL 模式
- `code_ai/task/task_series_inference.py` - Funboost task queue

**修改檔案** (5 個檔案 - 僅最小修改):
- `backend/app/main.py` - 註冊 inference router (2-3 行)
- `backend/app/sync/schemas.py` - 擴充 DCOPStatus enum (5 行)
- `code_ai/task/__init__.py` - 匯出新 task (1 行)
- `funboost_cli_user.py` - 啟動兩個 queue 消費者 (1 行)
- `backend/app/config/deps.py` - 新增 Redis 依賴（如需要，約 10 行）

**禁止修改**:
- ❌ `backend/app/sync/service.py` - 重用模式，不可修改
- ❌ `code_ai/task/task_pipeline.py` - 絕對不可動
- ❌ 任何既有的 DCOP 模型或資料庫 schema

### 架構影響
- **Queue 整合**: 新 queue 透過循序消費與既有 study 級別 queue 共享 GPU 資源
- **資料庫**: 使用既有的 DCOP 事件追蹤系統（不修改）
- **快取層**: 引入 Redis 快取用於推論去重
- **部署**: 透過參數注入模式支援雙重部署（重用既有模式）

### GPU 競爭解決方案
**唯一可行的方法**（符合不修改既有程式碼約束）：
```python
# funboost_cli_user.py - 單一 process 循序消費
BoostersManager.consume_queues(
    'task_pipeline_inference_queue',  # 既有的（不修改）
    'series_inference_queue'           # 新建的（獨立建立）
)
```

**原理**：
- 單一 process 循序輪詢兩個 queue
- 不需修改既有的 `task_pipeline_inference.py`
- 兩個 task 都使用 SOLO 模式（每個 queue 一次只執行一個）
- 循序輪詢自然提供 GPU 共享

**限制**：
- 不是真正的全域 qps=1 控制
- 輪詢順序可能偏向某個 queue
- 單一 process（但 GPU 本身就是瓶頸）

### 效能考量
- **GPU 排程**: 透過循序消費確保公平排程（不修改既有程式碼）
- **快取效能**: Redis 查詢增加約 5ms 延遲，防止重複 GPU 工作
- **資料庫負載**: 重用既有 DCOP 事件系統，無新資料表
- **API 延遲**: 批次驗證根據 series 數量增加 10-50ms

### 遷移計畫
1. 部署新的 `inference` 模組（與既有 `sync` 模組並行）
2. 在 FastAPI app 中註冊新 router
3. 部署新的 task queue 消費者（可在相同 GPU worker 上執行）
4. 新增 SERIES_INFERENCE_* 狀態碼到 DCOP 設定
5. 不需要資料遷移 - 使用既有資料表

### 回滾策略
1. 從 main.py 移除 inference router
2. 停止 series_inference_queue 消費者
3. 清除 Redis 快取項目（選用）
4. 移除 DCOP 狀態設定項目（選用）
5. 不需要資料庫回滾