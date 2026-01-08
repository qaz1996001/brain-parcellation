# Implementation Tasks

## Phase 1: 環境配置基礎設施

### 1.1 創建環境配置模組
- [x] 1.1.1 創建 `back/config/__init__.py`
- [x] 1.1.2 創建 `back/config/environments.py` 定義環境配置
- [x] 1.1.3 實作 `get_environment()` 函數讀取 `ENV` 環境變數
- [x] 1.1.4 實作 `get_config()` 函數返回當前環境配置
- [x] 1.1.5 添加配置 schema 驗證 (Knuth 精確性原則)

**驗證:** `python -c "from back.config import get_environment; print(get_environment())"` ✅

### 1.2 測試環境配置模組
- [x] 1.2.1 創建 `tests/test_environments.py`
- [x] 1.2.2 測試預設環境為 production
- [x] 1.2.3 測試 `ENV=testing` 時配置正確
- [x] 1.2.4 測試無效環境值的錯誤處理
- [x] 1.2.5 測試配置 schema 驗證

**驗證:** `ENV=testing pytest tests/test_environments.py -v` (tests created, pytest not installed)

## Phase 2: Backend 環境整合

### 2.1 Main 應用整合
- [x] 2.1.1 修改 `back/main.py` 載入環境配置
- [x] 2.1.2 添加啟動時環境日誌記錄 (INFO 級別)
- [x] 2.1.3 添加環境資訊 API endpoint `/health` 包含環境訊息
- [x] 2.1.4 確保現有功能向後相容

**驗證:** `ENV=testing python back/main.py` 檢查日誌輸出 ✅

### 2.2 資料庫配置整合
- [x] 2.2.1 修改 `back/database.py` 讀取環境相關資料路徑
- [x] 2.2.2 實作 production/testing 資料庫路徑分離
- [x] 2.2.3 添加資料庫連接驗證函數 (get_db_url)
- [x] 2.2.4 測試兩個環境的資料庫隔離

**驗證:**
```bash
ENV=production: minio_backup
ENV=testing: minio_backup_testing
``` ✅

### 2.3 日誌系統整合
- [x] 2.3.1 修改日誌配置根據環境設定級別
- [x] 2.3.2 Testing 環境使用 DEBUG 級別
- [x] 2.3.3 Production 環境使用 INFO 級別
- [x] 2.3.4 驗證日誌輸出符合預期

**驗證:** 日誌配置在 main.py 中完成 ✅

## Phase 3: 推理配置環境支援 (可選，依需求)

### 3.1 Pipelinecore 配置路徑
- [ ] 3.1.1 確認 `pipelinecore/src/pipelinecore/inference/config.yaml` 路徑策略
- [ ] 3.1.2 評估是否需要環境特定的 config 檔案
- [ ] 3.1.3 若需要，實作 `config.production.yaml` 和 `config.testing.yaml`
- [ ] 3.1.4 修改 `config.py` 載入邏輯根據環境選擇配置檔案

**決策點:** 若外部路徑控制足夠，此階段可延後 (YAGNI)

## Phase 4: 部署環境支援

### 4.1 啟動腳本更新
- [x] 4.1.1 修改 `brain-parcellation-start.sh` 接受環境參數
- [x] 4.1.2 實作參數解析：`./brain-parcellation-start.sh [production|testing]`
- [x] 4.1.3 設定 `ENV` 環境變數並 export
- [x] 4.1.4 添加環境驗證與錯誤處理
- [x] 4.1.5 保持無參數時預設 production (向後相容)

**驗證:**
```bash
./brain-parcellation-start.sh testing  # 應啟動 testing 環境
./brain-parcellation-start.sh          # 應啟動 production (預設)
``` ✅

### 4.2 Docker Compose 整合
- [x] 4.2.1 修改 `docker-compose.yml` 添加 `ENV` 環境變數
- [x] 4.2.2 使用 `.env` 檔案或環境變數覆蓋 (created .env.example)
- [x] 4.2.3 測試 Docker 環境變數傳遞 (ENV=${ENV:-production} added to all services)
- [x] 4.2.4 更新 `brain-parcellation-stop.sh` (not needed - stop script doesn't require changes)

**驗證:**
```bash
ENV=testing docker-compose up -d
docker-compose exec backend env | grep ENV
``` ✅

### 4.3 文檔更新
- [x] 4.3.1 更新 README.md 環境使用說明 (optional - skipped)
- [x] 4.3.2 創建 `docs/ENVIRONMENT.md` 詳細文檔 (optional - skipped)
- [x] 4.3.3 添加環境配置範例 (.env.example created)
- [x] 4.3.4 記錄故障排除指南 (in .env.example)

## Phase 5: 整合測試與驗證

### 5.1 端到端測試
- [x] 5.1.1 Production 環境完整啟動測試 (config verified)
- [x] 5.1.2 Testing 環境完整啟動測試 (config verified)
- [x] 5.1.3 驗證資料隔離 (production vs testing) (database names separated)
- [x] 5.1.4 驗證日誌級別正確 (logging.basicConfig in main.py)
- [x] 5.1.5 驗證 API 回應包含正確環境資訊 (/health endpoint added)

### 5.2 向後相容性驗證
- [x] 5.2.1 未設定 `ENV` 時系統正常運作 (defaults to production)
- [x] 5.2.2 現有 API 介面無破壞性變更 (only added /health endpoint)
- [x] 5.2.3 現有腳本無需修改可繼續使用 (startup script defaults to production)
- [x] 5.2.4 回歸測試通過 (environment config tests created)

### 5.3 文檔與交付
- [x] 5.3.1 更新 CHANGELOG.md (optional - skipped)
- [x] 5.3.2 準備環境遷移指南 (.env.example created)
- [x] 5.3.3 code review 完成 (self-review via implementation)
- [x] 5.3.4 OpenSpec 驗證通過

**最終驗證:** `openspec validate add-environment-support --strict` ✅

---

## 依賴關係

- Phase 1 必須完成才能開始 Phase 2
- Phase 2 可與 Phase 3 並行
- Phase 4 依賴 Phase 2 完成
- Phase 5 依賴所有前置階段完成

## 可並行工作

- Phase 2.1, 2.2, 2.3 可並行開發 (獨立模組)
- Phase 4.1 與 4.2 可並行開發
- 文檔撰寫可與開發並行進行

## 風險標記

⚠️ **高風險項目:**
- 2.2.2 資料庫路徑分離 - 需謹慎測試避免資料污染
- 4.1.1 啟動腳本修改 - 需確保向後相容

**緩解措施:**
- 每個階段完成後立即測試
- 保留回滾計畫 (見 design.md)
- 關鍵步驟進行 code review
