# Change: Refactor to Pure Functions

## Why

當前系統 (`backend/` 和 `code_ai/`) 中存在 350+ 次環境變數直接讀取 (`os.getenv()`)，導致:
- **可測試性差**: 函數行為依賴全域狀態,無法隔離測試
- **Side Effects 氾濫**: 150+ 函數含有副作用,違反函數式程式設計原則
- **配置混亂**: 環境變數散落各處,缺乏集中管理和型別安全
- **部署風險**: 配置錯誤在執行時才發現,而非編譯時或啟動時
- **重構困難**: 緊耦合的環境依賴使得程式碼難以重構和優化

遵循 **Knuth 的精確性原則**、**Linus 的資料結構優先**和 **"We do not break userspace"** 原則,我們需要將所有副作用隔離到系統邊界。

## What Changes

按照 **Martin Fowler 的 Refactoring** 和 **Robert Martin 的 Clean Architecture**,我們將實施零風險的純函數重構:

### 1. **配置集中化與型別安全 (Config Centralization)**
   - 建立 `BackendConfig` 和 `CodeAIConfig` 不可變 dataclass
   - 所有環境變數讀取集中在 `load_*_config_from_env()` 函數
   - 型別安全: 使用 Pydantic 或 dataclass 進行執行時驗證
   - 遵循 **Knuth 原則**: 每個配置項明確型別、邊界、預設值

### 2. **依賴注入 (Dependency Injection)**
   - 所有服務類別透過建構函數接收 `config` 參數
   - 消除函數內部的 `os.getenv()` 調用
   - Pipeline 支援雙模式: `config=None` (舊模式) 與 `config=injected` (新模式)
   - 目標: 350+ → 2 次環境變數讀取 (減少 99.4%)

### 3. **零風險向後相容 (Zero-Risk Backward Compatibility)**
   - **Adapter Pattern**: 保持舊類別簽名永久有效,內部路由到新實作
   - **Three-Phase Migration**: 新增 → 並行 → 可選棄用 (永不強制)
   - **Fail-Safe Loading**: Production 永不崩潰,Testing 嚴格驗證
   - **Contract Testing**: 自動化證明 `OLD ≡ NEW` 行為等價
   - **Feature Flags**: `USE_NEW_CONFIG=false` 瞬間回滾 (< 10 秒)
   - 遵循 **Linus "We do not break userspace"** 原則

### 4. **多層防護回滾 (Defense-in-Depth Rollback)**
   - Layer 1: Feature Flag 即時回滾 (< 10 秒)
   - Layer 2: Git Revert 精確回滾 (5-10 分鐘)
   - Layer 3: Git Tag Phase Reset (10-15 分鐘)
   - 展現工程成熟度,遵循 Linux Kernel、Google SRE 標準

## Impact

**受影響的 specs:**
- `config-management` (NEW) - 集中化配置管理與型別安全
- `dependency-injection` (NEW) - 依賴注入模式
- `backward-compatibility` (NEW) - 向後相容保證機制
- `pure-function-architecture` (NEW) - 純函數架構原則

**受影響的程式碼:**

Backend (150+ 環境變數讀取):
- `backend/app/config/` (NEW) - 配置載入器與 dataclass
- `backend/app/services/sync.py` - SyncService 適配器
- `backend/app/sync/server.py` - 服務入口點
- `backend/app/*.py` - 所有依賴環境變數的模組

Code_AI (200+ 環境變數讀取):
- `code_ai/config/` (NEW) - CodeAIConfig 系統
- `code_ai/pipeline/*.py` - 7 個 Pipeline 類別
- `code_ai/pipeline/base.py` - BasePipeline 雙模式支援

**設計原則:**
- **Knuth**: 精確型別定義,配置即文檔,先正確後效能
- **Linus (資料結構)**: 好的資料結構使程式碼自然簡單
- **Linus (向後相容)**: "We do not break userspace" - 外部介面永久穩定
- **Martin Fowler**: 小步重構,持續驗證,保持系統可運行
- **Robert Martin**: 依賴倒置,介面隔離,單一職責

**非破壞性變更:**
- ✅ **Adapter Pattern**: 舊類別永久有效 (`SyncService()` 永遠可用)
- ✅ **雙模式支援**: Pipeline 支援 `config=None` 與 `config=injected`
- ✅ **契約測試**: 自動化驗證 OLD ≡ NEW 行為等價
- ✅ **Feature Flags**: 瞬間回滾能力,零部署時間
- ✅ **漸進式遷移**: 一次重構一個 Pipeline,逐步驗證

**量化改進:**
- 環境變數讀取: 350+ → 2 (減少 99.4%)
- Side Effects 函數: 150+ → 0 (減少 100%)
- 配置管理: 分散 → 集中化 (15+ 類別型別安全)
- 測試隔離: 不可能 → 完全隔離 (mock config 而非環境)
- 啟動驗證: 執行時 → 啟動時 (配置錯誤立即發現)

**參考文件:**
- `openspec/changes/add-environment-support/PURE_FUNCTION_REFACTORING_PLAN.md` - 完整重構計劃與技術細節
- `openspec/changes/add-environment-support/KNUTH_LINUS_DESIGN.md` - 設計哲學與原則
