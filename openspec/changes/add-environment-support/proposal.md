# Change: Add Production/Testing Environment Support

## Why

當前系統 (`rad_ai_infer` 分支) 僅支援單一執行環境，無法區分 production 和 testing 場景。這導致：
- 無法使用不同的資料路徑進行測試而不影響正式資料
- 日誌輸出缺乏環境區分，debugging 困難
- 無法為不同環境載入不同的模型或配置版本
- 運營風險：測試操作可能誤觸 production 資料

遵循 **Twelve-Factor App** 方法論和 **Ken Thompson 的簡單設計原則**，我們需要一個清晰、最小化的環境支援機制。

## What Changes

按照 **Martin Fowler 的 YAGNI** 和 **Linus 的務實主義**，我們只實作當前明確需要的兩個環境：

1. **環境配置能力 (Environment Configuration)**
   - 透過環境變數 `ENV=production|testing` 控制環境
   - 精確定義每個環境的差異項目（資料路徑、日誌級別、模型配置）
   - 遵循 **Knuth 的精確性原則**：每個配置項都有明確的邊界條件

2. **Backend 環境感知 (Backend Environment Awareness)**
   - FastAPI backend 能讀取並響應環境配置
   - 根據環境載入對應的資料庫連接和資料路徑
   - 日誌系統根據環境調整輸出級別

3. **部署環境支援 (Deployment Environment Support)**
   - 啟動腳本 `brain-parcellation-start.sh` 支援環境參數
   - Docker Compose 配置支援環境變數注入
   - 保持 **Linus 的資料結構優先**：配置即資料，清晰可檢視

## Impact

**受影響的 specs:**
- `environment-config` (NEW) - 環境配置機制
- `backend-env` (NEW) - Backend 環境感知
- `deployment-env` (NEW) - 部署環境支援

**受影響的程式碼:**
- `back/main.py` - FastAPI 應用入口
- `back/database.py` - 資料庫配置
- `pipelinecore/src/pipelinecore/inference/config.py` - 推理配置載入
- `brain-parcellation-start.sh` - 啟動腳本
- `docker-compose.yml` - Docker 編排配置

**設計原則:**
- **Ken Thompson**: 從基礎元件（環境變數）向上構建，不引入框架
- **Linus**: 資料結構優先，配置即資料，清晰可見
- **Martin Fowler**: YAGNI - 只做 production/testing，不預設其他環境
- **Knuth**: 精確定義每個配置邊界，文檔化環境差異

**非破壞性變更:**
- 預設行為保持不變（未設定 `ENV` 時視為 production）
- 現有程式碼向後相容
- 可逐步遷移，無需一次性改動所有模組
