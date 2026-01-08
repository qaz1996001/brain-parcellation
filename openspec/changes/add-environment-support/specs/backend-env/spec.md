# Capability: Backend Environment Awareness

## ADDED Requirements

### Requirement: 資料庫路徑環境隔離
Backend 系統 SHALL 根據環境配置使用不同的資料庫連接路徑，確保 production 和 testing 資料完全隔離。

#### Scenario: Production 資料庫路徑
- **GIVEN** 環境為 `production`
- **WHEN** 系統初始化資料庫連接
- **THEN** 資料庫路徑 SHALL 使用 production 配置的 `data_root`
- **AND** 路徑 SHALL 不包含 "test" 或 "testing" 字樣

#### Scenario: Testing 資料庫路徑
- **GIVEN** 環境為 `testing`
- **WHEN** 系統初始化資料庫連接
- **THEN** 資料庫路徑 SHALL 使用 testing 配置的 `data_root`
- **AND** 路徑 SHALL 與 production 路徑不同

#### Scenario: 資料隔離驗證
- **GIVEN** 系統在 testing 環境中寫入資料
- **WHEN** 切換環境為 production 並重啟系統
- **THEN** production 環境 SHALL 無法讀取 testing 資料
- **AND** 兩個環境的資料庫檔案位於不同目錄

### Requirement: 日誌級別環境感知
Backend 日誌系統 SHALL 根據環境自動調整輸出級別，testing 環境提供詳細 debug 資訊，production 環境僅記錄重要事件。

#### Scenario: Testing 環境 DEBUG 級別日誌
- **GIVEN** 環境為 `testing`
- **WHEN** 系統記錄 DEBUG 級別訊息
- **THEN** 訊息 SHALL 出現在日誌輸出中
- **AND** 包含詳細的變數值和堆疊追蹤

#### Scenario: Production 環境過濾 DEBUG 日誌
- **GIVEN** 環境為 `production`
- **WHEN** 系統記錄 DEBUG 級別訊息
- **THEN** 訊息 SHALL NOT 出現在日誌輸出中
- **AND** 僅 INFO 及以上級別訊息被記錄

#### Scenario: 日誌級別一致性
- **GIVEN** 環境已設定為 `testing`
- **WHEN** 系統中所有模組 (main, database, routes) 記錄日誌
- **THEN** 所有模組 SHALL 使用統一的 DEBUG 級別
- **AND** 不存在模組使用不同日誌級別的情況

### Requirement: FastAPI 應用環境整合
FastAPI 應用 SHALL 在啟動時載入環境配置並在整個生命週期中使用該配置。

#### Scenario: 應用啟動載入環境
- **GIVEN** `ENV=testing` 環境變數已設定
- **WHEN** 執行 `python back/main.py` 啟動 FastAPI 應用
- **THEN** 應用 SHALL 成功啟動
- **AND** 啟動日誌 SHALL 顯示 "Environment: testing"

#### Scenario: 環境配置注入依賴
- **GIVEN** FastAPI 應用已啟動於 `production` 環境
- **WHEN** 請求處理函數呼叫 `get_config()` 取得配置
- **THEN** 返回的配置 SHALL 反映 production 設定
- **AND** 配置在所有請求間保持一致

#### Scenario: 向後相容性保證
- **GIVEN** 未設定 `ENV` 環境變數 (舊版啟動方式)
- **WHEN** 啟動 FastAPI 應用
- **THEN** 應用 SHALL 成功啟動
- **AND** 預設使用 production 配置
- **AND** 所有現有 API 端點 SHALL 正常運作

### Requirement: 環境資訊 API 暴露
Backend SHALL 提供 API 端點允許外部系統查詢當前運行環境，支援監控和自動化工具。

#### Scenario: Health Check 包含環境
- **GIVEN** 系統運行於任意環境
- **WHEN** 呼叫 `GET /health` 端點
- **THEN** 回應 JSON SHALL 包含 `"environment"` 欄位
- **AND** 值為 `"production"` 或 `"testing"`
- **AND** HTTP 狀態碼為 200

#### Scenario: 環境資訊僅可讀
- **GIVEN** 系統正在運行
- **WHEN** 嘗試透過 API 修改環境 (如 `POST /environment`)
- **THEN** 該端點 SHALL 不存在 (404 Not Found)
- **AND** 環境在執行時不可透過 API 更改
