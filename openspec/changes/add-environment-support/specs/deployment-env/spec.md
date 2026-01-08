# Capability: Deployment Environment Support

## ADDED Requirements

### Requirement: 啟動腳本環境參數支援
啟動腳本 `brain-parcellation-start.sh` SHALL 接受環境參數並正確設定 `ENV` 環境變數供後續程序使用。

#### Scenario: 明確指定 production 環境啟動
- **GIVEN** 執行命令 `./brain-parcellation-start.sh production`
- **WHEN** 腳本執行
- **THEN** `ENV=production` 環境變數 SHALL 被設定並 export
- **AND** Backend 和 Funboost 程序 SHALL 繼承該環境變數
- **AND** 日誌 SHALL 記錄 "Starting in production environment"

#### Scenario: 明確指定 testing 環境啟動
- **GIVEN** 執行命令 `./brain-parcellation-start.sh testing`
- **WHEN** 腳本執行
- **THEN** `ENV=testing` 環境變數 SHALL 被設定並 export
- **AND** 所有子程序 SHALL 在 testing 環境中運行
- **AND** 日誌 SHALL 記錄 "Starting in testing environment"

#### Scenario: 預設環境向後相容
- **GIVEN** 執行命令 `./brain-parcellation-start.sh` (無參數)
- **WHEN** 腳本執行
- **THEN** `ENV=production` 環境變數 SHALL 被設定 (預設值)
- **AND** 系統行為 SHALL 與舊版本完全一致

#### Scenario: 無效環境參數錯誤處理
- **GIVEN** 執行命令 `./brain-parcellation-start.sh invalid_env`
- **WHEN** 腳本執行
- **THEN** 腳本 SHALL 輸出錯誤訊息: "Invalid environment. Use: production or testing"
- **AND** 退出碼 SHALL 為非零值 (1)
- **AND** 系統 SHALL NOT 啟動任何服務

### Requirement: Docker Compose 環境變數傳遞
Docker Compose 配置 SHALL 支援透過環境變數控制容器內應用的運行環境。

#### Scenario: 環境變數傳遞至容器
- **GIVEN** 執行 `ENV=testing docker-compose up -d`
- **WHEN** 容器啟動
- **THEN** 容器內 `ENV` 環境變數 SHALL 為 `testing`
- **AND** 可透過 `docker-compose exec backend env | grep ENV` 驗證

#### Scenario: .env 檔案環境配置
- **GIVEN** 存在 `.env` 檔案包含 `ENV=testing`
- **WHEN** 執行 `docker-compose up -d` (不指定環境變數)
- **THEN** 容器 SHALL 使用 `.env` 中的 `ENV=testing`
- **AND** Backend 應用 SHALL 在 testing 環境中運行

#### Scenario: 命令列環境變數優先級
- **GIVEN** `.env` 檔案包含 `ENV=production`
- **WHEN** 執行 `ENV=testing docker-compose up -d`
- **THEN** 命令列的 `ENV=testing` SHALL 覆蓋 `.env` 設定
- **AND** 容器內環境變數 SHALL 為 `testing`

### Requirement: 環境啟動驗證
啟動程序 SHALL 包含環境驗證步驟，確保環境設定正確且一致。

#### Scenario: 啟動時環境驗證通過
- **GIVEN** 所有環境配置正確
- **WHEN** 啟動腳本執行環境驗證
- **THEN** 驗證 SHALL 成功
- **AND** 服務 SHALL 正常啟動
- **AND** 日誌包含 "Environment validation passed"

#### Scenario: 環境配置不一致偵測
- **GIVEN** `ENV=production` 但資料路徑指向 testing 目錄
- **WHEN** 啟動腳本執行環境驗證
- **THEN** 驗證 SHALL 失敗
- **AND** 日誌 SHALL 記錄 "Environment configuration mismatch"
- **AND** 服務 SHALL NOT 啟動

#### Scenario: 環境驗證日誌可追溯
- **GIVEN** 系統已啟動
- **WHEN** 檢視啟動日誌檔案 `/var/log/brain-parcellation/startup.log`
- **THEN** 日誌 SHALL 包含環境驗證結果
- **AND** 包含時間戳、環境名稱、配置路徑等關鍵資訊

### Requirement: 停止腳本環境安全性
停止腳本 `brain-parcellation-stop.sh` SHALL 確保 production 環境資料不被誤刪，僅執行程序終止操作。

#### Scenario: Production 環境停止時保留資料
- **GIVEN** 系統運行於 `production` 環境
- **WHEN** 執行 `./brain-parcellation-stop.sh`
- **THEN** 停止腳本 SHALL NOT 刪除任何 production 資料
- **AND** 僅執行程序終止操作 (停止 backend, funboost)
- **AND** 日誌記錄程序終止資訊

#### Scenario: Testing 環境停止允許清理
- **GIVEN** 系統運行於 `testing` 環境
- **WHEN** 執行 `./brain-parcellation-stop.sh`
- **THEN** 停止腳本 SHALL 終止所有程序
- **AND** MAY 選擇性清理 testing 暫存資料
- **AND** SHALL 記錄所有執行的操作
