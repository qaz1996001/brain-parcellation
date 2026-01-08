# Capability: Environment Configuration

## ADDED Requirements

### Requirement: Environment Detection
系統 SHALL 能透過 `ENV` 環境變數檢測當前運行環境，並載入對應的環境配置。

#### Scenario: 預設環境為 production
- **GIVEN** 環境變數 `ENV` 未設定
- **WHEN** 系統啟動並呼叫 `get_environment()`
- **THEN** 返回 `"production"`

#### Scenario: 明確指定 testing 環境
- **GIVEN** 環境變數 `ENV=testing`
- **WHEN** 系統啟動並呼叫 `get_environment()`
- **THEN** 返回 `"testing"`

#### Scenario: 無效環境值處理
- **GIVEN** 環境變數 `ENV=invalid`
- **WHEN** 系統啟動並呼叫 `get_environment()`
- **THEN** 拋出 `EnvironmentError` 並提示有效值為 `production|testing`

### Requirement: 環境配置映射
系統 SHALL 為每個支援的環境提供明確定義的配置映射，包含資料路徑、日誌級別和模型配置路徑。

#### Scenario: Production 環境配置結構
- **GIVEN** 環境為 `production`
- **WHEN** 呼叫 `get_config()`
- **THEN** 返回配置包含:
  - `data_root`: 指向 production 資料目錄
  - `log_level`: 設為 `"INFO"`
  - `model_config_path`: 指向 production 模型配置

#### Scenario: Testing 環境配置結構
- **GIVEN** 環境為 `testing`
- **WHEN** 呼叫 `get_config()`
- **THEN** 返回配置包含:
  - `data_root`: 指向 testing 資料目錄 (與 production 隔離)
  - `log_level`: 設為 `"DEBUG"`
  - `model_config_path`: 指向 testing 模型配置

#### Scenario: 配置鍵完整性驗證
- **GIVEN** 任意支援的環境
- **WHEN** 載入環境配置
- **THEN** 配置 SHALL 包含所有必要鍵: `data_root`, `log_level`, `model_config_path`
- **AND** 缺少任何鍵時 SHALL 拋出 `ConfigurationError`

### Requirement: 環境配置不可變性
系統 SHALL 確保環境配置在程序啟動後不可更改，避免執行時環境切換導致的不一致性。

#### Scenario: 執行時環境鎖定
- **GIVEN** 系統已使用 `production` 環境啟動
- **WHEN** 程序執行期間嘗試修改 `ENV` 環境變數為 `testing`
- **THEN** 系統配置 SHALL 維持 `production` (啟動時的值)
- **AND** 日誌 SHALL 記錄警告訊息關於環境變更嘗試

#### Scenario: 配置物件唯讀
- **GIVEN** 已取得環境配置物件
- **WHEN** 嘗試修改配置值 (如 `config.data_root = "/new/path"`)
- **THEN** 操作 SHALL 失敗或無效 (使用 immutable data structure)

### Requirement: 環境資訊可觀測性
系統 SHALL 在啟動時和透過 API 提供當前環境資訊，以支援監控和 debugging。

#### Scenario: 啟動日誌記錄環境
- **GIVEN** 系統使用任意環境啟動
- **WHEN** 應用完成初始化
- **THEN** 日誌 SHALL 包含 `INFO` 級別訊息: `"Environment: {environment}"`
- **AND** 訊息 SHALL 在其他應用日誌之前輸出

#### Scenario: Health 端點包含環境資訊
- **GIVEN** 系統正在運行於 `testing` 環境
- **WHEN** 呼叫 `GET /health` API 端點
- **THEN** 回應 SHALL 包含 `{"environment": "testing"}` 欄位
- **AND** HTTP 狀態碼為 200

#### Scenario: 環境資訊審計
- **GIVEN** 系統已運行
- **WHEN** 檢視系統日誌
- **THEN** 能透過 grep "Environment:" 找到環境資訊
- **AND** 環境資訊在日誌中唯一且不可偽造
