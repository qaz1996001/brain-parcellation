# Capability: Dual Deployment with GPU Resource Management

## ADDED Requirements

### Requirement: 分布式控頻 GPU 資源保護
當 Production 和 Testing 實例同時運行時，系統 SHALL 使用分布式控頻機制確保 GPU 不會過載。

#### Scenario: 啟用分布式控頻配置
- **GIVEN** 配置檔案 `code_ai/task/params.py`
- **WHEN** 設定 `is_using_distributed_frequency_control = True`
- **THEN** Funboost SHALL 啟用分布式控頻功能
- **AND** 每個 worker 的實際 QPS SHALL 為 `qps / active_consumer_num`
- **AND** 可通過 `python -c "from code_ai.task.params import BoosterParamsMyAI; print(BoosterParamsMyAI().is_using_distributed_frequency_control)"` 驗證配置為 `True`

#### Scenario: 雙實例部署 GPU 資源不競爭
- **GIVEN** Production worker (ENV=production, qps=1) 已啟動
- **AND** Testing worker (ENV=testing, qps=1) 已啟動
- **AND** 分布式控頻已啟用 (`is_using_distributed_frequency_control=True`)
- **WHEN** 兩個 worker 同時監聽統一隊列 `task_pipeline_inference_queue`
- **THEN** Redis SHALL 追蹤 `active_consumer_num = 2`
- **AND** 每個 worker 實際 QPS SHALL 為 0.5 (1 / 2)
- **AND** 全局總 QPS SHALL 為 1.0
- **AND** GPU 同時最多執行 1 個推理任務

#### Scenario: GPU 監控驗證無競爭
- **GIVEN** 雙實例部署已運行
- **WHEN** 執行 `./scripts/monitor-gpu-usage.sh`
- **THEN** 輸出 SHALL 顯示 "Active tasks: 1" 或 "Active tasks: 0"
- **AND** SHALL NOT 顯示 "Active tasks: 2" 或更高
- **AND** 日誌 SHALL 記錄 "Distributed frequency control: ACTIVE"

### Requirement: 統一部署指南
系統 SHALL 提供單一統一的部署指南，整合快速啟動和 GPU 資源管理方案。

#### Scenario: 統一部署指南存在
- **GIVEN** 專案根目錄
- **WHEN** 檢查檔案系統
- **THEN** SHALL 存在檔案 `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md`
- **AND** 該檔案 SHALL 包含完整的雙實例部署步驟
- **AND** 該檔案 SHALL 包含 GPU 資源管理方案選擇決策樹

#### Scenario: 部署指南提供明確決策樹
- **GIVEN** 閱讀 `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md`
- **WHEN** 用戶需要選擇部署方案
- **THEN** 文檔 SHALL 提供決策樹
- **AND** 決策樹 SHALL 包含少於 3 個決策點
- **AND** 90% 的場景 SHALL 指向方案 B（分布式控頻）
- **AND** 每個方案 SHALL 包含實作時間估計

#### Scenario: 每個步驟可驗證
- **GIVEN** 部署指南的任意步驟
- **WHEN** 用戶完成該步驟
- **THEN** SHALL 提供驗證命令
- **AND** SHALL 提供預期輸出範例
- **AND** 用戶 SHALL 能夠確認步驟正確完成

### Requirement: 部署驗證腳本
系統 SHALL 提供自動化驗證腳本，檢查雙實例部署的正確性。

#### Scenario: 部署驗證腳本執行成功
- **GIVEN** 雙實例部署已完成
- **WHEN** 執行 `./scripts/verify-dual-deployment.sh`
- **THEN** 腳本 SHALL 檢查以下項目：
  - 端口狀態（8000, 8001, 5672, 5673, 6379, 6380）
  - 環境變數配置（ENV=production, ENV=testing）
  - Docker 容器狀態
  - 資料庫連接
  - Redis/RabbitMQ 連接
  - Funboost worker 狀態
- **AND** 所有檢查通過時 SHALL 輸出 "All checks passed ✓"
- **AND** 退出碼 SHALL 為 0

#### Scenario: 部署驗證檢測到問題
- **GIVEN** Production 實例端口 8000 未開啟
- **WHEN** 執行 `./scripts/verify-dual-deployment.sh`
- **THEN** 腳本 SHALL 輸出錯誤訊息 "Port 8000 not accessible"
- **AND** 退出碼 SHALL 為非零值
- **AND** SHALL 提供故障排除建議

### Requirement: GPU 監控腳本
系統 SHALL 提供持續監控腳本，即時檢測 GPU 資源競爭。

#### Scenario: GPU 監控腳本正常運行
- **GIVEN** 雙實例部署正在運行
- **WHEN** 執行 `./scripts/monitor-gpu-usage.sh`
- **THEN** 腳本 SHALL 持續輸出 GPU 使用率資訊
- **AND** SHALL 顯示當前活躍任務數量
- **AND** SHALL 顯示分布式控頻狀態

#### Scenario: GPU 監控檢測到競爭
- **GIVEN** 分布式控頻未正確啟用
- **AND** 兩個推理任務同時執行
- **WHEN** 執行 `./scripts/monitor-gpu-usage.sh`
- **THEN** 腳本 SHALL 輸出警告 "⚠️ WARNING: Multiple concurrent tasks detected (2)"
- **AND** SHALL 提供診斷建議

### Requirement: 環境資料隔離
Production 和 Testing 環境 SHALL 使用不同的資料庫，確保資料完全隔離。

#### Scenario: Production 使用 production 資料庫
- **GIVEN** Production worker 接收推理任務
- **AND** 任務參數包含 `environment: "production"`
- **WHEN** Worker 處理任務
- **THEN** SHALL 使用資料庫 `dicom`
- **AND** SHALL 使用 Minio bucket `minio_backup`
- **AND** SHALL NOT 訪問 testing 資料庫

#### Scenario: Testing 使用 testing 資料庫
- **GIVEN** Testing worker 接收推理任務
- **AND** 任務參數包含 `environment: "testing"`
- **WHEN** Worker 處理任務
- **THEN** SHALL 使用資料庫 `dicom_testing`
- **AND** SHALL 使用 Minio bucket `minio_backup_testing`
- **AND** SHALL NOT 訪問 production 資料庫

#### Scenario: 資料庫隔離驗證
- **GIVEN** 雙實例部署已運行
- **WHEN** 同時發送 Production 和 Testing 推理請求
- **THEN** Production 資料 SHALL 只寫入 `dicom`
- **AND** Testing 資料 SHALL 只寫入 `dicom_testing`
- **AND** 無資料交叉污染

### Requirement: 獨立啟動/停止
Production 和 Testing 實例 SHALL 能夠獨立啟動和停止，互不影響。

#### Scenario: 獨立啟動 Production 實例
- **GIVEN** Testing 實例未運行
- **WHEN** 執行 `ENV=production ./brain-parcellation-start.sh`
- **THEN** Production 實例 SHALL 成功啟動
- **AND** 監聽端口 8000
- **AND** Funboost worker SHALL 正常運行
- **AND** 系統 SHALL 正常處理推理請求

#### Scenario: 獨立啟動 Testing 實例
- **GIVEN** Production 實例未運行
- **WHEN** 執行 `ENV=testing ./brain-parcellation-start.sh`
- **THEN** Testing 實例 SHALL 成功啟動
- **AND** 監聽端口 8001
- **AND** Funboost worker SHALL 正常運行
- **AND** 系統 SHALL 正常處理推理請求

#### Scenario: 停止 Production 不影響 Testing
- **GIVEN** Production 和 Testing 實例都在運行
- **WHEN** 停止 Production 實例
- **THEN** Testing 實例 SHALL 繼續運行
- **AND** Testing 端口 8001 SHALL 仍可訪問
- **AND** Testing worker SHALL 繼續處理任務

#### Scenario: 停止 Testing 不影響 Production
- **GIVEN** Production 和 Testing 實例都在運行
- **WHEN** 停止 Testing 實例
- **THEN** Production 實例 SHALL 繼續運行
- **AND** Production 端口 8000 SHALL 仍可訪問
- **AND** Production worker SHALL 繼續處理任務

### Requirement: 向後相容性
系統 SHALL 保持向後相容，未啟用分布式控頻時行為不變。

#### Scenario: 未啟用分布式控頻時預設行為
- **GIVEN** `is_using_distributed_frequency_control = False` 或未設定
- **WHEN** 啟動單個 worker (qps=1)
- **THEN** Worker SHALL 以 1 任務/秒執行
- **AND** 行為 SHALL 與舊版本完全一致
- **AND** 無功能破壞

#### Scenario: 單實例部署不受影響
- **GIVEN** 只啟動 Production 實例（無 Testing 實例）
- **WHEN** 使用分布式控頻 (`is_using_distributed_frequency_control=True`)
- **THEN** Redis 追蹤 `active_consumer_num = 1`
- **AND** Worker 實際 QPS SHALL 為 1.0 (1 / 1)
- **AND** 系統行為 SHALL 與單實例預期一致

### Requirement: 20 分鐘部署目標
新用戶遵循部署指南 SHALL 能在 20 分鐘內完成雙實例部署，包含 GPU 資源保護。

#### Scenario: 新用戶完整部署流程
- **GIVEN** 全新環境，未安裝任何組件
- **WHEN** 用戶按照 `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md` 執行
- **THEN** 從開始到完成 SHALL 不超過 20 分鐘
- **AND** 所有驗證步驟 SHALL 通過
- **AND** GPU 監控 SHALL 顯示無競爭

## MODIFIED Requirements

### Requirement: 舊文檔歸檔與棄用通知
原有的 `QUICK_START_DUAL.md` 和 `GPU_SOLUTION_COMPLETE.md` SHALL 移至歸檔目錄，並包含指向新文檔的連結。

#### Scenario: 舊文檔已歸檔
- **GIVEN** 專案根目錄
- **WHEN** 檢查檔案系統
- **THEN** SHALL 存在 `docs/archive/QUICK_START_DUAL.md`
- **AND** SHALL 存在 `docs/archive/GPU_SOLUTION_COMPLETE.md`
- **AND** 專案根目錄 SHALL NOT 包含這兩個檔案

#### Scenario: 歸檔文檔包含棄用通知
- **GIVEN** 歸檔文檔 `docs/archive/QUICK_START_DUAL.md`
- **WHEN** 開啟檔案
- **THEN** 檔案開頭 SHALL 包含棄用通知
- **AND** 通知 SHALL 指向新文檔 `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md`
- **AND** 通知 SHALL 說明為何棄用（文檔整合）

### Requirement: 專案文檔連結更新
`README.md` 和其他相關文檔 SHALL 更新連結，指向新的統一部署指南。

#### Scenario: README 連結已更新
- **GIVEN** 專案 `README.md`
- **WHEN** 檢查部署相關章節
- **THEN** SHALL 包含連結到 `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md`
- **AND** SHALL NOT 包含連結到 `QUICK_START_DUAL.md` 或 `GPU_SOLUTION_COMPLETE.md`
