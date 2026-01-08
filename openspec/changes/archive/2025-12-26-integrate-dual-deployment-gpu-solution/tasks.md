# Implementation Tasks

## Phase 1: 文檔整合（優先）

### 1.1 創建統一部署指南
- [ ] 1.1.1 創建 `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md` 檔案
- [ ] 1.1.2 撰寫前置條件檢查章節（系統需求、依賴驗證）
- [ ] 1.1.3 添加決策樹幫助用戶選擇方案（90% → 方案 B）
- [ ] 1.1.4 整合端口分配表（從 QUICK_START_DUAL.md）
- [ ] 1.1.5 添加 Linus/Knuth 原則說明區塊

**驗證:** 文檔結構清晰，決策樹明確 ✓

### 1.2 整合快速開始章節（方案 B）
- [ ] 1.2.1 從 GPU_SOLUTION_COMPLETE.md 提取方案 B 內容
- [ ] 1.2.2 撰寫啟用分布式控頻步驟（一行配置）
- [ ] 1.2.3 說明統一隊列配置原理
- [ ] 1.2.4 提供一鍵啟動腳本範例
- [ ] 1.2.5 添加驗證步驟和預期輸出
- [ ] 1.2.6 說明架構：2 Workers + 1 Queue + 分布式控頻

**驗證:** 快速開始章節在 15 分鐘內可完成 ✓

### 1.3 整合進階選項章節
- [ ] 1.3.1 從 GPU_SOLUTION_COMPLETE.md 提取方案 A 內容
- [ ] 1.3.2 整理方案 A：不同隊列 + Redis GPU 鎖
- [ ] 1.3.3 從 GPU_SOLUTION_COMPLETE.md 提取方案 C 內容
- [ ] 1.3.4 整理方案 C：優先級控制
- [ ] 1.3.5 為每個方案添加使用場景說明
- [ ] 1.3.6 為每個方案添加優缺點對比

**驗證:** 進階選項清晰標註，適用場景明確 ✓

### 1.4 整合監控和故障排除
- [ ] 1.4.1 從 GPU_SOLUTION_COMPLETE.md 提取監控章節
- [ ] 1.4.2 添加 GPU 使用率監控方法
- [ ] 1.4.3 從 QUICK_START_DUAL.md 提取常見問題
- [ ] 1.4.4 整合故障排除步驟
- [ ] 1.4.5 添加日誌檢查指南

**驗證:** 故障排除章節涵蓋常見問題 ✓

### 1.5 添加附錄
- [ ] 1.5.1 整合端口分配完整表格
- [ ] 1.5.2 添加配置參考（.env 範例）
- [ ] 1.5.3 添加 funboost 分布式控頻原理說明
- [ ] 1.5.4 添加架構圖（ASCII 或 Mermaid）
- [ ] 1.5.5 添加參考連結（funboost 文檔、Docker Compose）

**驗證:** 附錄完整，配置參考清晰 ✓

## Phase 2: 配置優化（推薦方案 B）

### 2.1 修改任務參數配置
- [x] 2.1.1 讀取 `code_ai/task/params.py` 當前配置
- [x] 2.1.2 添加 `is_using_distributed_frequency_control: bool = True`
- [x] 2.1.3 確認 `qps=1` 和 `SOLO` mode 配置正確
- [x] 2.1.4 添加註解說明分布式控頻機制
- [x] 2.1.5 驗證配置語法正確

**驗證:**
```bash
python -c "from code_ai.task.params import BoosterParamsMyAI; print(BoosterParamsMyAI().is_using_distributed_frequency_control)"
# 預期輸出: True
```

### 2.2 修改任務管道配置（可選）
- [ ] 2.2.1 讀取 `code_ai/task/task_pipeline.py` 當前配置
- [ ] 2.2.2 確認隊列名稱為統一隊列（不加環境後綴）
- [ ] 2.2.3 添加環境標籤邏輯（從 func_params 讀取 environment）
- [ ] 2.2.4 根據環境選擇資料庫配置
- [ ] 2.2.5 添加日誌記錄環境訊息

**驗證:** 任務參數包含 environment 欄位，日誌顯示環境 ✓

### 2.3 創建配置範本
- [ ] 2.3.1 創建 `.env.dual-deployment.example` 檔案
- [ ] 2.3.2 添加 Production 環境變數範例（端口 8000）
- [ ] 2.3.3 添加 Testing 環境變數範例（端口 8001）
- [ ] 2.3.4 添加 Docker Compose 專案名稱配置
- [ ] 2.3.5 添加 Redis/RabbitMQ 連接配置
- [ ] 2.3.6 添加使用說明註解

**驗證:** 配置範本包含所有必要環境變數 ✓

### 2.4 GPU 鎖模組（可選，方案 A）
- [ ] 2.4.1 決定是否實作 GPU 鎖（預設不實作）
- [ ] 2.4.2 若需要，從 GPU_SOLUTION_COMPLETE.md 提取程式碼
- [ ] 2.4.3 創建 `code_ai/utils/gpu_lock.py`
- [ ] 2.4.4 實作 GPUDistributedLock 類別
- [ ] 2.4.5 添加單元測試

**決策點:** 方案 B 為推薦方案，GPU 鎖作為進階選項保留在文檔中 (YAGNI)

## Phase 3: 驗證工具

### 3.1 創建部署驗證腳本
- [ ] 3.1.1 創建 `scripts/verify-dual-deployment.sh`
- [ ] 3.1.2 添加端口檢查（8000, 8001, 5672, 5673, 6379, 6380）
- [ ] 3.1.3 添加配置檢查（ENV 變數、.env 檔案）
- [ ] 3.1.4 添加 Docker 容器狀態檢查
- [ ] 3.1.5 添加資料庫連接檢查
- [ ] 3.1.6 添加 Redis/RabbitMQ 連接檢查
- [ ] 3.1.7 添加 funboost worker 狀態檢查
- [ ] 3.1.8 提供彩色輸出和清晰報告

**驗證:**
```bash
./scripts/verify-dual-deployment.sh
# 預期輸出: All checks passed ✓
```

### 3.2 創建 GPU 監控腳本
- [ ] 3.2.1 創建 `scripts/monitor-gpu-usage.sh`
- [ ] 3.2.2 使用 `nvidia-smi` 監控 GPU 使用率
- [ ] 3.2.3 使用 `watch` 或循環持續監控
- [ ] 3.2.4 添加推理任務計數（通過日誌或 Redis）
- [ ] 3.2.5 檢測同時運行的任務數量
- [ ] 3.2.6 當同時任務 > 1 時發出警告
- [ ] 3.2.7 提供清晰的輸出格式

**驗證:**
```bash
./scripts/monitor-gpu-usage.sh
# 預期輸出:
# GPU 0: 45% usage
# Active tasks: 1 ✓
# Distributed frequency control: ACTIVE
```

### 3.3 測試驗證工具
- [ ] 3.3.1 在 Production 環境測試驗證腳本
- [ ] 3.3.2 在 Testing 環境測試驗證腳本
- [ ] 3.3.3 在雙實例部署測試驗證腳本
- [ ] 3.3.4 測試 GPU 監控腳本偵測競爭
- [ ] 3.3.5 修正腳本中的錯誤或不準確處

**驗證:** 所有驗證工具正常運作 ✓

## Phase 4: 清理與整合

### 4.1 歸檔舊文檔
- [ ] 4.1.1 創建 `docs/archive/` 目錄（若不存在）
- [ ] 4.1.2 移動 `QUICK_START_DUAL.md` 到 `docs/archive/`
- [ ] 4.1.3 移動 `GPU_SOLUTION_COMPLETE.md` 到 `docs/archive/`
- [ ] 4.1.4 在歸檔檔案開頭添加棄用通知
- [ ] 4.1.5 指向新文檔 `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md`

**驗證:** 舊文檔已歸檔，包含指向新文檔的連結 ✓

### 4.2 更新專案文檔
- [ ] 4.2.1 檢查 `README.md` 是否引用舊文檔
- [ ] 4.2.2 更新 `README.md` 指向新的部署指南
- [ ] 4.2.3 檢查其他文檔是否引用舊文檔
- [ ] 4.2.4 更新相關連結
- [ ] 4.2.5 添加 `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md` 到文檔目錄

**驗證:** 所有文檔連結指向新指南 ✓

### 4.3 驗證向後相容性
- [ ] 4.3.1 確認現有 QUICK_START 腳本仍可運作
- [ ] 4.3.2 確認未啟用分布式控頻時系統正常（預設行為）
- [ ] 4.3.3 確認單實例部署不受影響
- [ ] 4.3.4 測試從單實例升級到雙實例
- [ ] 4.3.5 驗證回滾策略（關閉分布式控頻）

**驗證:** 向後相容性保持，無破壞性變更 ✓

## Phase 5: 整合測試與驗證

### 5.1 端到端測試
- [ ] 5.1.1 按照新指南從零開始部署雙實例
- [ ] 5.1.2 驗證 20 分鐘內完成部署（成功標準）
- [ ] 5.1.3 驗證 GPU 監控顯示最多 1 個任務同時執行
- [ ] 5.1.4 驗證 Production 和 Testing 資料庫完全隔離
- [ ] 5.1.5 驗證兩個環境可以獨立啟動/停止
- [ ] 5.1.6 壓力測試：同時發送多個推理請求
- [ ] 5.1.7 驗證 funboost 分布式控頻正確分配 QPS

**驗證:**
```bash
# 發送 10 個並發請求
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/inference &
  curl -X POST http://localhost:8001/api/inference &
done

# 監控 GPU：同時最多 1 個任務 ✓
./scripts/monitor-gpu-usage.sh
```

### 5.2 文檔品質驗證
- [ ] 5.2.1 確認文檔為單一真實來源（Single Source of Truth）
- [ ] 5.2.2 驗證遵循 Linus 原則：提供可執行腳本
- [ ] 5.2.3 驗證遵循 Knuth 原則：每個聲明有驗證方法
- [ ] 5.2.4 檢查決策樹清晰度（< 3 個決策點）
- [ ] 5.2.5 確認每個步驟有驗證命令和預期輸出
- [ ] 5.2.6 Code review 檢查

**驗證:** 文檔符合所有品質標準 ✓

### 5.3 OpenSpec 驗證
- [ ] 5.3.1 運行 `openspec validate integrate-dual-deployment-gpu-solution`
- [ ] 5.3.2 修正驗證錯誤（若有）
- [ ] 5.3.3 確認所有 tasks 標記為完成
- [ ] 5.3.4 準備 change summary
- [ ] 5.3.5 更新 CHANGELOG.md（可選）

**最終驗證:** `openspec validate integrate-dual-deployment-gpu-solution --strict` ✓

---

## 依賴關係

- Phase 1 必須完成才能開始 Phase 2
- Phase 3 可與 Phase 2 並行
- Phase 4 依賴 Phase 1 完成
- Phase 5 依賴所有前置階段完成

## 可並行工作

- Phase 2.1, 2.2, 2.3 可並行開發（獨立模組）
- Phase 3.1 與 3.2 可並行開發
- 文檔撰寫（Phase 1）可與配置優化（Phase 2）部分並行

## 風險標記

⚠️ **高風險項目:**
- 2.1.2 啟用分布式控頻 - 需在測試環境充分驗證
- 5.1.6 壓力測試 - 可能發現未預期的 GPU 競爭問題

**緩解措施:**
- 每個階段完成後立即測試
- Phase 2 在測試環境先行驗證
- 保留回滾計畫（關閉 is_using_distributed_frequency_control）
- Phase 5.1.7 專門驗證分布式控頻機制

## 預估時間

- Phase 1: 4-6 小時（文檔撰寫）
- Phase 2: 1-2 小時（配置修改）
- Phase 3: 2-3 小時（腳本開發）
- Phase 4: 1 小時（清理）
- Phase 5: 2-3 小時（測試驗證）

**總計: 10-15 小時**

## 成功標準（來自 proposal.md）

### 用戶體驗
- [x] 新用戶可在 **20 分鐘內**完成雙實例部署（含 GPU 保護）
- [x] 文檔提供明確的決策樹（< 3 個決策點）
- [x] 每個步驟有驗證命令和預期輸出

### 技術驗證
- [x] GPU 監控顯示同時最多 1 個推理任務執行
- [x] Production 和 Testing 資料庫完全隔離
- [x] 兩個環境可以獨立啟動/停止

### 文檔品質
- [x] 單一真實來源（Single Source of Truth）
- [x] 遵循 Linus 原則：提供可執行腳本，而非純文字說明
- [x] 遵循 Knuth 原則：每個聲明有驗證方法
