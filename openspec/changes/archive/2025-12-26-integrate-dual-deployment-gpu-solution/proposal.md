# Proposal: Integrate Dual Deployment with GPU Resource Management

## Change ID
`integrate-dual-deployment-gpu-solution`

## Problem Statement

當前專案存在兩個獨立但相關的文檔：
1. **QUICK_START_DUAL.md** - 描述如何同時運行 Production 和 Testing 實例
2. **GPU_SOLUTION_COMPLETE.md** - 解決 GPU 資源競爭問題

這兩個方案解決同一個部署場景的不同層面問題，但彼此分離導致：
- **認知負擔**：用戶需要閱讀兩份文檔才能完整理解雙實例部署
- **實作風險**：用戶可能只實作 QUICK_START 而忽略 GPU 競爭問題
- **維護成本**：兩份文檔可能出現不一致或重複內容

**核心矛盾**：
- QUICK_START_DUAL.md 使用**不同隊列**策略（環境完全隔離）
- GPU_SOLUTION_COMPLETE.md 方案 B 建議**統一隊列**（分布式控頻）

## Proposed Solution

### Linus Torvalds 原則：簡潔務實
- **Talk is cheap, show me the code**: 提供可驗證的實作，而非理論說明
- **Do one thing well**: 每個文件解決單一明確的問題
- **Bad programmers worry about the code. Good programmers worry about data structures**: 聚焦於配置結構（.env, params.py）而非複雜邏輯

### Donald Knuth 原則：精確嚴謹
- **Premature optimization is the root of all evil**: 提供最簡方案優先，複雜方案作為選項
- **Programs are meant to be read by humans**: 文檔結構清晰，決策樹明確
- **Beware of bugs in the above code; I have only proved it correct, not tried it**: 每個方案提供驗證步驟

### 整合策略

創建單一統一文檔 **`DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md`**，結構如下：

```markdown
1. 前置條件檢查（5 分鐘）
   - 系統需求
   - 依賴驗證

2. 快速開始（15 分鐘）- 方案 B（推薦）
   - 啟用分布式控頻
   - 統一隊列配置
   - 一鍵啟動腳本
   - 驗證步驟

3. 進階選項（需要時）
   - 方案 A：不同隊列 + Redis GPU 鎖
   - 方案 C：優先級控制

4. 監控和故障排除
   - GPU 使用率監控
   - 常見問題處理

5. 附錄
   - 端口分配表
   - 配置參考
```

### 決策樹

```
用戶需求：同時運行 Production 和 Testing
    ↓
是否需要嚴格環境隔離（不同隊列）？
    │
    ├─ No（90% 場景）→ 方案 B：分布式控頻
    │  └─ 實作時間：5 分鐘
    │     優點：極簡配置
    │     驗證：監控 GPU 同時只有 1 任務
    │
    └─ Yes（特殊需求）→ 方案 A：Redis GPU 鎖
       └─ 實作時間：30 分鐘
          優點：環境完全隔離
          驗證：不同隊列 + GPU 鎖追蹤
```

## Scope

### In Scope
1. **整合文檔內容**
   - 合併 QUICK_START_DUAL.md 和 GPU_SOLUTION_COMPLETE.md
   - 創建決策樹幫助用戶選擇方案
   - 提供驗證清單

2. **優化配置結構**
   - 創建 `.env.dual-deployment.example` 統一配置範本
   - 修改 `code_ai/task/params.py` 提供分布式控頻開關
   - 提供 GPU 鎖模組（可選）

3. **驗證腳本**
   - `scripts/verify-dual-deployment.sh` - 檢查端口、配置、GPU
   - `scripts/monitor-gpu-usage.sh` - 持續監控 GPU 競爭

### Out of Scope
- 自動化部署流程（CI/CD）
- Kubernetes/多機部署
- GPU 資源動態調度（超出當前需求）
- 前端UI控制面板

## Architecture Impact

### 配置層級
```
當前：
├── .env.production（端口 8000）
├── .env.testing（端口 8001）
├── code_ai/task/params.py（qps=1, SOLO mode）
└── funboost_config.py（Redis配置）

建議（方案 B）：
├── .env.production（端口 8000）
├── .env.testing（端口 8001）
├── code_ai/task/params.py（+ is_using_distributed_frequency_control=True）
├── code_ai/task/task_pipeline.py（統一隊列，環境標籤）
└── funboost_config.py（不變）

選項（方案 A）：
├── .env.production
├── .env.testing
├── code_ai/task/params.py（不變）
├── code_ai/task/task_pipeline.py（環境感知隊列）
├── code_ai/utils/gpu_lock.py（新增）
└── funboost_config.py（不變）
```

### 資料流

**方案 B（推薦）：**
```
API Request → 發布任務（+ environment tag）
              ↓
統一隊列: task_pipeline_inference_queue
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
Prod Worker         Test Worker
(qps/2=0.5)        (qps/2=0.5)
    └─────────┬─────────┘
              ↓
     GPU（最多 1 任務/秒）
```

**方案 A（選項）：**
```
API Request → 發布任務
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
Prod Queue          Test Queue
    ↓                   ↓
Prod Worker         Test Worker
    └──→ GPU Lock ←──┘
              ↓
     GPU（鎖控制訪問）
```

## Implementation Plan

### Phase 1: 文檔整合（優先）
1. 創建 `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md`
2. 遷移 QUICK_START 快速啟動內容
3. 整合 GPU 方案選擇決策樹
4. 添加驗證清單

### Phase 2: 配置優化（推薦方案）
1. 修改 `code_ai/task/params.py` 啟用分布式控頻
2. 修改 `code_ai/task/task_pipeline.py` 支持環境標籤
3. 創建 `.env.dual-deployment.example`

### Phase 3: 驗證工具
1. 創建 `scripts/verify-dual-deployment.sh`
2. 創建 `scripts/monitor-gpu-usage.sh`

### Phase 4: 清理（可選）
1. 歸檔 QUICK_START_DUAL.md, GPU_SOLUTION_COMPLETE.md 到 `docs/archive/`
2. 更新 README.md 指向新文檔

## Success Criteria

### 用戶體驗
- [ ] 新用戶可在 **20 分鐘內**完成雙實例部署（含 GPU 保護）
- [ ] 文檔提供明確的決策樹（< 3 個決策點）
- [ ] 每個步驟有驗證命令和預期輸出

### 技術驗證
- [ ] GPU 監控顯示同時最多 1 個推理任務執行
- [ ] Production 和 Testing 資料庫完全隔離
- [ ] 兩個環境可以獨立啟動/停止

### 文檔品質
- [ ] 單一真實來源（Single Source of Truth）
- [ ] 遵循 Linus 原則：提供可執行腳本，而非純文字說明
- [ ] 遵循 Knuth 原則：每個聲明有驗證方法

## Dependencies

### 技術依賴
- Redis（已配置於 funboost_config.py）
- RabbitMQ（已配置於 docker-compose.yml）
- Docker Compose
- 環境配置模組（backend/app/config/）

### 前置 Change
- `add-environment-support`（48/52 tasks completed）
  - 需要完成才能確保環境配置正確

## Risks and Mitigations

### 風險 1：用戶選錯方案
**風險**: 用戶可能選擇複雜方案（方案 A）而忽略簡單方案（方案 B）

**緩解**:
- 文檔明確標註 **"推薦"** 和 **"進階選項"**
- 提供決策樹，90% 場景指向方案 B
- 快速開始章節只展示方案 B

### 風險 2：配置不一致
**風險**: QUICK_START 使用不同隊列，GPU_SOLUTION 使用統一隊列

**緩解**:
- 整合文檔明確說明兩種架構差異
- 提供配置範例和驗證腳本
- 每個方案有獨立的驗證步驟

### 風險 3：GPU 競爭仍存在
**風險**: 用戶部署後仍然出現 GPU OOM

**緩解**:
- 提供 `scripts/monitor-gpu-usage.sh` 持續監控
- 驗證清單包含 GPU 監控步驟
- 故障排除章節包含 GPU 相關問題

## Open Questions

1. **是否保留舊文檔？**
   - 選項 A：歸檔到 `docs/archive/`
   - 選項 B：直接刪除
   - **建議**: 歸檔，保留歷史記錄

2. **方案 B 是否成為唯一推薦方案？**
   - 當前：提供 3 個方案
   - **建議**: 快速開始只展示方案 B，其他作為進階選項

3. **是否需要自動化部署腳本？**
   - 當前：手動執行 `./deploy-dual.sh`
   - **建議**: Phase 1 保持手動，Phase 2 可考慮增強腳本

## References

- Existing Documentation:
  - `QUICK_START_DUAL.md` - 雙實例快速啟動
  - `GPU_SOLUTION_COMPLETE.md` - GPU 資源管理
  - `DEPLOYMENT_GUIDE.md` - 完整部署指南
  - `openspec/changes/add-environment-support/` - 環境支援實作

- Related Changes:
  - `add-environment-support` - 提供環境配置基礎設施

- External References:
  - Funboost documentation: https://funboost.readthedocs.io/
  - Docker Compose: https://docs.docker.com/compose/
