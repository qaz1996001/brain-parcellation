# OpenSpec 整合指南（OpenSpec Integration Guide）

**文件 ID**: GUIDE-OPENSPEC-001  
**版本**: v1.0.0  
**建立日期**: 2025-12-22  

---

## 1. OpenSpec 與元框架的關係（OpenSpec and MetaFramework）

### 1.1 核心概念

| 系統 | 職責 | 內容 |
|------|------|------|
| **元框架（MetaFramework）** | 描述「系統應該是什麼」 | PRD/SR/SD 文檔（當前狀態的完整描述） |
| **OpenSpec** | 管理「如何改變系統」 | 變更提案（增量變更的規格） |

**關鍵點**：
- **PRD/SR/SD** 是「真相」（Ground Truth）：描述系統當前應有的完整行為
- **OpenSpec Changes** 是「變更提案」：描述如何從當前狀態變到目標狀態
- **同步機制**：每次歸檔 OpenSpec 變更時，必須同步更新 PRD/SR/SD

### 1.2 工作流程

```
[現有系統狀態]
    ↓ (描述於)
[PRD/SR/SD 文檔]
    ↓ (發現需要變更)
[創建 OpenSpec 提案]
    ↓ (審核通過)
[實作變更]
    ↓ (歸檔時)
[更新 PRD/SR/SD] ← 同步變更內容
    ↓
[新的系統狀態]
```

---

## 2. 何時創建 OpenSpec 提案？（When to Create Proposals）

### 2.1 需要提案的情況

✅ **必須創建提案**：
- 新增功能或能力（Capability）
- 修改現有功能行為（Behavioral Change）
- 架構性變更（Architecture Change）
- API 或資料庫 Schema 的 Breaking Change
- 效能優化（影響現有行為）
- 安全性強化（影響現有行為）

### 2.2 不需要提案的情況

❌ **可直接修改**：
- Bug 修復（恢復既有規格的行為）
- 錯字、格式修正
- 註解、文檔澄清（不改變規格）
- 依賴版本更新（非 Breaking Change）
- 配置調整（不影響功能）
- 單元測試補充（測試既有行為）

### 2.3 灰色地帶（建議創建提案）

🟡 **建議創建提案，保險起見**：
- 重構（可能影響多個模組）
- 效能優化（可能改變資源使用模式）
- UI/UX 調整（可能影響使用者操作流程）
- 第三方服務整合（新增外部依賴）

---

## 3. OpenSpec 提案結構（Proposal Structure）

### 3.1 目錄結構

```
openspec/
├── changes/
│   └── add-user-notification/          # change-id（kebab-case, verb-led）
│       ├── proposal.md                  # 提案摘要（Why, What, Impact）
│       ├── tasks.md                     # 實作檢查清單
│       ├── design.md                    # 技術決策（可選）
│       └── specs/                       # 需求變更差異
│           └── notifications/           # capability 名稱
│               └── spec.md              # ADDED/MODIFIED/REMOVED Requirements
```

### 3.2 proposal.md 範本

```markdown
# Change: [簡短描述變更]

## Why
[1-2 句話說明為什麼需要這個變更]
- 解決什麼問題？
- 帶來什麼價值？
- 對應哪個使用者需求（UR-xxx）？

## What Changes
- [變更點 1]
- [變更點 2]（標記 **BREAKING** 若為破壞性變更）
- [變更點 3]

## Impact
- **Affected specs**: [列出影響的 capabilities]
- **Affected code**: [列出關鍵檔案/模組]
- **Affected docs**: [列出需更新的文檔]
- **Migration required**: [是否需要遷移？如何遷移？]

## Risks
- [風險 1]：[緩解措施]
- [風險 2]：[緩解措施]

## Dependencies
- Depends on: [其他 OpenSpec 變更或外部因素]
```

### 3.3 tasks.md 範本

```markdown
# Implementation Tasks

## 1. Database / Schema
- [ ] 1.1 設計資料表 schema
- [ ] 1.2 撰寫 migration 腳本
- [ ] 1.3 驗證 migration（開發環境）

## 2. Backend / API
- [ ] 2.1 實作 API 端點
- [ ] 2.2 實作業務邏輯
- [ ] 2.3 撰寫單元測試
- [ ] 2.4 撰寫整合測試

## 3. Frontend / UI
- [ ] 3.1 設計 UI 元件
- [ ] 3.2 實作元件邏輯
- [ ] 3.3 整合 API
- [ ] 3.4 撰寫元件測試

## 4. Documentation
- [ ] 4.1 更新 API 文檔
- [ ] 4.2 更新使用者手冊
- [ ] 4.3 更新 PRD/SR/SD（於歸檔時）

## 5. Testing
- [ ] 5.1 手動測試（開發環境）
- [ ] 5.2 整合測試
- [ ] 5.3 驗收測試（UAT）

## 6. Deployment
- [ ] 6.1 準備部署腳本
- [ ] 6.2 通知相關人員（若有 Breaking Change）
- [ ] 6.3 執行部署
- [ ] 6.4 驗證生產環境
```

### 3.4 design.md 範本（可選）

**何時需要 design.md？**
- 跨系統變更（影響多個子系統）
- 新增外部依賴或整合
- 架構性變更
- 效能、安全、或遷移複雜度高
- 有技術選型需要討論

```markdown
# Design: [變更名稱]

## Context
**背景**：[為什麼需要這個變更？]  
**約束**：[技術限制、時程限制、資源限制]  
**利害關係人**：[誰關心這個變更？]

## Goals / Non-Goals
**Goals**（目標）：
- [目標 1]
- [目標 2]

**Non-Goals**（非目標）：
- [明確不做的事情 1]
- [明確不做的事情 2]

## Decisions
### 決策 1: [技術選擇]
**選項**：
- 選項 A：[優點 / 缺點]
- 選項 B：[優點 / 缺點]

**決定**：選擇 [選項 X]  
**理由**：[為什麼選這個？]

## Architecture / Design
[插入架構圖、流程圖、ER 圖等]

**關鍵元件**：
- [元件 1]：[職責]
- [元件 2]：[職責]

## Data Model
[資料表設計、欄位說明]

## API Design
[端點、請求/回應格式]

## Risks / Trade-offs
- **風險 1**：[說明] → 緩解措施：[措施]
- **權衡 1**：[取捨說明]

## Migration Plan
**現有資料**：[如何遷移？]  
**回滾計畫**：[如何回滾？]

## Open Questions
- [待解決問題 1]
- [待解決問題 2]
```

### 3.5 specs/[capability]/spec.md 範本

```markdown
## ADDED Requirements
### Requirement: [新需求名稱]
系統 SHALL [需求描述]。

#### Scenario: [情境名稱]
- **WHEN** [觸發條件]
- **THEN** [預期結果]

---

## MODIFIED Requirements
### Requirement: [既有需求名稱]
[完整的修改後需求內容，包含所有 Scenarios]

**變更說明**：[為什麼修改？改了什麼？]

---

## REMOVED Requirements
### Requirement: [被移除的需求名稱]
**原內容**：[簡述原需求]  
**移除原因**：[為什麼移除？]  
**遷移指引**：[使用者應如何應對？]

---

## RENAMED Requirements
- FROM: `### Requirement: Old Name`
- TO: `### Requirement: New Name`
```

**關鍵注意事項**：
- **MODIFIED Requirements**：必須包含完整的修改後內容（不是只寫改動的部分）
- **Scenario 格式**：必須使用 `#### Scenario:` 格式（4 個井號）
- **每個 Requirement 至少一個 Scenario**

---

## 4. OpenSpec 工作流程（Workflow）

### 4.1 Stage 1: 創建提案（Create Proposal）

**人類職責**：
1. 提出需求變更（基於 UR-xxx 或業務需求）
2. 審核 AI 產出的提案（proposal.md, tasks.md, design.md, spec deltas）
3. 決定批准或拒絕提案

**AI 職責**：
1. 閱讀 PRD/SR/SD 文檔，理解現有系統狀態
2. 創建 OpenSpec 提案結構（`openspec/changes/<change-id>/`）
3. 撰寫 proposal.md（Why, What, Impact）
4. 撰寫 tasks.md（可執行的檢查清單）
5. 撰寫 design.md（若需要）
6. 產生 spec deltas（ADDED/MODIFIED/REMOVED Requirements）
7. 執行驗證：`openspec validate <change-id> --strict`

**範例**（AI 創建提案的步驟）：

```bash
# 1. 閱讀現有需求
AI 閱讀：docs/requirements/01_SYSTEM_PRD_SR_SD.md

# 2. 選擇 change-id
CHANGE_ID="add-user-notification"

# 3. 創建目錄結構
mkdir -p openspec/changes/$CHANGE_ID/specs/notifications

# 4. 撰寫 proposal.md
cat > openspec/changes/$CHANGE_ID/proposal.md << 'EOF'
# Change: 新增使用者通知功能

## Why
當報告分析完成或專案有更新時，使用者需要即時收到通知，以便及時查看結果。
對應需求：UR-007（使用者希望收到即時通知）

## What Changes
- 新增通知功能模組（Email, Push, SMS）
- 新增通知管理 API
- 新增通知歷史記錄

## Impact
- Affected specs: notifications（新增）
- Affected code: backend/services/notification_service.py（新建）
- Affected docs: API_SPECIFICATION.md, DATABASE_DESIGN.md

## Risks
- Email 服務商限流：使用 Celery 任務佇列，避免瞬間大量發送
EOF

# 5. 撰寫 tasks.md
cat > openspec/changes/$CHANGE_ID/tasks.md << 'EOF'
## 1. Database
- [ ] 1.1 設計 notifications 資料表
- [ ] 1.2 撰寫 migration

## 2. Backend
- [ ] 2.1 實作 notification_service.py
- [ ] 2.2 實作 API 端點
- [ ] 2.3 撰寫測試

## 3. Documentation
- [ ] 3.1 更新 API 文檔
EOF

# 6. 撰寫 spec delta
cat > openspec/changes/$CHANGE_ID/specs/notifications/spec.md << 'EOF'
## ADDED Requirements
### Requirement: Push Notification
系統 SHALL 提供推播通知功能，支援 Email, Push, SMS 管道。

#### Scenario: 分析完成通知
- **WHEN** 報告 AI 分析完成
- **THEN** 系統發送 Email 通知給使用者
- **AND** 通知包含報告名稱與分析結果摘要
EOF

# 7. 驗證提案
openspec validate $CHANGE_ID --strict
```

### 4.2 Stage 2: 實作變更（Implement Change）

**人類職責**：
1. 驗收功能（根據 Scenario 測試）
2. 進行整合測試
3. 決定是否部署

**AI 職責**：
1. 依據 tasks.md 逐項完成開發
2. 撰寫測試（單元測試、整合測試）
3. 更新技術文檔（API、資料庫 schema）
4. 確保所有測試通過
5. 標記 tasks.md 中的項目為 `- [x]`

**實作檢查清單**：
- [ ] 所有 tasks.md 項目完成並標記 `[x]`
- [ ] 單元測試覆蓋率達標（> 70%）
- [ ] 整合測試通過
- [ ] 符合 spec deltas 中的所有 Scenarios
- [ ] 技術文檔已更新

### 4.3 Stage 3: 歸檔變更（Archive Change）

**何時歸檔？**
- 所有 tasks.md 項目完成
- 通過驗收測試
- 已部署到生產環境（或準備部署）

**人類職責**：
1. 最終審核與部署決策
2. 驗證 PRD/SR/SD 文檔已同步更新

**AI 職責**：
1. 執行 `openspec archive <change-id> --yes`
2. 將 spec deltas 同步到 `openspec/specs/<capability>/spec.md`
3. **關鍵**：將 spec deltas 同步到 PRD/SR/SD 文檔
4. 更新追溯矩陣
5. 更新需求狀態（Draft → Approved → Verified）

**歸檔步驟（AI 執行）**：

```bash
CHANGE_ID="add-user-notification"

# 1. 歸檔 OpenSpec 變更
openspec archive $CHANGE_ID --yes

# 2. 提取 spec deltas
# 從 openspec/changes/archive/2025-12-22-add-user-notification/specs/notifications/spec.md
# 提取 ADDED/MODIFIED/REMOVED 需求

# 3. 同步到 PRD/SR/SD
# 將新需求添加到 docs/requirements/03_BACKEND_PRD_SR_SD.md

cat >> docs/requirements/03_BACKEND_PRD_SR_SD.md << 'EOF'

#### BE-SR-080: 推播通知功能
**描述**：後端 SHALL 提供推播通知功能，支援 Email, Push, SMS 管道。

**來源 OpenSpec**：`changes/archive/2025-12-22-add-user-notification`  
**來源 SYS-SR**：SYS-SR-025  

**驗收情境**：

##### Scenario: 分析完成通知
- **WHEN** 報告 AI 分析完成
- **THEN** 系統發送 Email 通知給使用者
- **AND** 通知包含報告名稱與分析結果摘要

**追溯關係**：
- Traces to: SYS-SR-025, UR-007
- Traced by: API-015, DB-008
- Verified by: TC-NOTIF-001

EOF

# 4. 更新追溯矩陣
# 在 TRACEABILITY_MATRIX.md 中添加 BE-SR-080 的追溯關係

# 5. 驗證
openspec validate --strict
```

---

## 5. PRD/SR/SD 與 OpenSpec 的同步機制（Synchronization）

### 5.1 同步原則

**黃金法則**：歸檔 OpenSpec 變更時，必須同步更新 PRD/SR/SD。

**為什麼重要？**
- PRD/SR/SD 是「單一真相來源」（Single Source of Truth）
- 開發人員依據 PRD/SR/SD 理解系統行為
- 稽核時依據 PRD/SR/SD 檢查合規性

### 5.2 同步步驟

**步驟 1：提取 spec deltas**

從 `openspec/changes/archive/<date>-<change-id>/specs/<capability>/spec.md` 提取：
- ADDED Requirements → 新增到 PRD/SR/SD
- MODIFIED Requirements → 更新 PRD/SR/SD 中的對應需求
- REMOVED Requirements → 標記為 Deprecated 或刪除

**步驟 2：選擇正確的 PRD/SR/SD 文檔**

根據 capability 類型選擇：
- 系統層能力 → `01_SYSTEM_PRD_SR_SD.md`
- 前端能力 → `02_FRONTEND_PRD_SR_SD.md`
- 後端能力 → `03_BACKEND_PRD_SR_SD.md`

**步驟 3：插入新需求**

```markdown
#### BE-SR-080: [需求名稱]
**描述**：[需求內容]

**來源 OpenSpec**：`changes/archive/2025-12-22-add-user-notification`  
**來源 SYS-SR**：SYS-SR-xxx  
**優先級**：Must / Should  
**狀態**：Verified  

**驗收情境**：
[複製 spec delta 中的 Scenarios]

**追溯關係**：
- Traces to: SYS-SR-xxx, UR-xxx
- Traced by: API-xxx, DB-xxx
- Verified by: TC-xxx
```

**步驟 4：更新追溯矩陣**

在 `TRACEABILITY_MATRIX.md` 中添加新需求的追溯關係。

**步驟 5：更新需求狀態**

- Draft → Proposed → Approved → In Progress → Implemented → **Verified**

---

## 6. 常見問題（FAQ）

### Q1: 每次小改都要創建 OpenSpec 提案嗎？

**A**: 不用。依據「何時創建提案」規則：
- Bug 修復：直接修改
- 新功能：創建提案
- 重構（小範圍）：視情況，建議創建提案保險

### Q2: 如果提案被拒絕怎麼辦？

**A**: 
1. 刪除或保留提案（移到 `changes/rejected/`）
2. 記錄拒絕原因（在 proposal.md 頂部）
3. 不實作、不歸檔、不更新 PRD/SR/SD

### Q3: 歸檔時發現 PRD/SR/SD 過時怎麼辦？

**A**: 
1. 先補齊 PRD/SR/SD（逆向工程）
2. 再歸檔當前變更
3. 建議定期審計文檔與程式碼的一致性

### Q4: 如何處理跨多個 capability 的變更？

**A**: 
```
openspec/changes/add-complex-feature/
└── specs/
    ├── auth/
    │   └── spec.md       # 影響 auth capability
    ├── notifications/
    │   └── spec.md       # 影響 notifications capability
    └── data-import/
        └── spec.md       # 影響 data-import capability
```

歸檔時需同步更新所有相關的 PRD/SR/SD 章節。

---

## 7. 自動化建議（Automation Recommendations）

### 7.1 驗證腳本

**檢查 OpenSpec 提案格式**：
```bash
openspec validate <change-id> --strict
```

**檢查 PRD/SR/SD 與 OpenSpec 的一致性**：
```python
# scripts/validate_sync.py
# 檢查所有 archived changes 是否都已同步到 PRD/SR/SD
```

### 7.2 CI/CD 整合

在 Pull Request 時自動檢查：
```yaml
# .github/workflows/docs-validation.yml
- name: Validate OpenSpec
  run: openspec validate --strict

- name: Check docs sync
  run: python scripts/validate_sync.py
```

---

## 8. 下一步（Next Steps）

### 學習資源

- 閱讀 OpenSpec 官方文檔（`openspec/AGENTS.md`）
- 查看歷史歸檔範例（`openspec/changes/archive/`）
- 實作第一個提案（練習用）

### 實戰練習

1. 創建一個簡單的提案（如「新增使用者個人設定」）
2. 撰寫 proposal.md, tasks.md, spec delta
3. 執行驗證：`openspec validate <change-id> --strict`
4. （模擬）實作後歸檔，同步到 PRD/SR/SD

---

**文檔版本**: v1.0.0  
**維護團隊**: MetaFramework Core Team  
**最後審核**: 2025-12-22

