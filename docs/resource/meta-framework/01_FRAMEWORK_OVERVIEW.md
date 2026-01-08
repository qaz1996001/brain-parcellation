# 01 — 元框架總覽（MetaFramework Overview）

**文件 ID**: META-OVERVIEW-001  
**標題**: 通用產品與軟體開發元框架 — 快速入門與概念說明  
**版本**: v1.0.0  
**狀態**: Stable  
**建立日期**: 2025-12-22  
**最後更新**: 2025-12-22  

---

## 1. 什麼是本元框架？（What is This MetaFramework?）

本元框架是一套**通用的、標準化的文檔系統與開發流程**，旨在幫助團隊：

1. **結構化管理需求**：從利害關係人願景到可測試的系統需求
2. **建立完整追溯性**：需求→設計→實作→測試的雙向追溯
3. **符合國際標準**：內建 ISO/IEC/IEEE 29148 等標準支援
4. **人機協作開發**：明確人類與 AI 的職責分工
5. **支援階段式交付**：大型專案拆分為多個可驗證階段

### 1.1 核心價值主張

| 傳統開發痛點 | 元框架解決方案 |
|------------|--------------|
| 需求散亂、口頭傳達、易遺忘 | 標準化需求文檔（PRD/SR/SD）+ 唯一 ID 管理 |
| 設計與實作脫節 | 強制追溯性（每個設計必須對應需求，每個需求必須有設計） |
| 變更混亂、影響難評估 | OpenSpec 變更管理系統（提案→審核→實作→歸檔） |
| 文檔與程式碼不同步 | 每次變更歸檔必須同步更新文檔 |
| AI 產出品質不穩定 | 提供明確的規範與範本，AI 遵循執行 |
| 合規性證明困難 | 內建標準合規性範本與檢查清單 |

---

## 2. 框架核心組成（Core Components）

### 2.1 三層文檔架構（Three-Layer Documentation）

```
┌────────────────────────────────────────────────────┐
│   Layer 1: 系統層（System Level）                   │
│   - SYSTEM_PRD_SR_SD.md                            │
│   - 產品願景、系統邊界、系統需求                      │
│   - 讀者：產品經理、專案負責人、利害關係人            │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│   Layer 2: 子系統層（Subsystem Level）              │
│   - FRONTEND_PRD_SR_SD.md                          │
│   - BACKEND_PRD_SR_SD.md                           │
│   - FIRMWARE_PRD_SR_SD.md                          │
│   - 子系統需求、子系統設計                           │
│   - 讀者：領域工程師（前端、後端、韌體等）            │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│   Layer 3: 模組層（Module Level）                   │
│   - API 規格、資料庫設計、元件設計                   │
│   - 具體技術實作細節                                 │
│   - 讀者：開發人員、AI                              │
└────────────────────────────────────────────────────┘
```

**關鍵原則**：
- **向上追溯**：每個下層文檔都必須明確標註對應的上層需求 ID
- **向下展開**：每個上層需求都必須在下層有具體的設計或實作
- **雙向驗證**：可從需求查設計，也可從設計查需求

### 2.2 OpenSpec 變更管理（OpenSpec Change Management）

OpenSpec 是本框架的**變更管理核心**，處理所有系統變更：

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: 創建提案（Create Proposal）                │
│  - proposal.md: 為什麼改？改什麼？影響什麼？          │
│  - tasks.md: 實作檢查清單                            │
│  - design.md: 技術決策（可選）                       │
│  - specs/[capability]/spec.md: 需求變更差異          │
│                                                      │
│  人類職責：審核提案的商業價值與技術可行性              │
│  AI 職責：撰寫提案文檔、產生差異規格                  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Stage 2: 實作變更（Implement Change）               │
│  - 依據 tasks.md 逐項完成開發                        │
│  - 撰寫測試、更新文檔                                │
│  - 通過 CI/CD 驗證                                   │
│                                                      │
│  人類職責：驗收功能、進行整合測試                     │
│  AI 職責：依據規範完成開發、測試、文檔                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Stage 3: 歸檔變更（Archive Change）                 │
│  - 將 changes/[id]/ 移至 changes/archive/            │
│  - 更新 specs/[capability]/spec.md                  │
│  - 更新 PRD/SR/SD 文檔對應章節                       │
│  - 更新追溯矩陣                                      │
│                                                      │
│  人類職責：最終審核與部署決策                         │
│  AI 職責：歸檔變更、更新文檔、更新追溯矩陣            │
└─────────────────────────────────────────────────────┘
```

**關鍵原則**：
- **一次只改一件事**：每個變更提案專注於單一功能或修復
- **先提案後實作**：重大變更必須先通過提案審核
- **文檔與程式碼同步**：歸檔時必須更新所有相關文檔

### 2.3 標準合規性框架（Standards Compliance Framework）

內建國際標準支援，確保文檔符合業界最佳實踐：

| 標準 | 適用領域 | 本框架支援 |
|------|---------|-----------|
| **ISO/IEC/IEEE 29148:2018** | 需求工程 | ✅ PRD/SR 範本、需求品質檢查清單、追溯性指南 |
| **ISO 9001** | 品質管理系統 | 🟡 流程文檔範本（可擴展） |
| **ISO 14971** | 工業設備風險管理 | 🟡 風險橋接範本 |

**使用方式**：
1. 根據專案性質選擇適用標準
2. 使用 `regulations/[標準]/compliance-mapping-template.md` 建立合規性對照表
3. 定期進行合規性自我審計（每季度一次）

---

## 3. 人機協作模式（Human-AI Collaboration）

### 3.1 職責分工

| 階段 | 人類負責 | AI 負責 |
|------|---------|--------|
| **需求定義** | - 訪談利害關係人<br>- 定義產品願景<br>- 編寫 UR-xxx（使用者需求） | - 協助結構化需求<br>- 檢查需求完整性<br>- 產生需求 ID 與追溯表 |
| **系統設計** | - 審核技術方案<br>- 決策關鍵架構<br>- 評估技術風險 | - 撰寫設計文檔<br>- 產生架構圖<br>- 評估技術選項 |
| **變更提案** | - 審核提案商業價值<br>- 評估變更影響<br>- 批准/拒絕提案 | - 撰寫 proposal.md<br>- 產生 tasks.md<br>- 撰寫 spec deltas |
| **開發實作** | - 驗收功能<br>- 整合測試<br>- 程式碼審查（關鍵部分） | - 依據 tasks.md 實作<br>- 撰寫單元測試<br>- 更新技術文檔 |
| **變更歸檔** | - 最終審核<br>- 部署決策<br>- 驗證文檔一致性 | - 歸檔變更<br>- 更新 specs/<br>- 更新 PRD/SR/SD<br>- 更新追溯矩陣 |

### 3.2 AI 使用指南（For AI Agents）

**當你（AI）接收到任務時**：

1. **首先檢查**：
   - 閱讀 `requirements/SYSTEM_PRD_SR_SD.md` 了解總體目標
   - 閱讀相關子系統的 PRD/SR/SD（如 `FRONTEND_PRD_SR_SD.md`）
   - 檢查 `openspec list` 查看是否有進行中的變更提案

2. **判斷需求類型**：
   - **新功能**：創建 OpenSpec 提案（`/openspec-proposal`）
   - **Bug 修復**：直接修復（如果只是恢復既有行為）
   - **重構**：創建 OpenSpec 提案（如果涉及架構變更）
   - **文檔更新**：直接更新（如果只是澄清現有內容）

3. **創建提案時**（新功能/重大變更）：
   ```bash
   # 1. 選擇唯一的 change-id（kebab-case, verb-led）
   CHANGE_ID="add-user-notification"
   
   # 2. 創建提案結構
   mkdir -p openspec/changes/$CHANGE_ID/specs/notifications
   
   # 3. 撰寫 proposal.md
   # - Why: 為什麼需要這個變更？
   # - What Changes: 具體改什麼？
   # - Impact: 影響哪些 specs 與程式碼？
   
   # 4. 撰寫 tasks.md（可執行的檢查清單）
   # - [ ] 1.1 建立資料庫 schema
   # - [ ] 1.2 實作 API endpoint
   # - [ ] 1.3 撰寫前端元件
   # - [ ] 1.4 撰寫測試
   
   # 5. 撰寫 specs/notifications/spec.md（需求差異）
   # ## ADDED Requirements
   # ### Requirement: Push Notification
   # #### Scenario: User receives notification
   
   # 6. 驗證提案
   openspec validate $CHANGE_ID --strict
   ```

4. **實作時**：
   - 嚴格遵循 tasks.md 的順序
   - 每完成一項標記為 `- [x]`
   - 確保所有測試通過
   - 更新相關文檔

5. **歸檔時**：
   ```bash
   # 1. 移動到 archive
   openspec archive $CHANGE_ID --yes
   
   # 2. 更新 PRD/SR/SD 對應章節
   # 將 spec delta 的內容同步到 SYSTEM_PRD_SR_SD.md 或子系統文檔
   
   # 3. 更新追溯矩陣
   # 確保新需求的追溯鏈完整
   ```

### 3.3 人類使用指南（For Human Users）

**當你需要與 AI 協作時**：

1. **明確需求**：
   - 使用本框架的範本撰寫需求（UR-xxx, SYS-SR-xxx）
   - 提供具體的驗收標準（Acceptance Criteria）
   - 標註優先級與階段（Phase 1/2/3）

2. **審核 AI 提案**：
   - 檢查 `proposal.md` 的 Why 是否合理
   - 檢查 `tasks.md` 是否完整可執行
   - 檢查 `spec deltas` 是否符合需求
   - 提出修改意見或直接批准

3. **驗收實作**：
   - 執行功能測試（根據 Scenario）
   - 進行整合測試
   - 檢查文檔是否更新
   - 驗證追溯性是否完整

4. **定期審計**：
   - 每週檢查 OpenSpec 變更進度（`openspec list`）
   - 每月檢查需求覆蓋率（所有 SR 都有對應設計/實作）
   - 每季度進行合規性自我審計

---

## 4. 階段式開發流程（Phase-Based Development）

### 4.1 為什麼要分階段？

大型專案一次性交付風險高、週期長、難驗證。分階段開發可：

- **降低風險**：每階段獨立驗證，及早發現問題
- **快速交付**：Phase 1 即可產出可用的 MVP
- **靈活調整**：根據 Phase 1 的反饋調整 Phase 2/3
- **人員管理**：階段里程碑便於團隊協調與激勵

### 4.2 階段劃分原則

```
Phase 1: MVP（最小可行產品）— 8-12 週
├─ 目標：驗證核心價值假設
├─ 範圍：最關鍵的 3-5 個功能
├─ 驗收：可獨立運行、可演示、有基礎文檔
└─ 範例：用戶登入、資料 CRUD、基礎 UI

Phase 2: 完整功能 — 12-16 週
├─ 目標：補齊完整產品功能
├─ 範圍：進階功能、整合外部系統、優化 UX
├─ 驗收：所有核心需求實作完成、通過整合測試
└─ 範例：進階搜尋、批量操作、權限管理、API 整合

Phase 3: 規模化與優化 — 8-12 週
├─ 目標：效能優化、規模化部署、合規性完善
├─ 範圍：快取、負載平衡、監控、安全強化
├─ 驗收：通過壓力測試、合規性審計、完整文檔
└─ 範例：Redis 快取、CDN、日誌系統、合規性證明
```

### 4.3 階段間的文檔管理

每階段都有獨立的需求文檔版本：

```
docs/requirements/
├── 01_SYSTEM_PRD_SR_SD.md         # v1.0.0-Phase1
├── 01_SYSTEM_PRD_SR_SD_Phase2.md  # v2.0.0-Phase2（或附錄於同一文件）
└── 01_SYSTEM_PRD_SR_SD_Phase3.md  # v3.0.0-Phase3
```

**版本管理規則**：
- **Phase 1 完成**：凍結 v1.0.0-Phase1，開始撰寫 v2.0.0-Phase2
- **Phase 2 啟動**：不修改 Phase 1 文檔（除非發現錯誤），新需求寫在 Phase 2 文檔
- **追溯性**：Phase 2 需求可追溯到 Phase 1 設計（如 `Depends on: SYS-SR-010 (Phase 1)`）

---

## 5. 追溯性管理（Traceability Management）

### 5.1 為什麼需要追溯性？

追溯性確保：
- **需求可驗證**：每個需求都有測試案例驗證
- **變更可評估**：修改一個設計時，可快速找到影響的需求與測試
- **合規性可證明**：稽核時可展示需求→設計→實作→測試的完整鏈
- **知識可傳承**：新成員可透過追溯鏈理解設計決策

### 5.2 追溯鏈範例

```
UR-003: 使用者應能使用 AI 分析報告內容
    ↓ Traces to
SYS-SR-010: 系統應提供 AI 報告分析功能，支援分類、提取、評分
    ↓ Traces to
BE-SR-045: 後端應提供 `/api/reports/{id}/analyze` POST 端點
BE-SR-046: 後端應整合 Ollama LLM 客戶端
    ↓ Traces to
API-010: POST /api/reports/{id}/analyze 設計規格
DB-005: ai_annotations 資料表設計
    ↓ Traced by
CODE: backend/services/ai_service.py:analyze_report()
CODE: backend/api/reports.py:analyze_report_endpoint()
    ↓ Verified by
TEST: tests/test_ai_service.py::test_analyze_report_success()
TEST: tests/test_api_reports.py::test_analyze_endpoint()
```

### 5.3 追溯矩陣範例

| UR ID | SYS-SR ID | Subsystem-SR ID | Design ID | Code Location | Test Case ID | Verification Method |
|-------|-----------|----------------|-----------|---------------|--------------|---------------------|
| UR-003 | SYS-SR-010 | BE-SR-045, BE-SR-046 | API-010, DB-005 | `backend/services/ai_service.py:45` | TC-AI-001, TC-AI-002 | Test |
| UR-004 | SYS-SR-011 | FE-SR-020 | UI-003 | `frontend/components/ReportDetail.tsx:120` | TC-UI-005 | Demo |

**維護規則**：
- 每次創建新需求時，必須同時建立追溯鏈
- 每次歸檔變更時，必須更新追溯矩陣
- 使用工具輔助（如 `grep` 搜尋 ID、腳本驗證追溯完整性）

---

## 6. 常見使用場景（Common Use Cases）

### 場景 1：啟動全新專案

**步驟**：
1. 複製範本文件到專案 `docs/requirements/`
2. 撰寫 `SYSTEM_PRD_SR_SD.md`（定義產品願景、使用者需求）
3. 劃分階段（Phase 1/2/3）與里程碑
4. 拆解為子系統文檔（`FRONTEND_PRD_SR_SD.md`, `BACKEND_PRD_SR_SD.md`）
5. 建立初始追溯矩陣
6. 初始化 OpenSpec（`openspec init`）
7. 創建第一個變更提案（如 `setup-project-infrastructure`）

### 場景 2：為現有專案補文檔

**步驟**：
1. 進行需求與設計的現狀盤點（訪談開發人員、檢視程式碼）
2. 逆向工程：從程式碼提取隱含的需求與設計
3. 撰寫 PRD/SR/SD 文檔（可先寫 Phase 1 已完成部分）
4. 建立追溯矩陣（至少覆蓋核心功能）
5. 導入 OpenSpec 管理後續變更
6. 漸進式補齊文檔（每次變更時同步更新）

### 場景 3：跨團隊協作（前後端分離）

**步驟**：
1. 系統層由產品經理與架構師共同撰寫（`SYSTEM_PRD_SR_SD.md`）
2. 前端團隊維護 `FRONTEND_PRD_SR_SD.md`，後端團隊維護 `BACKEND_PRD_SR_SD.md`
3. 定義清楚的 API Contract（`API_SPECIFICATION.md`），作為前後端介面約定
4. 前後端各自創建 OpenSpec 提案，但需標註 Dependencies
5. 定期同步追溯矩陣，確保前後端需求對齊


---

## 7. 工具與自動化（Tools and Automation）

### 7.1 推薦工具

| 用途 | 工具 | 說明 |
|------|------|------|
| **需求管理** | Markdown + Git | 版本控制、易於協作 |
| **變更管理** | OpenSpec CLI | 提案、驗證、歸檔自動化 |
| **追溯性檢查** | 自訂腳本（Python/Shell） | 驗證追溯鏈完整性 |
| **文檔生成** | Pandoc | Markdown 轉 PDF/Word |
| **圖表繪製** | Mermaid / PlantUML | 架構圖、流程圖 |
| **合規性檢查** | 自訂檢查清單 | 對照標準要求逐項檢查 |

### 7.2 自動化腳本範例

**追溯性驗證腳本**：

```python
# scripts/validate_traceability.py
import re
from pathlib import Path

def extract_requirement_ids(file_path):
    """從文件中提取所有需求 ID"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return set(re.findall(r'(UR-\d+|SYS-SR-\d+|FE-SR-\d+|BE-SR-\d+)', content))

def validate_traceability():
    # 提取所有 UR
    ur_ids = extract_requirement_ids('docs/requirements/USER_REQUIREMENTS.md')
    
    # 提取所有 SYS-SR
    sys_sr_ids = extract_requirement_ids('docs/requirements/01_SYSTEM_PRD_SR_SD.md')
    
    # 檢查每個 UR 是否有對應的 SYS-SR
    for ur_id in ur_ids:
        if ur_id not in open('docs/requirements/01_SYSTEM_PRD_SR_SD.md').read():
            print(f"❌ {ur_id} 未追溯到系統需求")
    
    print("✅ 追溯性檢查完成")

if __name__ == '__main__':
    validate_traceability()
```

**需求覆蓋率報告**：

```bash
# scripts/coverage_report.sh
#!/bin/bash

echo "📊 需求覆蓋率報告"
echo "================="

TOTAL_UR=$(grep -c "^### UR-" docs/requirements/USER_REQUIREMENTS.md)
TRACED_UR=$(grep -o "UR-[0-9]\+" docs/requirements/01_SYSTEM_PRD_SR_SD.md | sort -u | wc -l)

echo "總使用者需求數: $TOTAL_UR"
echo "已追溯需求數: $TRACED_UR"
echo "覆蓋率: $(($TRACED_UR * 100 / $TOTAL_UR))%"
```

---

## 8. 最佳實踐（Best Practices）

### 8.1 需求撰寫

✅ **好的需求**：
- 使用 SHALL/MUST（強制）或 SHOULD（建議）
- 可測試、可驗證
- 有明確的驗收標準（Scenario）
- 唯一 ID、版本號、狀態

```markdown
### SYS-SR-010: AI 報告分析
系統 SHALL 提供 AI 報告分析功能，支援以下分析類型：
- 分類（Classification）
- 提取（Extraction）
- 評分（Scoring）

#### Scenario: 用戶提交分析請求
- **WHEN** 用戶選擇報告並提交 AI 分析請求
- **THEN** 系統在 60 秒內返回分析結果
- **AND** 分析結果包含置信度評分（0.0-1.0）
```

❌ **不好的需求**：
- 模糊不清："系統應該盡可能快"
- 無法測試："系統應該很美觀"
- 無 ID："報告分析功能"

### 8.2 設計文檔

✅ **好的設計**：
- 明確對應需求 ID（Traces to: SYS-SR-010）
- 包含技術決策理由
- 有架構圖或流程圖
- 標註風險與限制

❌ **不好的設計**：
- 只有程式碼，沒有說明
- 沒有對應需求
- 技術選擇無理由

### 8.3 OpenSpec 提案

✅ **好的提案**：
- 清楚的 Why（為什麼需要這個變更？）
- 具體的 What（改什麼？影響什麼？）
- 可執行的 tasks.md（檢查清單）
- 完整的 spec deltas（ADDED/MODIFIED/REMOVED）

❌ **不好的提案**：
- 只有 "add feature X"，沒有 Why
- tasks.md 太抽象（如 "implement backend"）
- spec deltas 缺少 Scenario

---

## 9. 故障排除（Troubleshooting）

### 問題 1：需求追溯鏈斷裂

**症狀**：某個 SYS-SR 找不到對應的上層 UR  
**原因**：新增需求時忘記更新追溯矩陣  
**解決**：
1. 執行追溯性驗證腳本（`scripts/validate_traceability.py`）
2. 補齊缺失的追溯關係
3. 更新追溯矩陣文檔

### 問題 2：OpenSpec 驗證失敗

**症狀**：`openspec validate <change-id> --strict` 報錯  
**原因**：spec delta 格式不正確（如 Scenario 使用錯誤的標題層級）  
**解決**：
1. 檢查 `openspec show <change-id> --json --deltas-only` 看哪裡解析失敗
2. 確保 Scenario 使用 `#### Scenario:` 格式（4 個井號）
3. 確保每個 Requirement 至少有一個 Scenario

### 問題 3：文檔與程式碼不同步

**症狀**：程式碼已改，但 PRD/SR/SD 沒更新  
**原因**：歸檔變更時忘記同步文檔  
**解決**：
1. 建立檢查清單（歸檔前必須確認文檔已更新）
2. 使用 Git hook 或 CI 檢查（如程式碼變更但文檔未更新則阻止提交）
3. 定期文檔審計（每月一次）

---

## 10. 下一步（Next Steps）

### 第一次使用本框架？

1. ✅ 閱讀本文件（你已經完成！）
2. 📖 查看 [`requirements/00_REQUIREMENTS_INDEX.md`](requirements/00_REQUIREMENTS_INDEX.md) 選擇範本
3. 📝 使用 `TEMPLATE_SYSTEM_PRD_SR_SD.md` 創建第一份系統需求文檔
4. 🔧 初始化 OpenSpec（`openspec init`）
5. 🚀 創建第一個變更提案（`/openspec-proposal`）

### 進階主題

- 📚 [`guides/TRACEABILITY_MANAGEMENT.md`](guides/TRACEABILITY_MANAGEMENT.md) — 深入了解追溯性管理
- 🤖 [`guides/AI_COLLABORATION_PATTERNS.md`](guides/AI_COLLABORATION_PATTERNS.md) — 人機協作模式
- 🔄 [`guides/MIGRATION_GUIDE.md`](guides/MIGRATION_GUIDE.md) — 為現有專案導入框架（待創建）

### 需要幫助？

- 查看範例專案（`docs/resource/01/` — 影像數據平台實例）
- 執行驗證工具（`scripts/validate_traceability.py`）
- 提出議題或回饋

---

**文檔版本**: v1.0.0  
**維護團隊**: MetaFramework Core Team  
**最後審核**: 2025-12-22

