# 需求撰寫指南（How to Write Requirements）

**文件 ID**: GUIDE-REQ-001  
**版本**: v1.0.0  
**建立日期**: 2025-12-22  
**參考標準**: 
- ISO/IEC/IEEE 29148:2018（需求工程）
- RFC 2119（需求層級關鍵字）
- RFC 8174（RFC 2119 澄清更新）

---

## 目錄

1. [什麼是好的需求？](#1-什麼是好的需求what-makes-a-good-requirement)
2. [需求的層級](#2-需求的層級requirement-levels)
3. [需求撰寫的常見錯誤](#3-需求撰寫的常見錯誤common-mistakes)
4. [需求撰寫的最佳實踐](#4-需求撰寫的最佳實踐best-practices)
   - [4.1 使用規範性語言（RFC 2119）](#41-使用規範性語言normative-language)
5. [需求追溯性](#5-需求追溯性requirement-traceability)
6. [需求變更管理](#6-需求變更管理requirement-change-management)
7. [實戰範例](#7-實戰範例practical-examples)
8. [檢查清單](#8-檢查清單checklist)

---

## 1. 什麼是好的需求？（What Makes a Good Requirement?）

### 1.1 SMART 原則

好的需求應該符合 **SMART** 原則：

| 原則 | 英文 | 說明 | 範例 |
|------|------|------|------|
| **S** | Specific | 具體明確，無歧義 | ❌ "系統應該快" → ✅ "系統應在 2 秒內返回搜尋結果" |
| **M** | Measurable | 可測量、可驗證 | ❌ "介面應該美觀" → ✅ "介面應通過 WCAG 2.1 AA 級無障礙標準" |
| **A** | Achievable | 技術上可實現 | ❌ "系統應讀取使用者心思" → ✅ "系統應根據使用者歷史行為推薦內容" |
| **R** | Relevant | 與專案目標相關 | ❌ "系統應支援區塊鏈" → ✅ "系統應支援資料匯入功能"（若匯入是核心需求） |
| **T** | Testable | 可測試 | ❌ "系統應易於使用" → ✅ "新使用者應在 5 分鐘內完成首次操作"（可用性測試） |

### 1.2 需求撰寫的「黃金法則」

1. **使用規範性語言**：SHALL（必須）、SHOULD（應該）、MAY（可以）
2. **一次只描述一件事**：避免 "系統應該做 A 和 B 和 C"（拆成三條需求）
3. **避免實作細節**：需求描述「做什麼」，設計描述「怎麼做」
4. **包含驗收情境**：至少一個 Scenario 說明如何驗證
5. **明確追溯關係**：標註來源需求與下層設計

---

## 2. 需求的層級（Requirement Levels）

### 2.1 使用者需求（User Requirements: UR-xxx）

**目的**：描述利害關係人或使用者的期望（Why & What）  
**來源**：訪談、市場調研、業務需求  
**讀者**：產品經理、專案經理、利害關係人  

**範本**：
```markdown
#### UR-001: [需求名稱]
**描述**：身為 [角色]，我希望能夠 [做什麼]，以便 [達成什麼目標]。

**優先級**：高/中/低  
**來源**：[訪談記錄/市場調研/...]  
**驗收標準**：
- [標準 1]
- [標準 2]

**追溯關係**：
- Traced by: SYS-SR-010, SYS-SR-011
```

**範例**（好的 UR）：
```markdown
#### UR-003: AI 輔助報告分析
**描述**：身為臨床研究人員，我希望能夠使用 AI 自動分析檢查報告內容，以便快速識別符合研究條件的病例，節省人工閱讀時間。

**優先級**：高  
**來源**：研究人員訪談（2025-11-10）  
**驗收標準**：
- AI 分析單份報告時間 < 60 秒
- 分析結果包含關鍵發現高亮或分類標籤
- 使用者可自訂分析提示詞

**追溯關係**：
- Traced by: SYS-SR-010, SYS-SR-011, SYS-SR-012
```

### 2.2 系統需求（System Requirements: SYS-SR-xxx）

**目的**：將使用者需求轉換為系統行為規格（What in detail）  
**來源**：UR 分析與拆解  
**讀者**：系統架構師、技術負責人  

**範本**：
```markdown
#### SYS-SR-010: [系統功能需求]
**描述**：系統 SHALL [可驗證的行為描述]。

**驗證方式**：Test / Demo / Inspection / Analysis  
**優先級**：Must / Should / Could  
**來源 UR**：UR-003  

**驗收情境（Scenarios）**：

##### Scenario 1: [成功案例]
- **GIVEN** [前置條件]
- **WHEN** [使用者操作]
- **THEN** [預期結果]
- **AND** [額外條件]

##### Scenario 2: [錯誤處理]
- **GIVEN** [異常條件]
- **WHEN** [使用者操作]
- **THEN** [系統行為]

**追溯關係**：
- Traces to: UR-003
- Traced by: FE-SR-020, BE-SR-045
- Verified by: TC-SYS-010-001
```

**範例**（好的 SYS-SR）：
```markdown
#### SYS-SR-010: AI 報告分析功能
**描述**：系統 SHALL 提供 AI 報告分析功能，支援以下分析類型：
- 分類（Classification）：將報告分為正常/異常/疑似
- 提取（Extraction）：提取關鍵測量值或發現
- 高亮（Highlight）：標記關鍵文字片段

**驗證方式**：Test  
**優先級**：Must  
**來源 UR**：UR-003  

**驗收情境**：

##### Scenario 1: 用戶提交分析請求
- **GIVEN** 使用者已選擇一份報告
- **WHEN** 使用者點擊「AI 分析」按鈕並選擇分析類型
- **THEN** 系統在 60 秒內返回分析結果
- **AND** 結果包含置信度評分（0.0-1.0）

##### Scenario 2: 分析失敗處理
- **GIVEN** 使用者已提交分析請求
- **WHEN** AI 服務無法完成分析（超時或錯誤）
- **THEN** 系統顯示錯誤訊息「分析失敗，請稍後重試」
- **AND** 系統記錄錯誤日誌供管理員檢視

**追溯關係**：
- Traces to: UR-003
- Traced by: BE-SR-045（AI 服務後端）, FE-SR-020（報告詳情頁 AI 按鈕）
- Verified by: TC-AI-001, TC-AI-002
```

### 2.3 子系統需求（Subsystem Requirements: FE-SR-xxx, BE-SR-xxx）

**目的**：將系統需求分配給特定子系統（Frontend, Backend, Firmware 等）  
**來源**：SYS-SR 分解  
**讀者**：領域工程師（前端、後端、韌體等）  

**範本**：
```markdown
#### BE-SR-045: [後端具體功能]
**描述**：後端 SHALL 提供 [具體功能]，[具體細節]。

**API 端點**：`POST /api/[resource]/[action]`  
**資料格式**：JSON  
**回應時間**：< X 秒  

**來源 SYS-SR**：SYS-SR-010  

**驗收情境**：
##### Scenario 1: 正常請求
- **GIVEN** 有效的請求 payload
- **WHEN** 呼叫 API
- **THEN** 返回 200 OK 與結果資料

**追溯關係**：
- Traces to: SYS-SR-010
- Traced by: API-010（API 設計規格）, CODE: `backend/services/ai_service.py:45`
- Verified by: TC-API-045
```

---

## 3. 需求撰寫的常見錯誤（Common Mistakes）

### 錯誤 1：需求太模糊

❌ **錯誤範例**：
```markdown
系統應該提供良好的使用者體驗。
```

**問題**：「良好」無法測量，無法驗證。

✅ **正確範例**：
```markdown
SYS-SR-015: 可用性需求
系統 SHALL 滿足以下可用性指標：
- 新使用者在 10 分鐘內完成首次資料匯入（可用性測試）
- 核心操作（搜尋、查看詳情）在 3 次點擊內完成（啟發式評估）
- 介面通過 WCAG 2.1 AA 級無障礙標準（自動化檢查）
```

### 錯誤 2：混淆需求與設計

❌ **錯誤範例**：
```markdown
系統應使用 Redis 快取資料。
```

**問題**：「使用 Redis」是設計決策，不是需求。

✅ **正確範例**：
```markdown
SYS-SR-020: 快取機制
系統 SHALL 實施快取機制，確保重複查詢的回應時間 < 200 ms。
```

（然後在設計文檔中說明：「決定使用 Redis 作為快取層，因為...」）

### 錯誤 3：一條需求描述多件事

❌ **錯誤範例**：
```markdown
系統應支援匯入 Excel、CSV、JSON 格式的資料，並自動驗證資料格式，並在匯入失敗時發送 Email 通知。
```

**問題**：三件事混在一起，無法單獨追溯與驗證。

✅ **正確範例**：
```markdown
SYS-SR-025: 多格式資料匯入
系統 SHALL 支援以下格式的資料匯入：Excel (.xlsx), CSV (.csv), JSON (.json)。

SYS-SR-026: 資料格式驗證
系統 SHALL 在資料匯入時驗證格式，若格式錯誤則拒絕匯入並顯示錯誤訊息。

SYS-SR-027: 匯入失敗通知
系統 SHALL 在資料匯入失敗時，透過 Email 通知使用者，包含錯誤原因與重試建議。
```

### 錯誤 4：缺少驗收情境（Scenario）

❌ **錯誤範例**：
```markdown
系統應提供報告搜尋功能。
```

**問題**：沒有說明如何驗證這個需求。

✅ **正確範例**：
```markdown
SYS-SR-030: 報告全文搜尋
系統 SHALL 提供全文搜尋功能，支援對報告內容、患者姓名、檢查日期進行搜尋。

Scenario 1: 搜尋報告內容
- **GIVEN** 資料庫中有 100 份報告
- **WHEN** 使用者輸入關鍵字「肺結節」並提交搜尋
- **THEN** 系統在 2 秒內返回包含「肺結節」的所有報告
- **AND** 搜尋結果高亮顯示關鍵字

Scenario 2: 搜尋無結果
- **GIVEN** 資料庫中無符合條件的報告
- **WHEN** 使用者提交搜尋
- **THEN** 系統顯示「無符合結果」訊息
```

---

## 4. 需求撰寫的最佳實踐（Best Practices）

### 4.1 使用規範性語言（Normative Language）

#### RFC 2119 關鍵字標準

本框架遵循 **RFC 2119**（Key words for use in RFCs to Indicate Requirement Levels）定義的規範性關鍵字，用於明確表達需求的強制程度。

| RFC 2119 關鍵字 | 中文對應 | 意義 | 使用時機 | 範例 |
|----------------|---------|------|---------|------|
| **MUST / SHALL** | 必須 | 絕對要求，無例外 | 核心功能、合規性要求、安全性關鍵 | "系統 SHALL 在使用者登入失敗 3 次後鎖定帳號 15 分鐘" |
| **MUST NOT / SHALL NOT** | 禁止 | 絕對禁止 | 安全性禁令、合規性禁令 | "系統 SHALL NOT 以明文儲存使用者密碼" |
| **SHOULD** | 應該 | 強烈建議，但在特定情況下可忽略 | 重要但非必要功能、最佳實踐 | "系統 SHOULD 提供深色模式選項" |
| **SHOULD NOT** | 不應該 | 強烈不建議 | 不良實踐、潛在問題 | "系統 SHOULD NOT 在未經確認前刪除資料" |
| **MAY / COULD** | 可以 | 可選，實作者自行決定 | 錦上添花功能、進階選項 | "系統 MAY 支援第三方 OAuth 登入" |
| **RECOMMENDED** | 建議 | 建議採用 | 最佳實踐、優化建議 | "RECOMMENDED 使用 HTTPS 加密傳輸" |
| **NOT RECOMMENDED** | 不建議 | 不建議採用 | 過時做法、次優方案 | "NOT RECOMMENDED 使用 MD5 雜湊" |
| **OPTIONAL** | 可選 | 完全可選 | 非核心功能 | "深色模式為 OPTIONAL 功能" |

#### 關鍵字使用原則

1. **一致性**：全文檔使用相同的關鍵字風格（建議使用大寫以便識別）
2. **明確性**：避免混用（如同時使用 MUST 和 SHALL）
3. **可測試性**：MUST/SHALL 需求必須有明確的驗證方法
4. **合理性**：過度使用 MUST 會導致實作僵化，適當使用 SHOULD

#### 錯誤與正確範例

❌ **錯誤使用**：
```markdown
系統應該盡可能快地回應。（模糊）
系統可能需要加密。（不明確）
系統要能夠處理錯誤。（非規範性語言）
```

✅ **正確使用**：
```markdown
SYS-SR-010: 系統回應時間
系統 SHALL 在正常負載下（<100 並發使用者）於 2 秒內回應查詢請求。

SYS-SR-011: 資料加密
系統 MUST 使用 AES-256 加密儲存敏感資料。

SYS-SR-012: 錯誤處理
系統 SHOULD 記錄所有錯誤到日誌系統，並 SHALL 向使用者顯示友善的錯誤訊息。

SYS-SR-013: OAuth 登入
系統 MAY 支援 Google、GitHub OAuth 登入作為補充登入方式。
```

#### 多層級需求的關鍵字使用

```markdown
SYS-SR-020: 使用者認證（系統層）
系統 SHALL 提供使用者認證功能。
   ↓ 拆解為子系統需求
BE-SR-050: 認證 API（後端層）
後端 MUST 提供 POST /api/auth/login 端點。
後端 SHOULD 支援 JWT Token 有效期可配置。
後端 MAY 支援 Refresh Token 機制。
```

### 4.2 使用主動語態與明確主詞

❌ 被動語態：「資料應被驗證」  
✅ 主動語態：「系統 SHALL 驗證資料格式」

### 4.3 量化需求（Quantify When Possible）

❌ 模糊："系統應該快"  
✅ 量化："系統應在 2 秒內完成搜尋（10,000 筆資料集）"

❌ 模糊:"系統應支援多使用者"  
✅ 量化："系統應支援至少 100 個並發使用者"

### 4.4 使用 Scenario 描述驗收標準

**GIVEN-WHEN-THEN** 格式（Behavior-Driven Development, BDD）：

```markdown
##### Scenario: [情境名稱]
- **GIVEN** [前置條件]（系統狀態、資料準備）
- **WHEN** [觸發事件]（使用者操作、系統事件）
- **THEN** [預期結果]（系統回應、資料變化）
- **AND** [額外條件]（可選，補充細節）
```

範例：
```markdown
##### Scenario: 用戶成功登入
- **GIVEN** 使用者帳號「test@example.com」已註冊且密碼為「Password123」
- **WHEN** 使用者輸入正確帳號密碼並點擊「登入」
- **THEN** 系統驗證成功並導向至首頁
- **AND** 系統產生 JWT Token 有效期 30 分鐘
- **AND** 系統記錄登入日誌（IP、時間）
```

---

## 5. 需求追溯性（Requirement Traceability）

### 5.1 為什麼需要追溯性？

- **影響分析**：當需求變更時，可快速找到受影響的設計與程式碼
- **合規性證明**：稽核時可展示需求→設計→實作→測試的完整鏈
- **覆蓋率檢查**：確保所有需求都有對應的實作與測試
- **知識傳承**：新成員可透過追溯鏈理解設計決策

### 5.2 追溯關係類型

在需求文檔中明確標註：

```markdown
**追溯關係**：
- Traces to: [上層需求 ID]（此需求來源於哪個上層需求？）
- Traced by: [下層設計/實作 ID]（哪些設計或程式碼實作了此需求？）
- Depends on: [相依需求 ID]（此需求依賴哪些其他需求？）
- Verified by: [測試案例 ID]（如何驗證此需求？）
- Related to: [相關需求 ID]（相關但非直接追溯的需求）
```

範例：
```markdown
#### SYS-SR-010: AI 報告分析功能
...
**追溯關係**：
- Traces to: UR-003（使用者需要 AI 輔助分析）
- Traced by: 
  - BE-SR-045（後端 AI 服務）
  - FE-SR-020（前端 AI 分析按鈕）
  - API-010（分析 API 端點設計）
  - DB-005（ai_annotations 資料表）
- Depends on: SYS-SR-008（報告資料已匯入）
- Verified by: TC-AI-001, TC-AI-002, TC-AI-003
- Related to: SYS-SR-011（批量分析功能）
```

---

## 6. 需求變更管理（Requirement Change Management）

### 6.1 變更流程

```
1. 提出變更請求（Change Request）
   ↓
2. 影響評估（Impact Analysis）
   - 追溯受影響的設計、程式碼、測試
   - 評估工時與風險
   ↓
3. 創建 OpenSpec 提案（若需重大變更）
   ↓
4. 審核與批准
   ↓
5. 實作變更
   ↓
6. 更新需求文檔與追溯矩陣
   ↓
7. 驗證與驗收
```

### 6.2 變更類型分級

| 變更類型 | 定義 | 處理方式 |
|---------|------|---------|
| **Minor** | 澄清或修正文字，不影響設計 | 直接修改文檔，記錄變更歷史 |
| **Moderate** | 新增或修改需求，影響部分設計 | 創建 OpenSpec 提案，審核後實作 |
| **Major** | 架構性變更，影響多個子系統 | 創建 OpenSpec 提案 + design.md，高層審核 |

---

## 7. 實戰範例（Practical Examples）

### 範例 1：從使用者故事到系統需求

**使用者故事**：
```
身為臨床研究人員，我希望能夠批量匯入檢查報告資料（Excel 檔案），以便快速建立研究資料集。
```

**拆解為需求**：

```markdown
#### UR-001: 批量匯入報告資料
**描述**：身為臨床研究人員，我希望能夠批量匯入檢查報告資料（Excel 或 CSV 檔案），以便快速建立研究資料集，避免逐筆手動輸入。

**優先級**：高  
**來源**：研究人員訪談  
**驗收標準**：
- 支援 Excel (.xlsx) 與 CSV (.csv) 格式
- 單次匯入至少支援 1000 筆資料
- 匯入過程顯示進度
- 匯入完成後顯示成功/失敗統計

---

#### SYS-SR-001: 資料匯入功能
系統 SHALL 提供資料匯入功能，支援 Excel (.xlsx) 與 CSV (.csv) 格式。

Scenario 1: 成功匯入 Excel 檔案
- **GIVEN** 使用者已準備符合格式的 Excel 檔案（包含必填欄位）
- **WHEN** 使用者上傳檔案並點擊「匯入」
- **THEN** 系統驗證檔案格式並開始匯入
- **AND** 系統顯示進度條（已匯入筆數 / 總筆數）
- **AND** 匯入完成後顯示「成功匯入 X 筆」訊息

---

#### SYS-SR-002: 匯入資料驗證
系統 SHALL 在匯入時驗證資料格式，若格式錯誤則拒絕匯入。

Scenario 1: 必填欄位缺失
- **GIVEN** Excel 檔案中有 10 筆資料，其中 2 筆缺少 patient_id
- **WHEN** 使用者嘗試匯入
- **THEN** 系統顯示錯誤訊息「第 3、5 行缺少必填欄位 patient_id」
- **AND** 系統拒絕匯入，不寫入任何資料

---

#### BE-SR-001: 資料匯入 API
後端 SHALL 提供 `/api/import/reports` POST 端點，接受 multipart/form-data 格式的檔案上傳。

Scenario 1: 正常上傳
- **GIVEN** 有效的 Excel 檔案
- **WHEN** 呼叫 API
- **THEN** 返回 202 Accepted（異步處理）與 task_id
- **AND** 使用者可透過 `/api/tasks/{task_id}` 查詢匯入進度
```

---

## 8. 檢查清單（Checklist）

撰寫需求後，使用此檢查清單自我審查：

**基本要求**：
- [ ] 需求有唯一 ID（如 UR-001, SYS-SR-010）
- [ ] 需求使用規範性語言（SHALL/SHOULD/MAY）
- [ ] 需求描述清楚、具體、無歧義
- [ ] 需求可測試、可驗證
- [ ] 需求有至少一個 Scenario

**追溯性**：
- [ ] 標註了 "Traces to"（對應上層需求）
- [ ] 標註了 "Traced by"（對應下層設計/實作）
- [ ] 標註了 "Verified by"（對應測試案例）

**品質**：
- [ ] 符合 SMART 原則
- [ ] 避免實作細節（設計留給設計文檔）
- [ ] 一次只描述一件事
- [ ] 量化需求（如有具體數值）

**文檔管理**：
- [ ] 填寫了優先級（高/中/低 或 Must/Should/Could）
- [ ] 填寫了來源（訪談/調研/上層需求）
- [ ] 填寫了狀態（Draft/Proposed/Approved/Verified）

---

## 9. 工具與資源（Tools and Resources）

### 推薦工具

- **需求管理**：Markdown + Git（版本控制、協作）
- **追溯性檢查**：自訂腳本（Python/Shell）驗證追溯鏈
- **需求審查**：Pull Request Review（團隊協作審查）
- **需求視覺化**：Mermaid 流程圖、PlantUML

### 延伸閱讀

- **ISO/IEC/IEEE 29148:2018**：Systems and software engineering — Requirements engineering
- **IREB CPRE**：Certified Professional for Requirements Engineering
- **書籍**：《Software Requirements》by Karl Wiegers

---

**文檔版本**: v1.0.0  
**維護團隊**: MetaFramework Core Team  
**最後審核**: 2025-12-22

