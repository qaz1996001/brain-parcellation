# TEMPLATE: 系統層 PRD / SR / SD（System-Level PRD/SR/SD Template）

> **使用說明**：本文件為通用範本，請依實際專案調整。複製此範本後：
> 1. 替換所有 `[專案名稱]` 為實際專案名稱
> 2. 填寫文件資訊（版本、作者、狀態等）
> 3. 依照章節結構填寫需求與設計內容
> 4. 刪除或調整不適用的章節
> 5. 建立追溯矩陣連結需求與設計

---

**文件 ID**: SYS-PRD-SR-SD-001  
**標題**: [專案名稱] — 系統層 PRD / SR / SD  
**版本**: v1.0.0-Phase1  
**狀態**: Draft  
**建立日期**: YYYY-MM-DD  
**最後更新**: YYYY-MM-DD  
**作者**: [待填]  
**審核人**: [技術負責人 / 品質負責人 / 產品負責人 待填]  
**適用階段**: Phase 1  

---

## 變更歷史（Change History）

| 版本 | 日期 | 修改者 | 變更摘要 |
|------|------|--------|---------|
| v0.1 | YYYY-MM-DD | [姓名] | 初始草稿 |
| v1.0 | YYYY-MM-DD | [姓名] | Phase 1 正式版 |

---

## 1. 範圍與系統概述（System Scope & Overview）

### 1.1 專案背景（Project Background）

> 簡述專案的業務背景、問題域、市場需求等。

**範例**：
```
本專案旨在開發一套 [領域] 系統，解決 [問題]，為 [目標使用者] 提供 [核心價值]。
基於 [技術基礎/資料來源]，構建 [系統類型] 平台。
```

### 1.2 專案目標（Project Goals）

**Phase 1 目標**（當前版本）：
1. [目標 1]：[具體說明]
2. [目標 2]：[具體說明]
3. [目標 3]：[具體說明]

**長期願景**（Phase 2+）：
- [未來目標 1]
- [未來目標 2]

### 1.3 系統範圍（System Scope）

**在範圍內**（Phase 1）：
- ✅ [功能 1]：[說明]
- ✅ [功能 2]：[說明]
- ✅ [功能 3]：[說明]

**不在範圍內**（Phase 1）：
- ❌ [功能 X]：[留待 Phase 2 實作]
- ❌ [功能 Y]：[非必要功能]

### 1.4 系統邊界（System Boundaries）

**內部組件**（本系統涵蓋）：
- [子系統 1]：[說明]
- [子系統 2]：[說明]

**外部介面**（與外部系統的互動）：
- [外部系統 1]：[介面類型、協議]
- [外部系統 2]：[介面類型、協議]

**假設與依賴**：
- 假設 1：[說明]
- 依賴 1：[說明]

---

## 2. 利害關係人與使用者（Stakeholders and Users）

### 2.1 利害關係人（Stakeholders）

| 角色 | 利益相關點 | 參與程度 |
|------|-----------|---------|
| [角色 1] | [利益點] | 高/中/低 |
| [角色 2] | [利益點] | 高/中/低 |

### 2.2 使用者角色（User Roles）

| 使用者角色 | 說明 | 主要需求 |
|-----------|------|---------|
| [角色 1] | [說明] | [需求 1], [需求 2] |
| [角色 2] | [說明] | [需求 3], [需求 4] |

---

## 3. 使用者需求（User Requirements）

> 使用者需求（UR-xxx）來自利害關係人訪談、市場調研等，描述「使用者想要什麼」。

### 3.1 功能性使用者需求（Functional User Requirements）

#### UR-001: [需求名稱]
**描述**：[使用者希望達成什麼？]

**優先級**：高/中/低  
**來源**：[訪談記錄/市場調研/...]  
**驗收標準**：
- [標準 1]
- [標準 2]

**追溯關係**：
- Traced by: SYS-SR-xxx, SYS-SR-yyy

---

#### UR-002: [需求名稱]
**描述**：[...]

（依此類推，建議 Phase 1 有 5-10 條核心 UR）

---

### 3.2 非功能性使用者需求（Non-Functional User Requirements）

#### NFR-PERF-001: 系統回應時間
**描述**：系統操作應迅速回應，不影響使用體驗。

**具體指標**：
- 一般查詢操作：< 2 秒
- 複雜分析操作：< 30 秒

**追溯關係**：
- Traced by: SYS-SR-xxx

---

#### NFR-SEC-001: 資料安全
**描述**：使用者資料應受到保護，防止未經授權的存取。

**追溯關係**：
- Traced by: SYS-SR-xxx

---

## 4. 系統需求規格（System Requirements Specification）

> 系統需求（SYS-SR-xxx）將使用者需求轉換為可驗證的系統行為，描述「系統應該如何運作」。

### 4.1 SYS-PRD: 系統產品需求（System Product Requirements）

#### SYS-PRD-001: [功能模組名稱]
**說明**：系統應提供 [功能描述]，以滿足 [使用者需求]。

**來源 UR**：UR-001, UR-002  
**追溯關係**：
- Traces to: UR-001, UR-002
- Traced by: SYS-SR-010, SYS-SR-011

---

### 4.2 SYS-SR: 系統軟體需求（System Software Requirements）

#### SYS-SR-010: [具體系統功能需求]
**描述**：系統 SHALL [可驗證的行為描述]。

**驗證方式**：Test / Demo / Inspection / Analysis  
**優先級**：Must / Should / Could  
**來源 PRD**：SYS-PRD-001  

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
- Traces to: SYS-PRD-001, UR-001
- Traced by: FE-SR-020, BE-SR-045
- Verified by: TC-SYS-010-001, TC-SYS-010-002

---

#### SYS-SR-011: [另一個系統需求]
（依此類推）

---

### 4.3 非功能性系統需求（Non-Functional System Requirements）

#### SYS-SR-NFR-001: 效能需求
**描述**：系統 SHALL 在正常負載下（[定義負載]）滿足以下效能指標：
- [指標 1]：[數值]
- [指標 2]：[數值]

**驗證方式**：Performance Test  
**追溯關係**：
- Traces to: NFR-PERF-001

---

#### SYS-SR-NFR-002: 安全需求
**描述**：系統 SHALL 實施以下安全措施：
- [措施 1]
- [措施 2]

**驗證方式**：Security Audit, Penetration Test  
**追溯關係**：
- Traces to: NFR-SEC-001

---

## 5. 系統設計（System Design）

> 系統設計（SYS-SD-xxx）描述「如何實現系統需求」，包含架構、元件、介面等。

### 5.1 系統架構概覽（System Architecture Overview）

**架構圖**：
```
[插入系統架構圖]
例如：
┌─────────────┐       ┌─────────────┐
│   前端      │ ◄───► │   後端      │
│  (React)    │       │ (FastAPI)   │
└─────────────┘       └─────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │   資料庫    │
                      │ (PostgreSQL)│
                      └─────────────┘
```

**元件說明**：
- **[元件 1]**：[職責、技術]
- **[元件 2]**：[職責、技術]

### 5.2 子系統劃分（Subsystem Decomposition）

| 子系統 | 職責 | 對應需求 | 文檔參考 |
|--------|------|---------|---------|
| [前端] | [職責說明] | SYS-SR-010, SYS-SR-011 | `02_FRONTEND_PRD_SR_SD.md` |
| [後端] | [職責說明] | SYS-SR-012, SYS-SR-013 | `03_BACKEND_PRD_SR_SD.md` |

### 5.3 關鍵技術決策（Key Technical Decisions）

#### 決策 1: [技術選擇]
**背景**：[為什麼需要做這個決策？]  
**選項**：
- 選項 A：[優點/缺點]
- 選項 B：[優點/缺點]

**決定**：選擇 [選項 X]  
**理由**：[為什麼選擇這個？]  
**風險**：[潛在風險與緩解措施]

---

### 5.4 資料模型概覽（Data Model Overview）

**核心實體**：
- **[實體 1]**：[說明、屬性]
- **[實體 2]**：[說明、屬性]

**ER 圖**：
```
[插入簡化的 ER 圖]
```

詳細設計參見：`database/DATABASE_DESIGN.md`

### 5.5 介面設計概覽（Interface Design Overview）

**使用者介面（UI）**：
- [主要頁面 1]：[說明]
- [主要頁面 2]：[說明]

詳細設計參見：`frontend/UI_DESIGN.md`

**API 介面**：
- [端點 1]：[說明]
- [端點 2]：[說明]

詳細設計參見：`api/API_SPECIFICATION.md`

---

## 6. 追溯矩陣（Traceability Matrix）

### 6.1 需求追溯表（Requirements Traceability）

| UR ID | SYS-PRD ID | SYS-SR ID | Subsystem-SR ID | Design ID | Code Location | Test Case ID | Verification Method | Status |
|-------|-----------|-----------|----------------|-----------|---------------|--------------|---------------------|--------|
| UR-001 | SYS-PRD-001 | SYS-SR-010 | FE-SR-020, BE-SR-045 | API-010, UI-003 | `backend/services/xxx.py:45` | TC-001, TC-002 | Test | Verified |
| UR-002 | SYS-PRD-001 | SYS-SR-011 | BE-SR-046 | DB-005 | `backend/models/xxx.py:20` | TC-003 | Test | In Progress |

### 6.2 設計追溯表（Design Traceability）

| Design ID | Type | Description | Traces to SYS-SR | Implemented in | Test Case ID |
|-----------|------|-------------|-----------------|----------------|--------------|
| API-010 | API Endpoint | [說明] | SYS-SR-010 | `backend/api/xxx.py` | TC-API-010 |
| UI-003 | UI Component | [說明] | SYS-SR-010 | `frontend/components/xxx.tsx` | TC-UI-003 |

---

## 7. 非功能性需求（Non-Functional Requirements）

### 7.1 效能需求（Performance Requirements）

| 指標 | 目標值 | 測量方式 | 對應 SR |
|------|--------|---------|---------|
| 頁面載入時間 | < 2 秒 | Lighthouse | SYS-SR-NFR-001 |
| API 回應時間 | < 500 ms | APM 監控 | SYS-SR-NFR-001 |
| 並發使用者數 | 100 人 | 壓力測試 | SYS-SR-NFR-001 |

### 7.2 安全需求（Security Requirements）

- **身份認證**：[機制]
- **授權控制**：[機制]
- **資料加密**：[機制]
- **稽核日誌**：[機制]

對應 SR：SYS-SR-NFR-002

### 7.3 可用性需求（Usability Requirements）

- **學習時間**：新使用者 < [時間] 內上手
- **操作效率**：[任務] 完成時間 < [時間]
- **錯誤率**：使用者操作錯誤率 < [百分比]

對應 SR：SYS-SR-NFR-003

### 7.4 可維護性需求（Maintainability Requirements）

- **程式碼覆蓋率**：> 70%
- **文檔完整性**：所有模組有設計文檔
- **部署自動化**：CI/CD 管線

對應 SR：SYS-SR-NFR-004

---

## 8. 風險與限制（Risks and Constraints）

### 8.1 技術風險（Technical Risks）

| 風險項 | 影響 | 機率 | 緩解措施 | 負責人 |
|--------|------|------|---------|--------|
| [風險 1] | 高/中/低 | 高/中/低 | [措施] | [姓名] |

### 8.2 業務風險（Business Risks）

| 風險項 | 影響 | 機率 | 緩解措施 | 負責人 |
|--------|------|------|---------|--------|
| [風險 1] | 高/中/低 | 高/中/低 | [措施] | [姓名] |

### 8.3 限制與約束（Constraints）

- **時程限制**：Phase 1 必須在 [日期] 前完成
- **資源限制**：[人力/預算/設備]
- **技術限制**：[技術平台/相容性]
- **法規限制**：[合規性要求]

---

## 9. 驗收標準（Acceptance Criteria）

### 9.1 Phase 1 驗收標準

**功能完整性**：
- ✅ 所有 Phase 1 的 UR 都已實作
- ✅ 所有 SYS-SR 都有對應的設計與實作
- ✅ 追溯矩陣完整無斷鏈

**品質標準**：
- ✅ 單元測試覆蓋率 > 70%
- ✅ 整合測試通過率 100%
- ✅ 核心 API 回應時間 < 2 秒
- ✅ 無 Critical/Blocker 級別 Bug

**文檔完整性**：
- ✅ PRD/SR/SD 文檔齊全
- ✅ API 文檔完整
- ✅ 使用者手冊（或操作指南）完成

**可演示性**：
- ✅ 可向利害關係人演示核心功能
- ✅ 系統可在目標環境部署與運行

---

## 10. 附錄（Appendix）

### 10.1 名詞解釋（Glossary）

| 術語 | 解釋 |
|------|------|
| [術語 1] | [解釋] |

### 10.2 參考文件（References）

- [文件 1]：[說明]
- [文件 2]：[說明]

### 10.3 相關標準（Related Standards）

- ISO/IEC/IEEE 29148:2018 — Requirements Engineering

---

**文檔版本**: v1.0.0  
**維護人**: [姓名]  
**最後審核**: YYYY-MM-DD

