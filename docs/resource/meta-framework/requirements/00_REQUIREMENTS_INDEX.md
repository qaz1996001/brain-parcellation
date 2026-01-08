# 需求文檔索引（Requirements Documentation Index）

**文件 ID**: REQ-INDEX-001  
**標題**: 元框架需求文檔範本與指南索引  
**版本**: v1.0.0  
**狀態**: Stable  
**建立日期**: 2025-12-22  

---

## 1. 需求文檔體系（Requirements Documentation System）

本目錄提供**通用的需求文檔範本**，可套用於任何產品或軟體專案。

### 1.1 文檔層級

```
需求文檔架構
├── Layer 1: 系統層
│   ├── 使用者需求（User Requirements: UR-xxx）
│   ├── 系統 PRD（Product Requirements）
│   ├── 系統 SR（System Requirements: SYS-SR-xxx）
│   └── 系統設計（System Design: SYS-SD-xxx）
│
├── Layer 2: 子系統層
│   ├── 前端 PRD/SR/SD（Frontend: FE-PRD-xxx, FE-SR-xxx）
│   ├── 後端 PRD/SR/SD（Backend: BE-PRD-xxx, BE-SR-xxx）
│   ├── 韌體 PRD/SR/SD（Firmware: FW-PRD-xxx, FW-SR-xxx）
│   └── 其他子系統...
│
└── Layer 3: 模組層
    ├── API 規格（API-xxx）
    ├── 資料庫設計（DB-xxx）
    ├── UI 元件設計（UI-xxx）
    └── 其他模組...
```

---

## 2. 文檔範本（Document Templates）

### 2.1 核心範本

| 範本 | 用途 | 適用場景 |
|------|------|---------|
| [`TEMPLATE_SYSTEM_PRD_SR_SD.md`](TEMPLATE_SYSTEM_PRD_SR_SD.md) | 系統層 PRD/SR/SD 範本 | 所有專案必備 |
| [`TEMPLATE_SUBSYSTEM_PRD_SR_SD.md`](TEMPLATE_SUBSYSTEM_PRD_SR_SD.md) | 子系統層 PRD/SR/SD 範本 | 複雜專案（多子系統） |
| [`TEMPLATE_USER_REQUIREMENTS.md`](TEMPLATE_USER_REQUIREMENTS.md) | 使用者需求文檔範本 | 需要詳細記錄利害關係人需求 |
| [`TEMPLATE_TRACEABILITY_MATRIX.md`](TEMPLATE_TRACEABILITY_MATRIX.md) | 追溯矩陣範本 | 所有需要追溯性管理的專案 |

### 2.2 範本使用步驟

1. **選擇範本**：根據專案規模選擇適合的範本
2. **複製到專案**：`cp TEMPLATE_SYSTEM_PRD_SR_SD.md <your-project>/docs/requirements/01_SYSTEM_PRD_SR_SD.md`
3. **填寫基本資訊**：文件 ID、專案名稱、版本、狀態、作者
4. **填寫需求內容**：依照範本結構填寫需求
5. **建立追溯關係**：使用追溯矩陣連結需求、設計、實作
6. **定期更新**：每次變更後更新需求文檔

---

## 3. 指南與最佳實踐（Guides and Best Practices）

| 指南 | 說明 |
|------|------|
| [`HOWTO_WRITE_REQUIREMENTS.md`](HOWTO_WRITE_REQUIREMENTS.md) | 需求撰寫指南（SMART 原則、可測試性、範例） |
| [`REQUIREMENTS_QUALITY_CHECKLIST.md`](REQUIREMENTS_QUALITY_CHECKLIST.md) | 需求品質檢查清單（自我審查用） |
| [`ID_NAMING_CONVENTION.md`](ID_NAMING_CONVENTION.md) | 需求 ID 命名規則（UR-xxx, SYS-SR-xxx 等） |
| [`PHASE_SCOPING_GUIDE.md`](PHASE_SCOPING_GUIDE.md) | 階段範圍劃分指南（如何拆分 Phase 1/2/3） |

---

## 4. 需求 ID 命名規則（Requirement ID Convention）

### 4.1 基本格式

```
<層級>-<類型>-<流水號>
```

### 4.2 常見 ID 類型

| 前綴 | 全稱 | 說明 | 範例 |
|------|------|------|------|
| `UR-` | User Requirement | 使用者需求 | UR-001, UR-002 |
| `SYS-PRD-` | System Product Requirement | 系統產品需求 | SYS-PRD-001 |
| `SYS-SR-` | System Software Requirement | 系統軟體需求 | SYS-SR-010 |
| `FE-PRD-` | Frontend Product Requirement | 前端產品需求 | FE-PRD-001 |
| `FE-SR-` | Frontend Software Requirement | 前端軟體需求 | FE-SR-020 |
| `BE-PRD-` | Backend Product Requirement | 後端產品需求 | BE-PRD-001 |
| `BE-SR-` | Backend Software Requirement | 後端軟體需求 | BE-SR-050 |
| `FW-SR-` | Firmware Software Requirement | 韌體軟體需求 | FW-SR-100 |
| `NFR-` | Non-Functional Requirement | 非功能性需求 | NFR-PERF-001, NFR-SEC-005 |

### 4.3 流水號分配建議

- **UR**: 001-099（使用者需求通常不多）
- **SYS-SR**: 001-199（系統需求）
- **FE-SR**: 001-199（前端需求）
- **BE-SR**: 001-299（後端需求通常較多）
- **NFR**: 依類型細分（PERF, SEC, USABILITY, RELIABILITY 等）

---

## 5. 需求狀態管理（Requirement Status Management）

### 5.1 需求狀態定義

| 狀態 | 說明 | 適用場景 |
|------|------|---------|
| `Draft` | 草稿 | 需求正在撰寫、待審核 |
| `Proposed` | 提議中 | 已完成初稿，等待利害關係人審核 |
| `Approved` | 已核准 | 審核通過，可進入設計階段 |
| `In Progress` | 實作中 | 正在開發實作 |
| `Implemented` | 已實作 | 開發完成，待測試 |
| `Verified` | 已驗證 | 測試通過，需求滿足 |
| `Deprecated` | 已廢棄 | 不再適用（需註明原因與替代方案） |

### 5.2 狀態轉換流程

```
Draft → Proposed → Approved → In Progress → Implemented → Verified
                      ↓                          ↓
                  Deprecated ←──────────────────┘
```

---

## 6. 需求追溯性（Requirement Traceability）

### 6.1 追溯關係類型

| 關係 | 說明 | 範例 |
|------|------|------|
| **Traces to** | 此需求追溯到上層需求 | `SYS-SR-010 Traces to UR-003` |
| **Traced by** | 此需求被下層設計/實作追溯 | `UR-003 Traced by SYS-SR-010, SYS-SR-011` |
| **Depends on** | 此需求依賴其他需求 | `SYS-SR-015 Depends on SYS-SR-010` |
| **Conflicts with** | 此需求與其他需求衝突（需解決） | `SYS-SR-020 Conflicts with SYS-SR-018` |
| **Refines** | 此需求精煉（細化）上層需求 | `FE-SR-025 Refines SYS-SR-010` |

### 6.2 追溯鏈完整性檢查

每個需求都應該能回答：
1. **為什麼需要這個需求？**（Traces to 上層需求）
2. **如何實現這個需求？**（Traced by 設計/實作）
3. **如何驗證這個需求？**（Verified by 測試案例）

---

## 7. 與 OpenSpec 的整合（OpenSpec Integration）

### 7.1 OpenSpec 與 PRD/SR/SD 的關係

- **OpenSpec Changes**：描述**如何改變**現有系統（增量變更）
- **PRD/SR/SD**：描述**系統應該是什麼樣**（當前狀態的完整描述）

### 7.2 同步機制

當歸檔 OpenSpec 變更時：
1. **提取 spec deltas**：從 `openspec/changes/<id>/specs/` 提取 ADDED/MODIFIED/REMOVED
2. **更新 PRD/SR/SD**：將變更同步到對應的需求文檔
3. **更新追溯矩陣**：確保新需求的追溯鏈完整
4. **更新需求狀態**：將新需求標記為 `Approved` 或 `In Progress`

範例：
```markdown
## OpenSpec 變更 add-user-notification 已歸檔

### 新增需求
- BE-SR-080: 系統應提供推播通知功能（來源：openspec/changes/archive/2025-12-20-add-user-notification）
- BE-SR-081: 系統應支援多種通知管道（Email, SMS, Push）

### 追溯關係
- BE-SR-080 Traces to SYS-SR-025（系統通知功能）
- BE-SR-080 Traced by API-015, DB-008
- BE-SR-080 Verified by TC-NOTIF-001, TC-NOTIF-002
```

---

## 8. 需求變更管理（Requirement Change Management）

### 8.1 變更流程

```
需求變更請求
    ↓
影響評估（追溯分析：影響哪些設計、實作、測試？）
    ↓
創建 OpenSpec 提案
    ↓
審核與批准
    ↓
實作變更
    ↓
更新 PRD/SR/SD 與追溯矩陣
    ↓
驗證與驗收
```

### 8.2 變更類型

| 類型 | 說明 | 處理方式 |
|------|------|---------|
| **Minor Change** | 澄清或修正文字，不影響設計 | 直接修改，無需 OpenSpec 提案 |
| **Moderate Change** | 新增或修改需求，影響部分設計 | 創建 OpenSpec 提案，審核後實作 |
| **Major Change** | 架構性變更，影響多個子系統 | 創建 OpenSpec 提案 + design.md，高層審核 |

---

## 9. 常見問題（FAQ）

### Q1: 我的專案很小，需要所有這些文檔嗎？

**A**: 不需要。小專案可以：
- 合併系統層與子系統層文檔（如 `01_PROJECT_REQUIREMENTS.md`）
- 使用簡化的追溯矩陣（Excel 表格即可）
- 專注於核心需求（UR + SR），設計可以較簡略

### Q2: 如何處理跨階段的需求？

**A**: 使用階段標記：
```markdown
### SYS-SR-010: AI 報告分析（Phase 1）
...

### SYS-SR-015: 進階 AI 分析（Phase 2）
- Depends on: SYS-SR-010 (Phase 1)
...
```

### Q3: 需求變更太頻繁，文檔跟不上怎麼辦？

**A**: 
1. 使用 OpenSpec 管理變更（每次變更都有提案記錄）
2. 歸檔時批量更新文檔（而非每次小改都更新）
3. 使用腳本驗證追溯性（自動檢查斷鏈）

---

## 10. 下一步（Next Steps）

### 新專案

1. 複製 `TEMPLATE_SYSTEM_PRD_SR_SD.md` 到專案
2. 閱讀 `HOWTO_WRITE_REQUIREMENTS.md` 學習需求撰寫
3. 填寫使用者需求（UR-xxx）
4. 展開為系統需求（SYS-SR-xxx）
5. 建立追溯矩陣

### 現有專案

1. 盤點現有需求（逆向工程）
2. 使用範本補齊文檔
3. 建立追溯矩陣
4. 導入 OpenSpec 管理後續變更

---

**文檔版本**: v1.0.0  
**維護團隊**: MetaFramework Core Team  
**最後審核**: 2025-12-22

