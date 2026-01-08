# 規範與合規性文檔索引（Regulations and Compliance Index）

**文件 ID**: REG-META-001  
**標題**: 元框架規範與合規性文檔索引  
**版本**: v1.0.0  
**狀態**: Stable  
**建立日期**: 2025-12-22  

---

## 1. 目的與範圍（Purpose and Scope）

本目錄 (`docs/resource/02/regulations/`) 提供**通用的標準合規性範本**，幫助專案符合國際軟體工程的標準要求。

### 1.1 適用標準

本框架主要參考以下國際標準：

| 標準 | 全稱 | 適用領域 | 本框架支援 |
|------|------|---------|-----------|
| **ISO/IEC/IEEE 29148:2018** | Systems and software engineering — Requirements engineering | 需求工程 | ✅ 完整支援 |
| **ISO 9001:2015** | Quality management systems | 品質管理系統 | 🟡 部分支援（流程文檔） |

---

## 2. 文檔結構（Document Structure）

```
regulations/
├── 00_REGULATIONS_INDEX.md                 # 本文件
└── ISO-IEC-IEEE-29148/                     # 需求工程標準
    ├── compliance-mapping-template.md      # 合規性對照範本
    ├── requirements-quality-checklist.md   # 需求品質檢查清單
    └── stakeholder-requirements-guide.md   # 利害關係人需求指南
```

---

## 3. ISO/IEC/IEEE 29148:2018（需求工程）

### 3.1 標準概述

**ISO/IEC/IEEE 29148:2018** 規範需求工程的流程（獲取、分析、規格）與需求文件的品質標準。

**關鍵要求**：
- **Stakeholder Requirements**：利害關係人需求（StRS）
- **System Requirements Specification**：系統需求規格（SyRS）
- **Software Requirements Specification**：軟體需求規格（SRS）
- **Requirements Traceability**：需求追溯性
- **Requirements Quality Attributes**：需求品質屬性（完整性、一致性、可驗證性等）

### 3.2 本框架的對應

| 標準要求 | 本框架對應 | 文檔位置 |
|---------|-----------|---------|
| Stakeholder Requirements (StRS) | 使用者需求（UR-xxx） | `requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md` § 3 |
| System Requirements (SyRS) | 系統需求（SYS-SR-xxx） | `requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md` § 4 |
| Software Requirements (SRS) | 子系統需求（FE-SR-xxx, BE-SR-xxx） | `requirements/TEMPLATE_SUBSYSTEM_PRD_SR_SD.md` |
| Traceability | 追溯矩陣（Traceability Matrix） | `requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md` § 6 |
| Quality Attributes | 需求品質檢查清單 | `regulations/ISO-IEC-IEEE-29148/requirements-quality-checklist.md` |

### 3.3 範本與指南

- [`compliance-mapping-template.md`](ISO-IEC-IEEE-29148/compliance-mapping-template.md)
  - 合規性對照範本：如何證明符合標準要求
- [`requirements-quality-checklist.md`](ISO-IEC-IEEE-29148/requirements-quality-checklist.md)
  - 需求品質檢查清單：自我審查用
- [`stakeholder-requirements-guide.md`](ISO-IEC-IEEE-29148/stakeholder-requirements-guide.md)
  - 利害關係人需求撰寫指南

---

## 4. 如何使用本框架達成合規（How to Achieve Compliance）

### 4.1 選擇適用標準

**步驟 1：識別專案性質**

| 專案類型 | 建議標準 |
|---------|---------|
| 一般軟體專案 | ISO/IEC/IEEE 29148（需求工程） |
| 企業級軟體（需品質認證） | ISO 9001 + ISO/IEC/IEEE 29148 |

**步驟 2：複製對應範本**

```bash
# 範例：一般軟體專案
cp docs/resource/02/regulations/ISO-IEC-IEEE-29148/compliance-mapping-template.md \
   <your-project>/docs/regulations/ISO-29148-compliance.md
```

**步驟 3：填寫合規性對照表**

在 `compliance-mapping-template.md` 中，逐條對應標準要求與專案文檔。

### 4.2 建立證明文件包（Evidence Package）

合規性稽核時，需提供以下文件：

```
evidence-package/
├── requirements/
│   ├── 01_SYSTEM_PRD_SR_SD.md
│   ├── 02_FRONTEND_PRD_SR_SD.md
│   └── 03_BACKEND_PRD_SR_SD.md
├── design/
│   ├── SYSTEM_ARCHITECTURE.md
│   └── MODULE_DESIGN.md
├── traceability/
│   └── TRACEABILITY_MATRIX.md
├── testing/
│   ├── TEST_PLAN.md
│   └── TEST_REPORTS.md
├── configuration-management/
│   ├── openspec/changes/archive/（所有歸檔變更）
│   └── VERSION_CONTROL.md
└── compliance/
    └── ISO-29148-compliance.md
```

### 4.3 定期合規性審計

**每季度審計檢查清單**：

```markdown
## ISO/IEC/IEEE 29148 檢查項目
- [ ] 所有 UR 都有對應的 SYS-SR
- [ ] 所有 SYS-SR 都有對應的子系統 SR
- [ ] 追溯矩陣完整無斷鏈
- [ ] 需求符合品質屬性（SMART 原則）
- [ ] 組態管理記錄完整（OpenSpec 歸檔）
- [ ] 問題解決流程有記錄
- [ ] 驗證與測試記錄完整
```

---

## 5. 與 OpenSpec 的整合（OpenSpec Integration）

### 5.1 變更管理與合規性

OpenSpec 的變更管理機制天然支援標準的「組態管理」要求：

| 標準要求 | OpenSpec 對應 |
|---------|--------------|
| 變更請求記錄 | proposal.md |
| 變更影響評估 | proposal.md § Impact |
| 變更審核記錄 | Pull Request Review |
| 變更實作記錄 | tasks.md + Git Commits |
| 變更驗證記錄 | Test Reports + 驗收記錄 |
| 變更追溯性 | spec deltas + 追溯矩陣更新 |

### 5.2 歸檔變更 = 合規性證明

每次歸檔 OpenSpec 變更時，自動產生：
- 變更歷史記錄（`changes/archive/`）
- 需求追溯更新（PRD/SR/SD 同步）
- 設計文檔更新（design.md）

這些記錄可直接用於合規性稽核。

---

## 6. 常見問題（FAQ）

### Q1: 合規性框架會增加多少開發成本？

**A**: 
- **初期投入**：建立文檔框架（約 1-2 週）
- **日常成本**：每次變更多 10-20% 時間（撰寫文檔、更新追溯）
- **長期收益**：減少返工、提升品質、便於維護、合規性認證更快

### Q3: 如何證明我們符合標準？

**A**: 使用合規性對照表（compliance-mapping）：
1. 列出標準的所有要求
2. 對應每個要求到專案文檔
3. 提供證明材料（文檔、測試報告、稽核記錄）
4. 定期審計確保一致性

### Q3: 可以部分採用框架嗎？

**A**: 可以。根據專案需求選擇：
- **最小集**：需求文檔（PRD/SR） + OpenSpec 變更管理
- **標準集**：最小集 + 追溯矩陣 + 設計文檔
- **完整集**：標準集 + 合規性文檔 + 風險管理（若需認證）

---

## 7. 延伸閱讀（Further Reading）

### 官方標準文件

- **ISO/IEC/IEEE 29148:2018**：可從 ISO 官網購買

### 推薦書籍

- **"Software Requirements" by Karl Wiegers**：需求工程經典
- **"Mastering Software Requirements" by IREB**：需求工程認證參考

### 線上資源

- **IREB (International Requirements Engineering Board)**：需求工程認證

---

## 8. 下一步（Next Steps）

### 開始使用合規性框架

1. ✅ 閱讀本索引（完成！）
2. 📋 選擇適用標準（ISO 29148）
3. 📄 複製對應範本到專案
4. ✍️ 填寫合規性對照表
5. 🔍 進行首次合規性自我審計

### 進階主題

- 🏭 品質管理系統（ISO 9001）
- 🔒 資訊安全管理（ISO 27001）（若處理敏感資料）

---

**文檔版本**: v1.0.0  
**維護團隊**: MetaFramework Core Team  
**最後審核**: 2025-12-22

