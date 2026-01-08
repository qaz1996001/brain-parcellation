# 通用產品與軟體開發元框架 v1.0.0

> **Universal Product & Software Development MetaFramework**  
> 一套通用的、標準化的文檔系統與開發流程，支援軟體、韌體、產品開發，內建國際標準合規性。

---

## 🎯 快速開始（Quick Start）

### 第一次使用？5 分鐘了解本框架

1. **閱讀概覽**：[`01_FRAMEWORK_OVERVIEW.md`](01_FRAMEWORK_OVERVIEW.md)（10 分鐘）
2. **選擇範本**：[`requirements/00_REQUIREMENTS_INDEX.md`](requirements/00_REQUIREMENTS_INDEX.md)
3. **創建需求文檔**：複製 `TEMPLATE_SYSTEM_PRD_SR_SD.md` 到專案
4. **開始第一個變更**：使用 OpenSpec 創建提案（`/openspec-proposal`）
5. **與 AI 協作**：遵循 [`guides/AI_COLLABORATION_PATTERNS.md`](guides/AI_COLLABORATION_PATTERNS.md)

---

## 📖 核心概念（Core Concepts）

### 什麼是本元框架？

本框架是一套**可套用到任何程式碼倉庫、韌體、新產品開發的通用文檔系統**，包含：

- ✅ **標準化需求文檔範本**（PRD/SR/SD）
- ✅ **OpenSpec 變更管理整合**（提案→實作→歸檔）
- ✅ **完整追溯性管理**（需求→設計→實作→測試）
- ✅ **國際標準合規性支援**（ISO 29148）
- ✅ **人機協作模式**（明確的人類與 AI 職責分工）
- ✅ **階段式開發支援**（Phase 1/2/3 漸進交付）

### 核心架構

```
┌────────────────────────────────────────────────────┐
│   文檔層級（Documentation Layers）                  │
├────────────────────────────────────────────────────┤
│  Layer 1: 系統層（System Level）                    │
│  - SYSTEM_PRD_SR_SD.md                             │
│  - 產品願景、系統邊界、系統需求                      │
├────────────────────────────────────────────────────┤
│  Layer 2: 子系統層（Subsystem Level）               │
│  - FRONTEND_PRD_SR_SD.md                           │
│  - BACKEND_PRD_SR_SD.md                            │
│  - FIRMWARE_PRD_SR_SD.md                           │
├────────────────────────────────────────────────────┤
│  Layer 3: 模組層（Module Level）                    │
│  - API 規格、資料庫設計、UI 元件設計                 │
└────────────────────────────────────────────────────┘
          ↕ (雙向追溯性 Traceability)
┌────────────────────────────────────────────────────┐
│   變更管理（Change Management）                     │
├────────────────────────────────────────────────────┤
│  OpenSpec 提案 → 審核 → 實作 → 歸檔 → 同步文檔     │
└────────────────────────────────────────────────────┘
```

---

## 📁 文檔結構（Documentation Structure）

```
docs/resource/meta-framework/                          # 元框架根目錄
├── 00_METAFRAMEWORK_INDEX.md             # 元框架索引（從這裡開始）
├── 01_FRAMEWORK_OVERVIEW.md              # 框架總覽與快速入門
├── README.md                              # 本文件
│
├── requirements/                          # 需求文檔範本與指南
│   ├── 00_REQUIREMENTS_INDEX.md           # 需求文檔索引
│   ├── TEMPLATE_SYSTEM_PRD_SR_SD.md       # 系統層 PRD/SR/SD 範本
│   ├── TEMPLATE_SUBSYSTEM_PRD_SR_SD.md    # 子系統層範本（待創建）
│   ├── TEMPLATE_TRACEABILITY_MATRIX.md    # 追溯矩陣範本（待創建）
│   └── HOWTO_WRITE_REQUIREMENTS.md        # 需求撰寫指南
│
├── regulations/                           # 標準合規性範本
│   ├── 00_REGULATIONS_INDEX.md            # 規範文檔索引
│   ├── ISO-IEC-IEEE-29148/                # 需求工程標準
│   │   ├── compliance-mapping-template.md
│   │   ├── requirements-quality-checklist.md
│   │   └── stakeholder-requirements-guide.md
│
├── architecture/                          # 架構設計範本（待創建）
│   ├── 00_ARCHITECTURE_INDEX.md
│   ├── TEMPLATE_SYSTEM_ARCHITECTURE.md
│   └── TEMPLATE_MODULE_DESIGN.md
│
├── openspec-integration/                  # OpenSpec 整合指南
│   ├── 00_OPENSPEC_INTEGRATION.md         # OpenSpec 整合總覽
│   ├── HOWTO_CREATE_PROPOSAL.md           # 提案創建指南（待創建）
│   └── HOWTO_ARCHIVE_CHANGE.md            # 變更歸檔指南（待創建）
│
└── guides/                                # 流程與最佳實踐
    ├── 00_GUIDES_INDEX.md                 # 指南索引（待創建）
    ├── PHASE_BASED_DEVELOPMENT.md         # 階段式開發指南（待創建）
    ├── TRACEABILITY_MANAGEMENT.md         # 追溯性管理指南（待創建）
    ├── AI_COLLABORATION_PATTERNS.md       # AI 協作模式（已完成）
    └── VALIDATION_AND_TESTING.md          # 驗證與測試指南（待創建）
```

---

## 🚀 使用場景（Use Cases）

### 場景 1：啟動全新軟體專案

```bash
# 1. 複製範本到專案
cp docs/resource/meta-framework/requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md \
   <your-project>/docs/requirements/01_SYSTEM_PRD_SR_SD.md

# 2. 填寫需求文檔
# - 專案名稱、版本、狀態
# - 使用者需求（UR-xxx）
# - 系統需求（SYS-SR-xxx）

# 3. 初始化 OpenSpec
cd <your-project>
openspec init

# 4. 創建第一個變更提案
# 使用 AI：/openspec-proposal "setup-project-infrastructure"
```

### 場景 2：韌體開發專案

```bash
# 1. 複製範本
cp docs/resource/meta-framework/requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md \
   <firmware-project>/docs/requirements/01_SYSTEM_PRD_SR_SD.md

# 2. 創建韌體子系統文檔
# - FIRMWARE_PRD_SR_SD.md（硬體介面、嵌入式邏輯）
# - HARDWARE_INTERFACE_SPEC.md（GPIO、I2C、SPI 等）

# 3. 建立追溯矩陣
# - 需求 → 韌體模組 → 測試案例

# 4. 使用 OpenSpec 管理韌體變更
```

### 場景 3：為現有專案補文檔

```bash
# 1. 逆向工程：從程式碼提取隱含需求
# 2. 撰寫 PRD/SR/SD 文檔（先覆蓋已完成部分）
# 3. 建立追溯矩陣（至少覆蓋核心功能）
# 4. 導入 OpenSpec 管理後續變更
# 5. 漸進式補齊文檔（每次變更時同步更新）
```

---

## 🤝 人機協作模式（Human-AI Collaboration）

### 職責分工

| 階段 | 人類負責 | AI 負責 |
|------|---------|--------|
| **需求定義** | - 訪談利害關係人<br>- 定義產品願景<br>- 編寫 UR-xxx | - 協助結構化需求<br>- 檢查需求完整性<br>- 產生需求 ID 與追溯表 |
| **系統設計** | - 審核技術方案<br>- 決策關鍵架構<br>- 評估技術風險 | - 撰寫設計文檔<br>- 產生架構圖<br>- 評估技術選項 |
| **變更提案** | - 審核提案商業價值<br>- 評估變更影響<br>- 批准/拒絕提案 | - 撰寫 proposal.md<br>- 產生 tasks.md<br>- 撰寫 spec deltas |
| **開發實作** | - 驗收功能<br>- 整合測試<br>- 程式碼審查（關鍵部分） | - 依據 tasks.md 實作<br>- 撰寫單元測試<br>- 更新技術文檔 |
| **變更歸檔** | - 最終審核<br>- 部署決策<br>- 驗證文檔一致性 | - 歸檔變更<br>- 更新 specs/<br>- 更新 PRD/SR/SD<br>- 更新追溯矩陣 |

詳見：[`guides/AI_COLLABORATION_PATTERNS.md`](guides/AI_COLLABORATION_PATTERNS.md)

---

## 📐 國際標準支援（Standards Compliance）

### ISO/IEC/IEEE 29148:2018（需求工程）

- ✅ Stakeholder Requirements（利害關係人需求）
- ✅ System Requirements Specification（系統需求規格）
- ✅ Software Requirements Specification（軟體需求規格）
- ✅ Requirements Traceability（需求追溯性）
- ✅ Requirements Quality Attributes（需求品質屬性）

詳見：[`regulations/00_REGULATIONS_INDEX.md`](regulations/00_REGULATIONS_INDEX.md)

---

## 📚 核心文檔導覽（Document Navigator）

### 新手必讀

| 文檔 | 用途 | 閱讀時間 |
|------|------|---------|
| [`00_METAFRAMEWORK_INDEX.md`](00_METAFRAMEWORK_INDEX.md) | 元框架索引，了解整體結構 | 5 分鐘 |
| [`01_FRAMEWORK_OVERVIEW.md`](01_FRAMEWORK_OVERVIEW.md) | 框架總覽，核心概念與使用流程 | 15 分鐘 |
| [`requirements/HOWTO_WRITE_REQUIREMENTS.md`](requirements/HOWTO_WRITE_REQUIREMENTS.md) | 需求撰寫指南（SMART 原則、範例） | 20 分鐘 |

### 實作指南

| 文檔 | 用途 |
|------|------|
| [`requirements/00_REQUIREMENTS_INDEX.md`](requirements/00_REQUIREMENTS_INDEX.md) | 選擇合適的需求文檔範本 |
| [`requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md`](requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md) | 系統層 PRD/SR/SD 範本 |
| [`openspec-integration/00_OPENSPEC_INTEGRATION.md`](openspec-integration/00_OPENSPEC_INTEGRATION.md) | OpenSpec 整合指南 |
| [`guides/AI_COLLABORATION_PATTERNS.md`](guides/AI_COLLABORATION_PATTERNS.md) | AI 協作模式與最佳實踐 |

### 合規性指南

| 文檔 | 用途 |
|------|------|
| [`regulations/00_REGULATIONS_INDEX.md`](regulations/00_REGULATIONS_INDEX.md) | 選擇適用標準，了解合規性要求 |
| [`regulations/ISO-IEC-IEEE-29148/`](regulations/ISO-IEC-IEEE-29148/) | 需求工程標準範本 |

---

## ✅ 成功標準（Success Criteria）

### 文檔品質

- ✅ 所有需求都有唯一 ID 與版本號
- ✅ 需求符合 SMART 原則
- ✅ 追溯矩陣完整無斷鏈
- ✅ 設計文檔與實作一致

### 流程效率

- ✅ 需求變更到實作週期 ≤ 2 週
- ✅ OpenSpec 提案通過率 > 80%
- ✅ AI 實作正確率 > 90%

### 合規性

- ✅ 符合選定的國際標準要求
- ✅ 定期合規性審計無重大缺失
- ✅ 文檔可追溯、可稽核

### 團隊協作

- ✅ 人類與 AI 的職責清晰
- ✅ 文檔更新及時（變更後 24 小時內）
- ✅ 新成員 ≤ 1 週熟悉框架

---

## 🔄 框架版本與維護（Version and Maintenance）

### 當前版本

- **版本號**：v1.0.0
- **發布日期**：2025-12-22
- **狀態**：Stable

### 版本歷史

| 版本 | 日期 | 變更摘要 |
|------|------|---------|
| v1.0.0 | 2025-12-22 | 初始版本，核心框架完成 |

### 更新計畫

- [ ] 補充子系統範本（`TEMPLATE_SUBSYSTEM_PRD_SR_SD.md`）
- [ ] 補充追溯矩陣範本（`TEMPLATE_TRACEABILITY_MATRIX.md`）
- [ ] 補充架構設計範本（`architecture/`）
- [ ] 補充階段式開發指南（`guides/PHASE_BASED_DEVELOPMENT.md`）
- [ ] 補充驗證與測試指南（`guides/VALIDATION_AND_TESTING.md`）
- [ ] 增加實際案例研究（Case Studies）

---

## 🙋 常見問題（FAQ）

### Q1: 這個框架適合我的專案嗎？

**A**: 本框架適用於：
- ✅ 中大型軟體專案（需結構化管理）
- ✅ 韌體與嵌入式系統開發
- ✅ 需要合規性證明的專案（醫療、航空、汽車等）
- ✅ 跨團隊協作專案（需明確文檔與介面）
- ✅ 人機協作開發（與 AI 共同開發）

不太適合：
- ❌ 極簡單的腳本或工具（過度工程）
- ❌ 快速原型專案（除非後續要產品化）

### Q2: 需要全部採用嗎？

**A**: 不需要。可根據需求選擇：
- **最小集**：需求文檔（PRD/SR） + OpenSpec 變更管理
- **標準集**：最小集 + 追溯矩陣 + 設計文檔
- **完整集**：標準集 + 合規性文檔 + 風險管理

### Q3: 與其他框架（如 Agile, Scrum）相容嗎？

**A**: 完全相容。本框架聚焦於「文檔結構與內容」，不限制開發流程：
- **Scrum**：Sprint 可對應 Phase，User Story 對應 UR
- **Kanban**：OpenSpec 提案對應 Kanban 卡片
- **Waterfall**：直接對應各階段文檔要求

### Q4: 如何貢獻或回饋？

**A**: 歡迎：
- 提出問題或建議
- 分享使用經驗與案例
- 貢獻範本或指南
- 提交改進 Pull Request

---

## 📞 支援與貢獻（Support and Contribution）

### 問題回報

如發現框架缺陷或不清楚之處：
1. 檢查 FAQ
2. 搜尋現有議題
3. 創建新議題，描述問題與建議

### 貢獻指南

歡迎貢獻：
- 新的範本文檔
- 案例研究
- 最佳實踐
- 工具腳本
- 文檔翻譯

提交前請確保：
- 遵循現有文檔格式
- 提供清晰的說明與範例
- 通過內部審查

---

## 📄 授權（License）

本元框架以 **[待定授權]** 釋出，可自由用於商業或非商業專案。

---

**元框架版本**: v1.0.0  
**維護團隊**: MetaFramework Core Team  
**最後更新**: 2025-12-22  
**文檔語言**: 繁體中文（Traditional Chinese）

---

## 🎉 開始使用（Get Started）

1. ✅ 閱讀本 README（完成！）
2. 📖 閱讀 [`01_FRAMEWORK_OVERVIEW.md`](01_FRAMEWORK_OVERVIEW.md)
3. 📝 選擇範本並創建第一份需求文檔
4. 🚀 使用 OpenSpec 創建第一個變更提案
5. 🤖 與 AI 協作，實現高效開發

**祝你的專案成功！ 🎊**

