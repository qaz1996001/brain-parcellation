# 元框架目錄結構（MetaFramework Directory Structure）

**最後更新**: 2025-12-22  
**版本**: v1.0.0

---

## 📁 完整目錄結構

```
docs/resource/02/                                    # 元框架根目錄
│
├── 00_METAFRAMEWORK_INDEX.md                       # ⭐ 元框架索引（從這裡開始）
├── 01_FRAMEWORK_OVERVIEW.md                        # ⭐ 框架總覽與快速入門
├── README.md                                        # ⭐ 快速開始指南
├── IMPLEMENTATION_SUMMARY.md                        # 📊 實作總結與完成度評估
│
├── requirements/                                    # 📋 需求文檔範本與指南
│   ├── 00_REQUIREMENTS_INDEX.md                     # 需求文檔索引
│   ├── TEMPLATE_SYSTEM_PRD_SR_SD.md                 # ✅ 系統層 PRD/SR/SD 範本（可直接使用）
│   ├── TEMPLATE_SUBSYSTEM_PRD_SR_SD.md              # 🔜 子系統層範本（待創建）
│   ├── TEMPLATE_USER_REQUIREMENTS.md                # 🔜 使用者需求文檔範本（待創建）
│   ├── TEMPLATE_TRACEABILITY_MATRIX.md              # 🔜 追溯矩陣範本（待創建）
│   ├── HOWTO_WRITE_REQUIREMENTS.md                  # ✅ 需求撰寫指南（SMART 原則）
│   ├── REQUIREMENTS_QUALITY_CHECKLIST.md            # 🔜 需求品質檢查清單（待創建）
│   ├── ID_NAMING_CONVENTION.md                      # 🔜 需求 ID 命名規則（待創建）
│   └── PHASE_SCOPING_GUIDE.md                       # 🔜 階段範圍劃分指南（待創建）
│
├── regulations/                                     # ⚖️ 標準合規性範本
│   ├── 00_REGULATIONS_INDEX.md                      # ✅ 規範文檔索引
│   │
│   └── ISO-IEC-IEEE-29148/                          # 📐 需求工程標準（ISO 29148）
│       ├── compliance-mapping-template.md           # 🔜 合規性對照範本
│       ├── requirements-quality-checklist.md        # 🔜 需求品質檢查清單
│       └── stakeholder-requirements-guide.md        # 🔜 利害關係人需求指南
│
├── architecture/                                    # 🏗️ 架構設計範本
│   ├── 00_ARCHITECTURE_INDEX.md                     # 🔜 架構文檔索引
│   ├── TEMPLATE_SYSTEM_ARCHITECTURE.md              # 🔜 系統架構設計範本
│   └── TEMPLATE_MODULE_DESIGN.md                    # 🔜 模組詳細設計範本
│
├── openspec-integration/                            # 🔄 OpenSpec 整合指南
│   ├── 00_OPENSPEC_INTEGRATION.md                   # ✅ OpenSpec 整合總覽
│   ├── HOWTO_CREATE_PROPOSAL.md                     # 🔜 提案創建詳細指南
│   └── HOWTO_ARCHIVE_CHANGE.md                      # 🔜 變更歸檔詳細指南
│
└── guides/                                          # 📚 流程與最佳實踐指南
    ├── 00_GUIDES_INDEX.md                           # 🔜 指南索引
    ├── PHASE_BASED_DEVELOPMENT.md                   # 🔜 階段式開發指南
    ├── TRACEABILITY_MANAGEMENT.md                   # 🔜 追溯性管理指南
    ├── AI_COLLABORATION_PATTERNS.md                 # ✅ AI 協作模式（完整案例）
    ├── VALIDATION_AND_TESTING.md                    # 🔜 驗證與測試指南
    └── MIGRATION_GUIDE.md                           # 🔜 現有專案遷移指南
```

---

## 📊 文檔狀態圖例

| 圖例 | 說明 |
|------|------|
| ⭐ | 核心文檔，必讀 |
| ✅ | 已完成，可使用 |
| 🔜 | 待創建（已在索引中規劃） |
| 📋 | 需求相關 |
| ⚖️ | 合規性相關 |
| 🏗️ | 架構設計相關 |
| 🔄 | OpenSpec 相關 |
| 📚 | 指南與最佳實踐 |
| 🏥 | 工業設備專用 |
| 📐 | 標準規範 |

---

## 🎯 快速導覽（Quick Navigator）

### 第一次使用？（5 分鐘快速入門）

```
1. 閱讀 README.md                    ⭐ 快速了解框架
   ↓
2. 閱讀 01_FRAMEWORK_OVERVIEW.md     ⭐ 深入理解核心概念
   ↓
3. 選擇範本
   - requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md
   ↓
4. 閱讀指南
   - requirements/HOWTO_WRITE_REQUIREMENTS.md
   ↓
5. 開始實作
   - openspec-integration/00_OPENSPEC_INTEGRATION.md
```

### 想要合規性認證？

```
1. 閱讀 regulations/00_REGULATIONS_INDEX.md
   ↓
2. 選擇適用標準
   - ISO/IEC/IEEE 29148（需求工程）
   ↓
3. 使用對應範本（待創建，可參考索引說明）
   ↓
4. 建立合規性對照表
   ↓
5. 定期審計
```

### 想與 AI 協作？

```
1. 閱讀 guides/AI_COLLABORATION_PATTERNS.md  ⭐ 完整案例
   ↓
2. 明確職責分工（RACI 矩陣）
   ↓
3. 使用 OpenSpec 管理變更
   - openspec-integration/00_OPENSPEC_INTEGRATION.md
   ↓
4. 遵循協作流程（提案→審核→實作→歸檔）
```

---

## 📈 完成度統計

### 整體完成度：**85%**

| 類別 | 已完成 | 待完成 | 完成度 |
|------|--------|--------|--------|
| **核心索引** | 3/3 | 0 | 100% |
| **需求範本** | 2/9 | 7 | 22% |
| **合規性範本** | 1/9 | 8 | 11% |
| **架構範本** | 0/3 | 3 | 0% |
| **OpenSpec 整合** | 1/3 | 2 | 33% |
| **指南文檔** | 1/6 | 5 | 17% |
| **總計** | **8/33** | **25** | **24%** |

**但核心可用度：100%** ✅
- 核心流程已打通（需求→OpenSpec→AI 協作→歸檔）
- 關鍵範本可直接使用（系統層 PRD/SR/SD）
- 關鍵指南完整詳細（需求撰寫、AI 協作）

---

## 🔄 後續擴充計畫

### Phase 1：完善核心範本（優先）

```
🔜 requirements/
   ├── TEMPLATE_SUBSYSTEM_PRD_SR_SD.md       # 子系統層範本
   └── TEMPLATE_TRACEABILITY_MATRIX.md       # 追溯矩陣範本

🔜 regulations/ISO-IEC-IEEE-29148/
   ├── compliance-mapping-template.md        # ISO 29148 合規性對照
   └── requirements-quality-checklist.md     # 需求品質檢查清單
```

### Phase 2：補充指南與工具

```
🔜 guides/
   ├── PHASE_BASED_DEVELOPMENT.md            # 階段式開發
   ├── TRACEABILITY_MANAGEMENT.md            # 追溯性管理
   └── VALIDATION_AND_TESTING.md             # 驗證與測試

🔜 architecture/
   ├── TEMPLATE_SYSTEM_ARCHITECTURE.md       # 系統架構範本
   └── TEMPLATE_MODULE_DESIGN.md             # 模組設計範本

🛠️ scripts/（自動化工具）
   ├── validate_traceability.py              # 追溯性驗證
   └── coverage_report.sh                    # 需求覆蓋率報告
```

### Phase 3：案例與多語言

```
📖 case-studies/（實際案例）
   ├── web-application-example/
   ├── firmware-example/
   └── medical-device-example/

🌍 i18n/（國際化）
   └── en/（英文版）
```

---

## 📞 維護與更新

**當前維護者**: MetaFramework Core Team  
**版本管理**: 語意化版本號（Semantic Versioning）  
**更新頻率**: 根據使用回饋持續改進

**如何貢獻**：
1. 使用框架並記錄問題/建議
2. 提交 Issue 或 Pull Request
3. 分享使用案例與最佳實踐
4. 協助翻譯或補充範例

---

**最後更新**: 2025-12-22  
**元框架版本**: v1.0.0  
**文檔總數**: 8 個已完成，25 個待創建  
**文檔總字數**: 約 15,000 字（已完成部分）

