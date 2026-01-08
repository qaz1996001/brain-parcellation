# IEC 62304 合規性對照範本（Compliance Mapping Template）

**文件 ID**: IEC62304-COMPLIANCE-001
**標題**: IEC 62304 醫療設備軟體生命週期合規性對照表
**版本**: v1.0.0
**狀態**: Template
**建立日期**: YYYY-MM-DD
**最後更新**: YYYY-MM-DD
**作者**: [待填]
**審核人**: [品質負責人 / 法規負責人 待填]

---

## 1. 文件目的（Purpose）

本範本用於：
- 建立 IEC 62304 標準要求與專案實作的對照關係
- 追蹤合規性實施進度
- 提供稽核證據索引
- 支援法規申報與認證

---

## 2. IEC 62304 概述（Overview）

### 2.1 標準適用範圍

IEC 62304:2006+AMD1:2015 Medical device software — Software life cycle processes

**適用對象**：
- 醫療設備軟體（作為醫療設備或醫療設備的一部分）
- 軟體生命週期管理流程
- 軟體維護與除役

**不適用對象**：
- 非醫療用途軟體
- 純硬體醫療設備（無軟體組件）

### 2.2 軟體安全分級（Safety Classification）

| 安全級別 | 定義 | 風險等級 | 文檔要求 |
|---------|------|---------|---------|
| **Class A** | 不可能造成傷害 | 低 | 基本文檔 |
| **Class B** | 可能造成非嚴重傷害 | 中 | 標準文檔 |
| **Class C** | 可能造成死亡或嚴重傷害 | 高 | 完整文檔 |

**本專案軟體安全級別**：[Class A / Class B / Class C] — [填寫理由]

---

## 3. 生命週期流程對照（Process Mapping）

### 3.1 軟體開發規劃（Software Development Planning）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.1.1 | 軟體開發計畫 | `DEVELOPMENT_PLAN.md` | ✅ 完成 | `docs/planning/` | - |
| 5.1.2 | 保持開發計畫更新 | 版本控制系統 | 🔄 持續 | Git commit history | - |
| 5.1.3 | 軟體開發標準、方法和工具規劃 | `DEVELOPMENT_STANDARDS.md` | ✅ 完成 | `docs/standards/` | - |
| 5.1.4 | 軟體整合與整合測試規劃 | `INTEGRATION_PLAN.md` | ⏳ 進行中 | `docs/testing/` | Phase 2 |
| 5.1.5 | 軟體驗證規劃 | `VERIFICATION_PLAN.md` | ⏳ 進行中 | `docs/verification/` | Phase 2 |
| 5.1.6 | 軟體風險管理規劃 | `RISK_MANAGEMENT_PLAN.md` | ✅ 完成 | `docs/risk/` | 配合 ISO 14971 |
| 5.1.7 | 軟體配置管理規劃 | `CONFIGURATION_MANAGEMENT.md` | ✅ 完成 | `docs/cm/` | - |
| 5.1.8 | 軟體問題解決規劃 | `ISSUE_RESOLUTION.md` | ✅ 完成 | `docs/quality/` | - |
| 5.1.9 | 文檔化活動 | 本文檔 | ✅ 完成 | 此文件 | - |

---

### 3.2 軟體需求分析（Software Requirements Analysis）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.2.1 | 定義並文檔化軟體需求 | `SRS_TEMPLATE.md` | ✅ 完成 | `requirements/` | 使用元框架範本 |
| 5.2.2 | 軟體需求內容 | SRS 文檔 | ✅ 完成 | 各專案 SRS | - |
| 5.2.3 | 包含風險控制措施 | 風險控制追溯矩陣 | ✅ 完成 | `RISK_TRACEABILITY.md` | - |
| 5.2.4 | 重新評估醫療設備風險分析 | 風險管理報告 | 🔄 持續 | `risk/RISK_REPORT.md` | 每次需求變更 |
| 5.2.5 | 更新系統需求 | 系統需求文檔 | 🔄 持續 | `SYSTEM_REQUIREMENTS.md` | - |
| 5.2.6 | 驗證軟體需求 | 需求審查記錄 | ✅ 完成 | `reviews/SR_REVIEW.md` | - |

---

### 3.3 軟體架構設計（Software Architectural Design）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.3.1 | 將軟體需求轉換為架構 | `ARCHITECTURE.md` | ✅ 完成 | `architecture/` | - |
| 5.3.2 | 開發軟體架構 | 架構設計文檔 | ✅ 完成 | `SYSTEM_ARCHITECTURE.md` | - |
| 5.3.3 | 支援正確實施的架構 | 架構決策記錄 (ADR) | ✅ 完成 | `architecture/decisions/` | - |
| 5.3.4 | 隔離 SOUP | SOUP 管理計畫 | ✅ 完成 | `SOUP_MANAGEMENT.md` | - |
| 5.3.5 | 識別可分離的軟體項目 | 模組分解文檔 | ✅ 完成 | `MODULE_DESIGN.md` | - |
| 5.3.6 | 驗證軟體架構 | 架構審查記錄 | ✅ 完成 | `reviews/ARCH_REVIEW.md` | - |

---

### 3.4 軟體詳細設計（Software Detailed Design）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.4.1 | 細化軟體架構為軟體單元 | `MODULE_DESIGN.md` | ✅ 完成 | `design/modules/` | - |
| 5.4.2 | 開發詳細設計 | 詳細設計文檔 | ✅ 完成 | `SDD_TEMPLATE.md` | - |
| 5.4.3 | 詳細設計的介面 | API 規格文檔 | ✅ 完成 | `API_SPECIFICATION.md` | - |
| 5.4.4 | 驗證詳細設計 | 設計審查記錄 | ✅ 完成 | `reviews/DD_REVIEW.md` | - |

---

### 3.5 軟體單元實施與驗證（Software Unit Implementation and Verification）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.5.1 | 實施每個軟體單元 | 原始碼 | ✅ 完成 | `src/`, `backend/`, `frontend/` | - |
| 5.5.2 | 建立驗收準則 | 單元測試規格 | ✅ 完成 | `tests/unit/` | - |
| 5.5.3 | 單元驗證 | 單元測試執行報告 | 🔄 持續 | CI/CD 測試報告 | 自動化 |
| 5.5.4 | 單元驗證的附加方法 | 程式碼審查記錄 | 🔄 持續 | `reviews/CODE_REVIEW.md` | PR 審查 |
| 5.5.5 | 軟體單元驗證記錄 | 測試覆蓋率報告 | 🔄 持續 | `coverage/` | >70% 目標 |

---

### 3.6 軟體整合與整合測試（Software Integration and Integration Testing）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.6.1 | 整合軟體單元 | 整合計畫 | ✅ 完成 | `INTEGRATION_PLAN.md` | - |
| 5.6.2 | 驗證軟體整合 | 整合測試規格 | ✅ 完成 | `tests/integration/` | - |
| 5.6.3 | 整合測試程序 | 測試案例文檔 | ✅ 完成 | `STP_TEMPLATE.md` | - |
| 5.6.4 | 整合測試記錄 | 測試執行報告 | 🔄 持續 | CI/CD 測試報告 | - |
| 5.6.5 | 測試 SOUP 整合 | SOUP 整合測試 | ✅ 完成 | `tests/soup/` | - |
| 5.6.6 | 整合測試的附加方法 | 手動測試記錄 | 🔄 持續 | `manual_tests/` | 關鍵功能 |
| 5.6.7 | 重新評估風險 | 風險更新報告 | 🔄 持續 | `risk/UPDATES.md` | 每次整合 |

---

### 3.7 軟體系統測試（Software System Testing）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.7.1 | 建立系統測試 | 系統測試計畫 | ✅ 完成 | `SYSTEM_TEST_PLAN.md` | - |
| 5.7.2 | 使用風險管理 | 風險基測試矩陣 | ✅ 完成 | `RISK_BASED_TESTING.md` | - |
| 5.7.3 | 系統測試程序 | 測試案例文檔 | ✅ 完成 | `tests/system/` | - |
| 5.7.4 | 系統測試記錄 | 測試執行報告 | ⏳ 進行中 | `test_reports/system/` | Phase 3 |
| 5.7.5 | 重新評估風險 | 最終風險報告 | ⏳ 計畫中 | `risk/FINAL_REPORT.md` | Phase 4 |

---

### 3.8 軟體發布（Software Release）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.8.1 | 確保整合與測試完成 | 發布檢查清單 | ⏳ 計畫中 | `RELEASE_CHECKLIST.md` | Phase 4 |
| 5.8.2 | 文檔化已知遺留異常 | 已知問題文檔 | 🔄 持續 | `KNOWN_ISSUES.md` | - |
| 5.8.3 | 評估已知異常的可接受性 | 異常評估報告 | 🔄 持續 | `risk/ANOMALY_ASSESSMENT.md` | - |
| 5.8.4 | 文檔化發布版本 | 版本發布說明 | 🔄 持續 | `CHANGELOG.md` | - |
| 5.8.5 | 創建可歸檔的記錄 | 發布檔案包 | ⏳ 計畫中 | `releases/` | - |
| 5.8.6 | 確保活動已完成 | 完成確認清單 | ⏳ 計畫中 | `COMPLETION_CHECKLIST.md` | - |
| 5.8.7 | 批准發布 | 發布批准記錄 | ⏳ 計畫中 | `approvals/RELEASE_APPROVAL.md` | QA + 管理層 |
| 5.8.8 | 交付發布軟體 | 發布交付記錄 | ⏳ 計畫中 | `delivery/` | - |

---

## 4. 支援流程對照（Supporting Processes）

### 4.1 軟體配置管理（Software Configuration Management）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.1.7, 8.1 | 配置管理計畫 | `CONFIGURATION_MANAGEMENT.md` | ✅ 完成 | `docs/cm/` | - |
| 8.1.1 | 配置項目識別 | CI 清單 | ✅ 完成 | `CM_ITEMS.md` | - |
| 8.1.2 | 配置項目的基準 | Git tags, releases | ✅ 完成 | Git 倉庫 | - |
| 8.1.3 | 配置項目變更控制 | Git 分支策略 | ✅ 完成 | `BRANCHING_STRATEGY.md` | - |

---

### 4.2 問題解決（Problem Resolution）

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.1.8, 9.1 | 問題解決流程 | `ISSUE_RESOLUTION.md` | ✅ 完成 | `docs/quality/` | - |
| 9.1 | 準備問題報告 | Issue tracking system | ✅ 完成 | GitHub Issues | - |
| 9.2 | 建立問題評估流程 | 問題分類與優先級 | ✅ 完成 | `ISSUE_TRIAGE.md` | - |
| 9.3 | 調查問題 | 根本原因分析記錄 | 🔄 持續 | `investigations/` | - |
| 9.4 | 向監管機構通報 | 報告流程文檔 | ✅ 完成 | `REGULATORY_REPORTING.md` | 若適用 |
| 9.5 | 使用配置管理 | Git commit 追溯 | ✅ 完成 | Git 歷史 | - |
| 9.6 | 問題解決記錄 | 解決方案文檔 | 🔄 持續 | `resolutions/` | - |
| 9.7 | 分析問題趨勢 | 品質趨勢報告 | 🔄 季度 | `quality/TRENDS.md` | - |
| 9.8 | 驗證問題解決 | 驗證測試記錄 | 🔄 持續 | `tests/fixes/` | - |
| 9.9 | 測試文檔更新 | 回歸測試更新 | 🔄 持續 | `tests/regression/` | - |

---

## 5. 風險管理整合（Risk Management Integration）

### 5.1 IEC 62304 + ISO 14971 整合

| 活動 | IEC 62304 | ISO 14971 | 整合文檔 | 實施狀態 |
|------|-----------|-----------|---------|---------|
| 風險分析 | 5.1.6, 5.2.3, 5.2.4 | 4.3, 4.4 | `RISK_ANALYSIS.md` | ✅ 完成 |
| 風險控制 | 5.2.3 | 6.2, 6.3, 6.4 | `RISK_CONTROLS.md` | ✅ 完成 |
| 風險評估 | 5.6.7, 5.7.5 | 4.5 | `RISK_EVALUATION.md` | 🔄 持續 |
| 殘餘風險 | 5.8.3 | 6.5 | `RESIDUAL_RISK.md` | ⏳ Phase 3 |
| 風險追溯 | 全流程 | 全流程 | `RISK_TRACEABILITY.md` | ✅ 完成 |

---

## 6. SOUP 管理（SOUP Management）

### 6.1 SOUP 識別與評估

| IEC 62304 條款 | 要求描述 | 實作文檔 | 實施狀態 | 證據位置 | 備註 |
|---------------|---------|---------|---------|---------|------|
| 5.3.4 | 隔離 SOUP | SOUP 隔離設計 | ✅ 完成 | `architecture/SOUP_ISOLATION.md` | - |
| 7.1.1 | 建立軟體開發需求 | SOUP 需求文檔 | ✅ 完成 | `SOUP_REQUIREMENTS.md` | - |
| 7.1.2 | 識別 SOUP | SOUP 清單 | ✅ 完成 | `SOUP_LIST.md` | - |
| 7.1.3 | 記錄 SOUP 異常序列 | 已知問題記錄 | 🔄 持續 | `soup/KNOWN_ISSUES.md` | - |

---

## 7. 合規性狀態追蹤（Compliance Status）

### 7.1 整體完成度

| 階段 | 條款數量 | 已完成 | 進行中 | 計畫中 | 完成率 |
|------|---------|--------|--------|--------|--------|
| 軟體開發規劃 | 9 | 7 | 2 | 0 | 78% |
| 軟體需求分析 | 6 | 4 | 2 | 0 | 67% |
| 軟體架構設計 | 6 | 6 | 0 | 0 | 100% |
| 軟體詳細設計 | 4 | 4 | 0 | 0 | 100% |
| 軟體單元實施 | 5 | 2 | 3 | 0 | 40% |
| 軟體整合測試 | 7 | 3 | 4 | 0 | 43% |
| 軟體系統測試 | 5 | 3 | 1 | 1 | 60% |
| 軟體發布 | 8 | 0 | 2 | 6 | 0% |
| 配置管理 | 4 | 4 | 0 | 0 | 100% |
| 問題解決 | 9 | 5 | 4 | 0 | 56% |
| **總計** | **63** | **38** | **18** | **7** | **60%** |

### 7.2 按安全級別的合規性要求

#### Class A 要求（最低）
- ✅ 基本文檔化
- ✅ 基本測試
- ✅ 基本配置管理

#### Class B 要求（標準）
- ✅ 完整需求追溯
- ✅ 架構與設計文檔
- 🔄 整合測試與系統測試
- ✅ 風險管理整合

#### Class C 要求（最嚴格）
- ✅ 完整生命週期文檔
- 🔄 全面測試與驗證
- ✅ 詳細風險分析
- ⏳ 完整發布流程
- ⏳ 上市後監控計畫

---

## 8. 稽核證據索引（Audit Evidence Index）

### 8.1 文檔證據

| 類別 | 文檔名稱 | 位置 | IEC 62304 條款 |
|------|---------|------|---------------|
| 規劃 | 軟體開發計畫 | `docs/planning/DEVELOPMENT_PLAN.md` | 5.1.1 |
| 需求 | 軟體需求規格 | `requirements/SRS_*.md` | 5.2.1, 5.2.2 |
| 設計 | 架構設計文檔 | `architecture/SYSTEM_ARCHITECTURE.md` | 5.3.2 |
| 設計 | 詳細設計文檔 | `design/SDD_*.md` | 5.4.2 |
| 測試 | 測試計畫 | `tests/STP_*.md` | 5.6.3, 5.7.3 |
| 風險 | 風險管理檔案 | `risk/RISK_MANAGEMENT_FILE.md` | 5.1.6 |
| SOUP | SOUP 管理計畫 | `soup/SOUP_MANAGEMENT.md` | 7.1 |
| 配置 | 配置管理計畫 | `docs/cm/CONFIGURATION_MANAGEMENT.md` | 8.1 |

### 8.2 記錄證據

| 類別 | 記錄類型 | 位置 | IEC 62304 條款 |
|------|---------|------|---------------|
| 審查 | 需求審查記錄 | `reviews/SR_REVIEW_*.md` | 5.2.6 |
| 審查 | 架構審查記錄 | `reviews/ARCH_REVIEW_*.md` | 5.3.6 |
| 審查 | 設計審查記錄 | `reviews/DD_REVIEW_*.md` | 5.4.4 |
| 測試 | 單元測試報告 | `test_reports/unit/` | 5.5.5 |
| 測試 | 整合測試報告 | `test_reports/integration/` | 5.6.4 |
| 測試 | 系統測試報告 | `test_reports/system/` | 5.7.4 |
| 問題 | 問題報告 | GitHub Issues | 9.1 |
| 發布 | 發布批准記錄 | `approvals/RELEASE_*.md` | 5.8.7 |

---

## 9. 差距分析（Gap Analysis）

### 9.1 待完成項目

#### 高優先級（Phase 2 內完成）
- [ ] 整合測試完整執行（5.6.4）
- [ ] 系統測試計畫執行（5.7.4）
- [ ] 單元測試覆蓋率達標（5.5.5）

#### 中優先級（Phase 3 內完成）
- [ ] 發布檢查清單建立（5.8.1）
- [ ] 已知問題評估流程（5.8.3）
- [ ] 上市後監控計畫（若 Class C）

#### 低優先級（Phase 4 內完成）
- [ ] 完整發布流程執行（5.8）
- [ ] 歸檔記錄創建（5.8.5）
- [ ] 監管機構通報流程測試（9.4）

### 9.2 改進建議

1. **自動化強化**：
   - 增加自動化測試覆蓋率（目標 >80%）
   - 建立 CI/CD 自動合規性檢查
   - 自動生成測試報告與追溯矩陣

2. **文檔管理**：
   - 使用 OpenSpec 管理所有合規性文檔變更
   - 建立文檔版本控制與審批流程
   - 定期審查文檔完整性

3. **培訓與教育**：
   - 開發團隊 IEC 62304 培訓
   - 建立內部合規性最佳實踐指南
   - 定期合規性審計演練

---

## 10. 審查與批准（Review and Approval）

### 10.1 審查記錄

| 審查日期 | 審查人 | 審查範圍 | 發現問題 | 解決狀態 |
|---------|--------|---------|---------|---------|
| YYYY-MM-DD | [姓名] | 完整文檔 | [問題清單] | [狀態] |

### 10.2 批准記錄

| 角色 | 姓名 | 簽名 | 日期 |
|------|------|------|------|
| 專案經理 | [待填] | [待填] | YYYY-MM-DD |
| 品質負責人 | [待填] | [待填] | YYYY-MM-DD |
| 法規負責人 | [待填] | [待填] | YYYY-MM-DD |

---

## 11. 附錄（Appendix）

### 11.1 縮寫與術語

| 術語 | 全稱 | 說明 |
|------|------|------|
| IEC 62304 | International Electrotechnical Commission 62304 | 醫療設備軟體生命週期流程 |
| SOUP | Software of Unknown Provenance | 未知來源軟體（第三方庫） |
| SRS | Software Requirements Specification | 軟體需求規格 |
| SDD | Software Design Document | 軟體設計文檔 |
| STP | Software Test Plan | 軟體測試計畫 |
| ADR | Architecture Decision Record | 架構決策記錄 |

### 11.2 參考文件

- IEC 62304:2006+AMD1:2015 — Medical device software — Software life cycle processes
- ISO 14971:2019 — Medical devices — Application of risk management to medical devices
- ISO 13485:2016 — Medical devices — Quality management systems
- FDA Software Validation Guidance
- IMDRF Software as a Medical Device (SaMD) Guidance

---

**文檔版本**: v1.0.0
**維護人**: [姓名]
**最後審核**: YYYY-MM-DD
**下次審核**: YYYY-MM-DD（建議每季度或重大變更時）
