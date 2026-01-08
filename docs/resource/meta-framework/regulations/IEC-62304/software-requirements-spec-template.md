# IEC 62304 軟體需求規格範本（Software Requirements Specification Template）

**文件 ID**: IEC62304-SRS-001
**標題**: [醫療設備名稱] — 軟體需求規格
**版本**: v1.0.0
**狀態**: Draft
**IEC 62304 安全級別**: [Class A / Class B / Class C]
**建立日期**: YYYY-MM-DD
**最後更新**: YYYY-MM-DD
**作者**: [待填]
**審核人**: [品質負責人 / 醫療顧問 / 法規負責人 待填]

---

## 變更歷史（Change History）

| 版本 | 日期 | 修改者 | 變更摘要 | IEC 62304 條款 |
|------|------|--------|---------|---------------|
| v0.1 | YYYY-MM-DD | [姓名] | 初始草稿 | 5.2.1 |
| v1.0 | YYYY-MM-DD | [姓名] | 首次正式版本 | 5.2.6 |

---

## 1. 範圍與目的（Scope and Purpose）

### 1.1 專案背景

**醫療用途**：[此軟體作為醫療設備或醫療設備的一部分，用於 ...]

**適用標準**：
- IEC 62304:2006+AMD1:2015 — Medical device software life cycle processes
- ISO 14971:2019 — Risk management for medical devices
- ISO 13485:2016 — Quality management systems for medical devices
- [其他適用標準]

### 1.2 軟體安全分級（Safety Classification）

**安全級別**：[Class A / Class B / Class C]

**分級理由**：
[簡述為何此軟體被分類為此級別，參考 `software-safety-classification.md`]

**分級批准**：

| 角色 | 姓名 | 簽名 | 日期 |
|------|------|------|------|
| 風險管理負責人 | [待填] | [待填] | YYYY-MM-DD |

### 1.3 系統級需求追溯

**上層需求來源**：
- 使用者需求規格（User Requirements Specification, URS）：`[文檔位置]`
- 系統需求規格（System Requirements Specification, SysRS）：`[文檔位置]`
- 風險管理檔案（Risk Management File）：`[文檔位置]`

---

## 2. 利害關係人需求（Stakeholder Requirements）

### 2.1 使用者角色

| 角色 | 說明 | 醫療資格 | 使用環境 | 安全考量 |
|------|------|---------|---------|---------|
| 臨床醫師 | 診斷與治療決策 | 執照醫師 | 醫院診療室 | 高風險決策 |
| 護理人員 | 監測與操作設備 | 註冊護士 | 病房/ICU | 持續監測 |
| 患者 | [若患者直接使用] | 無醫療背景 | 家庭環境 | 自我管理風險 |
| 維護人員 | 設備維護與校準 | 生物醫學工程師 | 維護室 | 設備安全 |

### 2.2 使用環境（Intended Use Environment）

**臨床環境**：[醫院 / 診所 / 居家 / 急救]

**環境條件**：
- 溫度範圍：[X°C ~ Y°C]
- 濕度範圍：[X% ~ Y%]
- 電源：[AC/DC, 電壓範圍]
- 電磁環境：[醫療級 EMC 要求]

**使用限制**：
- [ ] 需要專業醫療人員操作
- [ ] 需要特定訓練
- [ ] 單次使用 / 可重複使用
- [ ] 滅菌要求

---

## 3. 功能性軟體需求（Functional Software Requirements）

> **RFC 2119 關鍵詞說明**：
> **SHALL** = 強制要求
> **SHOULD** = 強烈建議
> **MAY** = 可選項
> **MUST NOT** = 禁止

### 3.1 核心功能需求

#### SR-FUNC-001: [核心功能名稱]

**描述**：軟體 **SHALL** [可驗證的行為描述，使用主動語態]

**系統需求追溯**：
- Traces to: SysR-001, UR-005
- Risk Control for: HS-012 (ISO 14971 危害場景)

**驗證方式**：Test
**優先級**：Must
**安全影響**：高/中/低

**驗收標準（Acceptance Criteria）**：

##### Scenario 1: 正常操作（Normal Operation）
- **GIVEN** [系統處於正常狀態]
- **WHEN** [使用者執行操作 X]
- **THEN** [系統 SHALL 產生結果 Y]
- **AND** [系統 SHALL 在 Z 秒內完成]

##### Scenario 2: 異常處理（Abnormal Condition）
- **GIVEN** [系統處於異常狀態，如感測器故障]
- **WHEN** [使用者嘗試執行操作 X]
- **THEN** [系統 SHALL 顯示錯誤訊息]
- **AND** [系統 SHALL 進入安全狀態]
- **AND** [系統 MUST NOT 產生不正確的輸出]

**追溯關係**：
- Traces to: SysR-001, UR-005, HS-012
- Traced by: DES-001 (設計規格)
- Verified by: TC-FUNC-001-01, TC-FUNC-001-02 (測試案例)

---

#### SR-FUNC-002: [另一個功能需求]

[依此類推，建議每個主要功能有獨立的 SR-FUNC-xxx]

---

### 3.2 使用者介面需求（User Interface Requirements）

#### SR-UI-001: 可用性（Usability）

**描述**：使用者介面 **SHALL** 符合 IEC 62366-1 可用性工程要求

**具體要求**：
- 介面 **SHALL** 使用清晰可讀的字體（最小字號 12pt）
- 關鍵警報 **SHALL** 使用視覺（紅色）與聽覺（蜂鳴）雙重提示
- 操作按鈕 **SHALL** 有明確的標籤與功能說明
- 錯誤訊息 **SHALL** 提供明確的問題描述與建議操作

**追溯關係**：
- Traces to: Usability Risk Analysis (IEC 62366)
- Verified by: Usability Testing Report

---

### 3.3 數據輸入與輸出需求（Data I/O Requirements）

#### SR-DATA-001: 數據輸入驗證

**描述**：軟體 **SHALL** 驗證所有輸入數據的有效性

**具體要求**：
- 數值輸入 **SHALL** 檢查範圍（最小值、最大值）
- 日期/時間 **SHALL** 符合 ISO 8601 格式
- 患者 ID **SHALL** 經過格式與檢查碼驗證
- 無效輸入 **SHALL** 被拒絕並提示錯誤訊息

**安全考量**：
- 防止誤輸入導致錯誤診斷或治療

**追溯關係**：
- Traces to: HS-003 (輸入錯誤風險)
- Verified by: TC-DATA-001-01 (邊界值測試)

---

#### SR-DATA-002: 數據輸出準確性

**描述**：軟體 **SHALL** 確保輸出數據的準確性與一致性

**具體要求**：
- 計算結果 **SHALL** 精確到小數點後 [X] 位
- 測量值 **SHALL** 包含單位（如 mg/dL, mmHg）
- 輸出 **SHALL** 包含時間戳記與數據來源標識

**追溯關係**：
- Traces to: HS-005 (輸出錯誤風險)
- Verified by: TC-DATA-002-01 (準確性測試)

---

### 3.4 報警與通知需求（Alarms and Notifications）

#### SR-ALARM-001: 報警優先級

**描述**：軟體 **SHALL** 實施三級報警系統（高/中/低）

**優先級定義**：
- **高優先級（High）**：危及生命的狀況（如心跳停止）
  - **SHALL** 立即顯示紅色閃爍警報
  - **SHALL** 發出連續高音蜂鳴（不可靜音）
  - **SHALL** 記錄到報警日誌
- **中優先級（Medium）**：需要注意但非立即威脅（如異常讀數）
  - **SHALL** 顯示黃色警報
  - **SHALL** 發出間歇蜂鳴（可暫時靜音）
- **低優先級（Low）**：提示性訊息（如維護提醒）
  - **SHALL** 顯示訊息通知
  - **MAY** 發出單次提示音

**追溯關係**：
- Traces to: IEC 60601-1-8 (Medical electrical equipment alarms)
- Traces to: HS-010 (報警失效風險)
- Verified by: TC-ALARM-001-01

---

## 4. 非功能性軟體需求（Non-Functional Software Requirements）

### 4.1 效能需求（Performance Requirements）

#### SR-PERF-001: 回應時間

**描述**：系統 **SHALL** 在指定時間內回應使用者操作

**具體指標**：
- 關鍵生命體徵顯示更新：< 1 秒
- 一般查詢操作：< 2 秒
- 報表生成：< 10 秒
- 資料庫查詢：< 3 秒

**驗證方式**：Performance Test
**追溯關係**：
- Traces to: UR-010 (使用者體驗需求)
- Verified by: TC-PERF-001-01 (效能測試)

---

#### SR-PERF-002: 並發處理能力

**描述**：系統 **SHALL** 支援多使用者同時操作

**具體指標**：
- 同時線上使用者：≥ [X] 人
- 每秒交易處理數：≥ [Y] TPS
- 資料庫連線池：[Z] connections

---

### 4.2 安全需求（Security Requirements）

#### SR-SEC-001: 身份認證

**描述**：系統 **SHALL** 要求使用者身份驗證

**具體要求**：
- **SHALL** 支援多因素認證（MFA）for 管理員
- **SHALL** 實施帳號鎖定機制（3 次失敗嘗試）
- **SHALL** 強制密碼複雜度（≥8 字元，包含大小寫、數字、符號）
- **SHALL** 定期要求密碼變更（90 天）

**追溯關係**：
- Traces to: ISO 27001, HIPAA/GDPR (數據保護法規)
- Traces to: HS-020 (未授權存取風險)
- Verified by: TC-SEC-001-01 (滲透測試)

---

#### SR-SEC-002: 數據加密

**描述**：系統 **SHALL** 保護敏感數據的機密性

**具體要求**：
- 傳輸中數據 **SHALL** 使用 TLS 1.2+ 加密
- 靜態數據（如患者記錄）**SHALL** 使用 AES-256 加密
- 密碼 **SHALL** 使用安全哈希（bcrypt, Argon2）
- **MUST NOT** 以明文儲存敏感資訊

---

#### SR-SEC-003: 稽核日誌

**描述**：系統 **SHALL** 記錄所有安全相關事件

**日誌內容**：
- 使用者登入/登出
- 數據存取（讀取/修改患者記錄）
- 權限變更
- 系統配置修改
- 安全事件（失敗的登入嘗試、權限拒絕）

**日誌保存**：
- **SHALL** 保存至少 [X] 年（符合法規要求）
- **SHALL** 防止篡改（唯讀、數位簽章）

---

### 4.3 可靠性與可用性（Reliability and Availability）

#### SR-REL-001: 系統可用性

**描述**：系統 **SHALL** 達到高可用性目標

**具體指標**：
- 年度正常運作時間：≥ 99.9% (允許 8.76 小時停機/年)
- 平均故障間隔時間（MTBF）：≥ [X] 小時
- 平均修復時間（MTTR）：≤ [Y] 小時

---

#### SR-REL-002: 數據完整性

**描述**：系統 **SHALL** 確保數據完整性與一致性

**具體要求**：
- **SHALL** 使用交易機制（ACID）保護數據庫操作
- **SHALL** 定期自動備份（每日完整備份 + 每小時增量備份）
- **SHALL** 驗證備份可恢復性（月度測試）
- **SHALL** 使用檢查碼（checksum）驗證數據傳輸

---

### 4.4 可維護性（Maintainability）

#### SR-MAINT-001: 診斷與日誌

**描述**：系統 **SHALL** 提供診斷與除錯能力

**具體要求**：
- **SHALL** 記錄系統錯誤與警告（Log level: ERROR, WARN, INFO）
- **SHALL** 提供遠程診斷介面（僅限授權維護人員）
- **SHALL** 生成系統健康狀態報告

---

## 5. 風險控制需求（Risk Control Requirements）

> 本節對應 IEC 62304:5.2.3 — 軟體需求應包含風險控制措施

### 5.1 風險控制措施追溯

| 風險 ID | 危害場景 | 嚴重性 | 風險控制措施（軟體需求） | SR ID | 驗證方式 |
|---------|---------|--------|------------------------|-------|---------|
| HS-001 | 感測器數據錯誤導致誤診 | 嚴重 | 數據驗證與範圍檢查 | SR-DATA-001 | TC-DATA-001-01 |
| HS-002 | 軟體故障導致設備停機 | 嚴重 | 自動重啟與故障安全模式 | SR-REL-003 | TC-REL-003-01 |
| HS-003 | 未授權存取患者數據 | 中等 | 身份認證與授權控制 | SR-SEC-001 | TC-SEC-001-01 |
| HS-004 | 報警失效導致延誤治療 | 嚴重 | 冗餘報警機制與自我測試 | SR-ALARM-002 | TC-ALARM-002-01 |
| HS-005 | 數據丟失影響治療連續性 | 中等 | 自動備份與恢復機制 | SR-REL-002 | TC-REL-002-01 |

### 5.2 風險控制需求詳述

#### SR-RISK-001: 故障安全模式

**描述**：當檢測到軟體異常時，系統 **SHALL** 進入故障安全模式

**故障安全行為**：
- **SHALL** 停止所有可能造成傷害的操作（如藥物給藥、能量輸出）
- **SHALL** 發出故障警報（視覺 + 聽覺）
- **SHALL** 保存當前數據與系統狀態
- **SHALL** 記錄故障事件到日誌
- **SHOULD** 嘗試自動恢復（如果安全）

**追溯關係**：
- Traces to: HS-002 (軟體故障風險)
- Verified by: TC-RISK-001-01 (故障注入測試)

---

## 6. SOUP 需求（SOUP Requirements）

> 本節對應 IEC 62304:7.1 — 軟體開發使用 SOUP 的需求

### 6.1 SOUP 清單與需求

| SOUP 名稱 | 版本 | 用途 | 功能需求 | 安全要求 | 隔離措施 | 驗證方式 |
|----------|------|------|---------|---------|---------|---------|
| OpenSSL | 3.0.0 | 數據加密 | AES-256, TLS 1.2+ | 已知漏洞修補 | 錯誤處理包裝 | 整合測試 |
| PostgreSQL | 14.x | 數據庫 | ACID 交易支援 | 存取控制 | ORM 層隔離 | 數據完整性測試 |
| React | 18.x | 前端框架 | UI 渲染 | XSS 防護 | 輸入驗證 | UI 測試 |

### 6.2 SOUP 異常序列

**已知問題與緩解措施**：

| SOUP | 已知問題/限制 | 異常序列 | 緩解措施 | SR ID |
|------|-------------|---------|---------|-------|
| OpenSSL | CVE-XXXX-XXXX (已修補) | 加密失敗 → 返回錯誤碼 | 檢查返回值，記錄錯誤 | SR-SEC-004 |
| PostgreSQL | 連線池耗盡 | 連線超時 → 拒絕服務 | 連線超時處理，錯誤提示 | SR-PERF-003 |

詳見：`soup-management-template.md`

---

## 7. 介面需求（Interface Requirements）

### 7.1 硬體介面

| 介面 | 類型 | 協議 | 數據格式 | SR ID |
|------|------|------|---------|-------|
| 感測器 1 | Serial | RS-232, 9600 baud | ASCII, CR/LF | SR-HW-001 |
| 顯示器 | HDMI | HDMI 1.4 | 1920x1080, 60Hz | SR-HW-002 |

### 7.2 軟體介面

| 介面 | 系統 | 協議 | 數據格式 | SR ID |
|------|------|------|---------|-------|
| HIS 整合 | 醫院資訊系統 | HL7 v2.5 | HL7 ADT, ORM | SR-SW-001 |
| PACS 整合 | 影像系統 | DICOM | DICOM SR | SR-SW-002 |

### 7.3 通訊介面

| 介面 | 協議 | 安全 | 頻寬 | SR ID |
|------|------|------|------|-------|
| 網路通訊 | HTTPS | TLS 1.2+ | ≥ 100 Mbps | SR-COMM-001 |
| 數據同步 | WebSocket | WSS | ≥ 10 Mbps | SR-COMM-002 |

---

## 8. 追溯矩陣（Traceability Matrix）

### 8.1 需求追溯表

| UR ID | SysR ID | SR ID | 設計 ID | 測試 ID | 狀態 | 風險控制 |
|-------|---------|-------|---------|---------|------|---------|
| UR-001 | SysR-010 | SR-FUNC-001 | DES-001 | TC-FUNC-001-01 | ✅ Verified | HS-001 |
| UR-002 | SysR-011 | SR-ALARM-001 | DES-005 | TC-ALARM-001-01 | 🔄 In Progress | HS-010 |
| UR-003 | SysR-015 | SR-SEC-001 | DES-010 | TC-SEC-001-01 | ✅ Verified | HS-020 |

### 8.2 風險控制追溯

| 風險 ID | 嚴重性 | 風險控制措施 | SR ID | 設計 ID | 驗證方式 | 狀態 |
|---------|--------|-------------|-------|---------|---------|------|
| HS-001 | 嚴重 | 數據驗證 | SR-DATA-001 | DES-002 | TC-DATA-001-01 | ✅ Mitigated |
| HS-002 | 嚴重 | 故障安全模式 | SR-RISK-001 | DES-008 | TC-RISK-001-01 | 🔄 In Progress |

---

## 9. 驗收標準（Acceptance Criteria）

### 9.1 功能完整性

- ✅ 所有 SR-FUNC-xxx 需求都有對應的設計與實作
- ✅ 所有需求都有測試案例覆蓋
- ✅ 追溯矩陣完整無斷鏈

### 9.2 品質標準

- ✅ 需求審查通過（無 Critical/Major 問題）
- ✅ 所有風險控制措施都有對應的 SR
- ✅ SOUP 隔離與驗證完成

### 9.3 合規性標準

- ✅ 符合 IEC 62304:5.2 所有要求
- ✅ 需求可驗證（具體、可測試）
- ✅ 安全分級適當且有文檔支持

---

## 10. 審查與批准（Review and Approval）

### 10.1 需求審查記錄（IEC 62304:5.2.6）

| 審查日期 | 審查人 | 角色 | 審查範圍 | 發現問題數 | 解決狀態 |
|---------|--------|------|---------|-----------|---------|
| YYYY-MM-DD | [姓名] | 品質負責人 | 完整 SRS | 5 Major, 10 Minor | All Resolved |
| YYYY-MM-DD | [姓名] | 醫療顧問 | 臨床需求 | 2 Major | Resolved |
| YYYY-MM-DD | [姓名] | 風險管理負責人 | 風險控制需求 | 3 Minor | Resolved |

### 10.2 批准簽名

| 角色 | 姓名 | 簽名 | 日期 | 備註 |
|------|------|------|------|------|
| 專案經理 | [待填] | [待填] | YYYY-MM-DD | - |
| 品質負責人 | [待填] | [待填] | YYYY-MM-DD | - |
| 法規負責人 | [待填] | [待填] | YYYY-MM-DD | - |
| 醫療顧問 | [待填] | [待填] | YYYY-MM-DD | 若適用 |

---

## 11. 附錄（Appendix）

### 11.1 縮寫與術語

| 術語 | 全稱 | 說明 |
|------|------|------|
| SRS | Software Requirements Specification | 軟體需求規格 |
| SOUP | Software of Unknown Provenance | 未知來源軟體（第三方庫） |
| UR | User Requirement | 使用者需求 |
| SR | Software Requirement | 軟體需求 |
| HS | Hazard Scenario | 危害場景 |
| TC | Test Case | 測試案例 |

### 11.2 參考文件

- IEC 62304:2006+AMD1:2015 — Medical device software life cycle processes
- ISO 14971:2019 — Risk management for medical devices
- IEC 62366-1:2015 — Usability engineering for medical devices
- RFC 2119 — Key words for use in RFCs to Indicate Requirement Levels
- 系統需求規格：`[文檔位置]`
- 風險管理檔案：`[文檔位置]`
- SOUP 管理計畫：`soup-management-template.md`

---

**文檔版本**: v1.0.0
**維護人**: [姓名]
**最後審核**: YYYY-MM-DD
**下次審核**: YYYY-MM-DD（建議每次需求變更時）
