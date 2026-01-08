# AI 協作模式指南（AI Collaboration Patterns Guide）

**文件 ID**: GUIDE-AI-001  
**版本**: v1.0.0  
**建立日期**: 2025-12-22  

---

## 1. 人機協作哲學（Human-AI Collaboration Philosophy）

### 1.1 核心原則

本元框架建立在以下原則之上：

1. **人類負責決策，AI 負責執行**
   - 人類：定義需求、審核提案、驗收結果
   - AI：撰寫文檔、產生程式碼、執行測試

2. **明確的職責邊界**
   - 不存在「灰色地帶」— 每個任務都有明確的負責方
   - AI 不應代替人類做商業決策或產品方向判斷

3. **可追溯、可稽核**
   - 所有 AI 產出都記錄來源（OpenSpec 提案、commit 記錄）
   - 人類可隨時檢視 AI 的推理過程與決策依據

4. **持續學習與改進**
   - 人類回饋 AI 產出品質（透過審查、驗收）
   - AI 根據規範與歷史案例持續優化產出

---

## 2. 職責分工矩陣（Responsibility Matrix）

### 2.1 RACI 矩陣

| 階段 | 任務 | 人類角色 | AI 角色 |
|------|------|---------|--------|
| **需求定義** | 訪談利害關係人 | **R** (Responsible) | I (Informed) |
| | 撰寫使用者需求（UR-xxx） | **A** (Accountable) | **C** (Consulted) |
| | 結構化需求文檔 | **C** | **R** |
| **系統設計** | 技術方案選擇 | **A** | **C** |
| | 撰寫設計文檔 | **C** | **R** |
| | 繪製架構圖 | C | **R** |
| **變更提案** | 提出變更需求 | **R** | I |
| | 撰寫 proposal.md | I | **R** |
| | 審核提案 | **A** | I |
| | 批准/拒絕提案 | **R** | - |
| **開發實作** | 撰寫程式碼 | **C** | **R** |
| | 撰寫單元測試 | C | **R** |
| | 程式碼審查（關鍵部分） | **R** | I |
| | 整合測試 | **A** | **R** |
| **變更歸檔** | 歸檔 OpenSpec 變更 | I | **R** |
| | 更新 PRD/SR/SD 文檔 | **A** | **R** |
| | 最終審核與部署 | **R** | I |

**圖例**：
- **R** (Responsible)：負責執行
- **A** (Accountable)：最終負責、承擔後果
- **C** (Consulted)：諮詢意見
- **I** (Informed)：知會

---

## 3. AI 使用指南（For AI Agents）

### 3.1 接收任務時的檢查清單

當你（AI）接收到開發任務時，依照以下流程：

#### ✅ 步驟 1：理解專案現狀

```markdown
1. 閱讀專案核心文檔
   - [ ] `docs/requirements/01_SYSTEM_PRD_SR_SD.md`（系統層需求）
   - [ ] `docs/requirements/0X_SUBSYSTEM_PRD_SR_SD.md`（相關子系統需求）
   - [ ] `openspec/project.md`（專案慣例與規範）

2. 檢查進行中的變更
   - [ ] 執行 `openspec list` 查看活動中的提案
   - [ ] 檢查是否與你的任務有衝突或依賴

3. 確認範圍邊界
   - [ ] 這是新功能（需要 OpenSpec 提案）還是 Bug 修復（直接修改）？
   - [ ] 是否影響多個子系統？
   - [ ] 是否有 Breaking Change？
```

#### ✅ 步驟 2：規劃工作

```markdown
**如果是新功能或重大變更**：
1. [ ] 創建 OpenSpec 提案（`/openspec-proposal`）
2. [ ] 撰寫 proposal.md（Why, What, Impact）
3. [ ] 撰寫 tasks.md（可執行檢查清單）
4. [ ] 撰寫 design.md（如果跨系統或複雜）
5. [ ] 撰寫 spec deltas（ADDED/MODIFIED/REMOVED）
6. [ ] 執行 `openspec validate <change-id> --strict`
7. [ ] 等待人類審核批准

**如果是 Bug 修復或小調整**：
1. [ ] 直接修改程式碼
2. [ ] 撰寫或更新測試
3. [ ] 更新相關文檔（如 API 文檔）
```

#### ✅ 步驟 3：實作與驗證

```markdown
1. 嚴格遵循 tasks.md 的順序
2. 每完成一項，標記為 `- [x]`
3. 確保所有測試通過
4. 更新技術文檔（API、資料庫 schema）
5. 提交前執行 linter 檢查
```

#### ✅ 步驟 4：歸檔與同步

```markdown
1. [ ] 執行 `openspec archive <change-id> --yes`
2. [ ] 將 spec deltas 同步到 PRD/SR/SD
3. [ ] 更新追溯矩陣
4. [ ] 更新需求狀態（Draft → Verified）
5. [ ] 執行 `openspec validate --strict` 確保一致性
```

### 3.2 AI 產出品質標準

**文檔撰寫**：
- ✅ 使用規範性語言（SHALL/SHOULD/MAY）
- ✅ 每個需求都有至少一個 Scenario
- ✅ 追溯關係完整（Traces to, Traced by, Verified by）
- ✅ 符合 SMART 原則（Specific, Measurable, Achievable, Relevant, Testable）

**程式碼撰寫**：
- ✅ 遵循專案的程式碼風格（ESLint, Prettier, Black, Ruff）
- ✅ 單元測試覆蓋率 > 70%
- ✅ 關鍵邏輯有註解說明
- ✅ 無 linter 錯誤或警告

**OpenSpec 提案**：
- ✅ proposal.md 清楚說明 Why, What, Impact
- ✅ tasks.md 可執行、具體、可檢查
- ✅ spec deltas 格式正確（通過 `openspec validate --strict`）

### 3.3 當遇到模糊需求時

**AI 應該做的**：
1. ✅ 提出 1-2 個具體問題澄清需求
2. ✅ 提供多個選項供人類選擇
3. ✅ 明確說明不確定的地方

**AI 不應該做的**：
1. ❌ 自行猜測需求並直接實作
2. ❌ 忽略模糊需求繼續進行
3. ❌ 產出過於通用或抽象的解決方案

**範例**：

❌ **不好的回應**：
```
我理解了，我會實作一個通知功能。
```

✅ **好的回應**：
```
我理解你需要通知功能，但有幾點需要澄清：

1. 通知管道：
   - 選項 A：僅 Email 通知
   - 選項 B：Email + 站內訊息
   - 選項 C：Email + 站內訊息 + Push（需整合 Firebase）

2. 通知時機：
   - 即時發送？
   - 批量發送（如每日彙整）？

3. 通知內容：
   - 僅標題與連結？
   - 包含完整內容摘要？

請選擇選項或提供更多細節，我將據此創建 OpenSpec 提案。
```

---

## 4. 人類使用指南（For Human Users）

### 4.1 如何與 AI 有效協作

#### ✅ 提供清楚的需求

**不好的需求**：
```
幫我做一個搜尋功能。
```

**好的需求**：
```
需求：全文搜尋功能

目標使用者：臨床研究人員
使用情境：從 10,000+ 份報告中快速找到包含特定關鍵字的報告
驗收標準：
- 支援報告內容、患者姓名、檢查日期的搜尋
- 搜尋回應時間 < 2 秒
- 結果高亮顯示關鍵字
- 支援分頁（20 筆/頁）

優先級：高（Phase 1 必備）
對應 UR：UR-004
```

#### ✅ 審核 AI 產出的提案

當 AI 提交 OpenSpec 提案時，檢查：

```markdown
**proposal.md 檢查清單**：
- [ ] Why 部分清楚說明商業價值
- [ ] What Changes 具體且完整
- [ ] Impact 評估合理（受影響的 specs、程式碼、文檔）
- [ ] 有考慮風險與緩解措施

**tasks.md 檢查清單**：
- [ ] 任務拆分合理（不太粗也不太細）
- [ ] 有明確的驗證步驟（測試、文檔更新）
- [ ] 依賴關係清楚（如需要先完成其他任務）

**spec deltas 檢查清單**：
- [ ] 需求符合 SMART 原則
- [ ] 每個需求都有 Scenario
- [ ] 追溯關係完整
```

#### ✅ 驗收實作成果

```markdown
**功能驗收**：
- [ ] 依據 Scenario 逐項測試
- [ ] 測試邊界條件與錯誤處理
- [ ] 測試效能（如有效能需求）

**程式碼品質**：
- [ ] 關鍵邏輯進行 Code Review
- [ ] 檢查測試覆蓋率
- [ ] 檢查有無明顯的技術債

**文檔完整性**：
- [ ] PRD/SR/SD 已同步更新
- [ ] API 文檔已更新（如有 API 變更）
- [ ] 使用者手冊已更新（如影響使用者操作）
```

### 4.2 如何拒絕 AI 提案

**範例情境**：AI 提出的設計方案過於複雜。

**不好的回應**：
```
不行，重做。
```

**好的回應**：
```
提案拒絕原因：

1. 複雜度過高：提案引入了 Redis、Celery、RabbitMQ 三個新依賴，但 Phase 1 目標是 MVP，應優先使用簡單方案。

2. 建議調整：
   - 使用 FastAPI BackgroundTasks 處理非同步任務（而非 Celery）
   - 使用 PostgreSQL JSONB 儲存通知記錄（而非 Redis）
   - Phase 2 再考慮引入訊息佇列

請根據以上建議調整提案，重新提交。
```

### 4.3 定期審計與維護

**每週檢查**：
```bash
# 檢查進行中的變更
openspec list

# 檢查是否有提案卡住未完成
# 檢查是否有提案需要審核
```

**每月檢查**：
```bash
# 檢查需求覆蓋率
python scripts/coverage_report.sh

# 檢查追溯性完整性
python scripts/validate_traceability.py
```

**每季度檢查**：
```markdown
- [ ] PRD/SR/SD 與實作一致性審計
- [ ] OpenSpec archived changes 是否都已同步到文檔
- [ ] 合規性自我審計（如適用相關標準）
```

---

## 5. 協作模式案例（Collaboration Patterns）

### 案例 1：新功能開發

**情境**：產品經理提出「需要 AI 批量分析功能」。

**流程**：

```markdown
1. **人類（PM）**：撰寫使用者需求
   UR-008: 批量 AI 分析
   身為研究人員，我希望能一次分析多份報告，以便快速完成篩選工作。
   驗收標準：
   - 單次批量分析支援至少 50 份報告
   - 顯示進度（已完成 / 總數）
   - 支援背景執行，使用者可離開頁面

2. **AI**：創建 OpenSpec 提案
   - 閱讀現有的 SYS-SR-010（單報告分析）
   - 創建提案 `add-batch-ai-analysis`
   - 撰寫 proposal.md, tasks.md, spec deltas
   - 執行 `openspec validate --strict`
   - 提交審核

3. **人類（Tech Lead）**：審核提案
   - 檢查技術方案（使用 Celery 任務佇列）
   - 確認 tasks.md 可執行
   - 批准提案

4. **AI**：實作功能
   - 依據 tasks.md 實作（後端 API、Celery 任務、前端 UI）
   - 撰寫測試
   - 更新 API 文檔

5. **人類（QA）**：驗收測試
   - 依據 Scenario 測試
   - 提出 Bug 或改進建議

6. **AI**：修復 Bug 並歸檔
   - 修復問題
   - 執行 `openspec archive add-batch-ai-analysis --yes`
   - 同步到 PRD/SR/SD
   - 更新追溯矩陣
```

### 案例 2：Bug 修復

**情境**：使用者報告「搜尋結果不正確」。

**流程**：

```markdown
1. **人類（Developer）**：重現 Bug
   - 確認搜尋「肺結節」時部分報告未顯示
   - 檢查程式碼，發現是 SQL 查詢條件錯誤

2. **AI**：修復 Bug（無需 OpenSpec 提案）
   - 修改 SQL 查詢邏輯
   - 補充測試案例（防止迴歸）
   - 提交 commit：「fix: 修正搜尋查詢條件錯誤」

3. **人類（Developer）**：驗證修復
   - 執行測試
   - 手動驗證搜尋結果正確
   - 批准合併
```

### 案例 3：技術重構

**情境**：後端需要將同步 API 改為異步處理（效能優化）。

**流程**：

```markdown
1. **人類（Tech Lead）**：提出重構需求
   - 目標：提升 API 回應速度
   - 範圍：AI 分析 API（`/api/reports/{id}/analyze`）
   - 影響：前端需調整（polling 查詢結果）

2. **AI**：創建 OpenSpec 提案（因為是 Breaking Change）
   - 提案 ID：`refactor-async-ai-analysis`
   - proposal.md 說明：
     - Why：同步處理導致 API 超時（> 60 秒）
     - What：改為異步，立即返回 task_id
     - Impact：**BREAKING** — 前端需修改 API 呼叫方式
   - design.md 說明：
     - 使用 Celery 任務佇列
     - 前端 polling `/api/tasks/{task_id}` 查詢狀態
   - spec deltas：
     - MODIFIED SYS-SR-010（API 行為變更）

3. **人類（Tech Lead + PM）**：審核提案
   - 評估 Breaking Change 影響（需通知前端團隊）
   - 確認遷移計畫（提供相容層或直接切換）
   - 批准提案

4. **AI**：實作重構
   - 後端改為異步
   - 前端修改 API 呼叫邏輯
   - 更新 API 文檔（標註 Breaking Change）

5. **人類（QA）**：驗收測試
   - 功能測試
   - 效能測試（確認改善）
   - 驗證前端相容性

6. **AI**：歸檔並同步
   - 歸檔 OpenSpec 變更
   - 更新 PRD/SR/SD（SYS-SR-010 的 Scenario 已變更）
   - 更新 API 文檔的 Breaking Change 記錄
```

---

## 6. 常見協作陷阱（Common Pitfalls）

### 陷阱 1：人類過度依賴 AI

**問題**：人類不審查 AI 產出，直接部署。

**後果**：
- 潛在 Bug 進入生產環境
- 技術債累積
- 設計決策不符合業務需求

**解決**：
- 建立審核機制（Pull Request Review）
- 關鍵功能必須人類驗收
- 定期程式碼審查（不只是 AI 產出）

### 陷阱 2：AI 自作主張

**問題**：AI 未經審核就實作功能或做設計決策。

**後果**：
- 需求偏離預期
- 產生不必要的複雜度
- 浪費開發時間

**解決**：
- 強制 OpenSpec 提案審核流程（重大變更必須先審核）
- AI 遇到模糊需求時明確提問
- 定期檢查 AI 的工作方向

### 陷阱 3：文檔與程式碼不同步

**問題**：程式碼已改，但 PRD/SR/SD 沒更新。

**後果**：
- 文檔失去參考價值
- 新成員無法理解系統
- 合規性稽核失敗

**解決**：
- 歸檔 OpenSpec 變更時強制同步文檔
- 使用 CI/CD 檢查文檔一致性
- 定期文檔審計（每月或每季）

---

## 7. 成功協作的指標（Success Metrics）

### 7.1 流程指標

| 指標 | 目標值 | 說明 |
|------|--------|------|
| OpenSpec 提案通過率 | > 80% | 表示 AI 產出品質高 |
| 需求變更到實作週期 | < 2 週 | 表示流程高效 |
| 文檔與程式碼一致性 | > 95% | 定期審計檢查 |
| AI 產出測試覆蓋率 | > 70% | 表示程式碼品質高 |

### 7.2 品質指標

| 指標 | 目標值 | 說明 |
|------|--------|------|
| 生產環境 Bug 率 | < 5% | 每月新功能的 Bug 數 / 總功能數 |
| 需求追溯完整性 | 100% | 所有需求都有追溯鏈 |
| 文檔更新延遲 | < 24 小時 | 程式碼變更後文檔更新時間 |

### 7.3 協作效率指標

| 指標 | 目標值 | 說明 |
|------|--------|------|
| AI 產出返工率 | < 20% | 需要重新實作的比例 |
| 人類審核時間 | < 1 天 | 提案提交到審核完成的時間 |
| 知識傳承效率 | < 1 週 | 新成員熟悉框架與流程的時間 |

---

## 8. 持續改進（Continuous Improvement）

### 8.1 回饋機制

**定期回顧會議**（每兩週一次）：
- 檢視完成的 OpenSpec 變更
- 討論 AI 產出品質
- 識別流程瓶頸
- 提出改進建議

**回顧模板**：
```markdown
## 完成的變更
- [變更 1]：[結果、學到什麼]
- [變更 2]：[結果、學到什麼]

## AI 產出品質
- 好的方面：[...]
- 需改進：[...]

## 流程問題
- 問題 1：[描述] → 解決方案：[...]
- 問題 2：[描述] → 解決方案：[...]

## 下次迭代改進
- [ ] 行動項 1
- [ ] 行動項 2
```

### 8.2 框架演進

- 根據使用經驗更新範本
- 補充常見案例與最佳實踐
- 調整流程以提升效率
- 與社群分享經驗

---

## 9. 下一步（Next Steps）

### 開始使用本框架

1. ✅ 閱讀本指南（完成！）
2. 📖 閱讀 OpenSpec 整合指南（`openspec-integration/00_OPENSPEC_INTEGRATION.md`）
3. 🎯 實作第一個提案（練習人機協作流程）
4. 🔄 進行第一次回顧會議（評估協作效果）
5. 🚀 逐步完善流程與範本

### 進階主題

- 📚 追溯性管理指南（`guides/TRACEABILITY_MANAGEMENT.md`）（待創建）
- 🏥 合規性管理（`regulations/`）
- 🔧 自動化工具開發（驗證腳本、CI/CD 整合）

---

**文檔版本**: v1.0.0  
**維護團隊**: MetaFramework Core Team  
**最後審核**: 2025-12-22

