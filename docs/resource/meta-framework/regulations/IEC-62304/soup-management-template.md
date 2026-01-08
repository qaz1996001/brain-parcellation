# IEC 62304 SOUP 管理範本（SOUP Management Template）

**文件 ID**: IEC62304-SOUP-001
**標題**: [醫療設備名稱] — SOUP 管理計畫
**版本**: v1.0.0
**狀態**: Active
**建立日期**: YYYY-MM-DD
**最後更新**: YYYY-MM-DD
**作者**: [待填]
**審核人**: [架構師 / 品質負責人 待填]

---

## 變更歷史（Change History）

| 版本 | 日期 | 修改者 | 變更摘要 | SOUP 變更 |
|------|------|--------|---------|----------|
| v0.1 | YYYY-MM-DD | [姓名] | 初始SOUP清單 | 新增 10 個 SOUP |
| v1.0 | YYYY-MM-DD | [姓名] | 正式版本 | 更新 OpenSSL 3.0.0 → 3.0.1 |

---

## 1. SOUP 概述（SOUP Overview）

### 1.1 SOUP 定義

**SOUP** = **S**oftware **o**f **U**nknown **P**rovenance（未知來源軟體）

根據 IEC 62304:3.30 定義：
> 軟體項目，已經開發且通常可供使用，但**並非為了被納入正在開發的醫療設備軟體**而開發的軟體。

**典型 SOUP 範例**：
- 第三方開源庫（React, Django, NumPy）
- 商業軟體組件（Oracle Database, Windows OS）
- 標準工具（OpenSSL, zlib）
- 執行環境（Python interpreter, Node.js）

### 1.2 SOUP 管理目的

本計畫用於：
- 識別所有 SOUP（IEC 62304:7.1.2）
- 記錄 SOUP 需求與限制（IEC 62304:7.1.1）
- 評估 SOUP 風險與已知問題（IEC 62304:7.1.3）
- 隔離 SOUP 避免污染主系統（IEC 62304:5.3.4）
- 驗證 SOUP 整合（IEC 62304:5.6.5）

---

## 2. SOUP 清單（SOUP Inventory）

### 2.1 後端 SOUP（Backend）

| SOUP ID | SOUP 名稱 | 版本 | 供應商 | 用途 | 授權 | 風險等級 |
|---------|----------|------|--------|------|------|---------|
| **SOUP-001** | Python | 3.11.x | PSF | 執行環境 | PSF License | 高 |
| **SOUP-002** | FastAPI | 0.109.x | Tiangolo | Web 框架 | MIT | 中 |
| **SOUP-003** | PostgreSQL | 14.x | PostgreSQL Global Development Group | 數據庫 | PostgreSQL License | 高 |
| **SOUP-004** | SQLAlchemy | 2.0.x | SQLAlchemy authors | ORM | MIT | 中 |
| **SOUP-005** | Redis | 7.0.x | Redis Ltd | 快取 | BSD-3-Clause | 中 |
| **SOUP-006** | OpenSSL | 3.0.1 | OpenSSL Project | 加密 | Apache 2.0 | **Critical** |
| **SOUP-007** | Celery | 5.3.x | Celery Project | 異步任務 | BSD | 中 |
| **SOUP-008** | Pydantic | 2.5.x | Pydantic | 數據驗證 | MIT | 低 |

### 2.2 前端 SOUP（Frontend）

| SOUP ID | SOUP 名稱 | 版本 | 供應商 | 用途 | 授權 | 風險等級 |
|---------|----------|------|--------|------|------|---------|
| **SOUP-010** | React | 18.2.x | Meta | UI 框架 | MIT | 中 |
| **SOUP-011** | TypeScript | 5.3.x | Microsoft | 類型檢查 | Apache 2.0 | 低 |
| **SOUP-012** | Axios | 1.6.x | Axios Project | HTTP 客戶端 | MIT | 中 |
| **SOUP-013** | Chart.js | 4.4.x | Chart.js | 圖表渲染 | MIT | 低 |

### 2.3 開發工具 SOUP（Development Tools）

| SOUP ID | SOUP 名稱 | 版本 | 用途 | 風險等級 |
|---------|----------|------|------|---------|
| **SOUP-020** | pytest | 7.x | 單元測試 | 低（僅開發） |
| **SOUP-021** | Jest | 29.x | 前端測試 | 低（僅開發） |
| **SOUP-022** | ESLint | 8.x | 程式碼檢查 | 低（僅開發） |

**備註**：開發工具 SOUP 不納入最終交付產品，風險等級較低。

---

## 3. SOUP 需求規格（SOUP Requirements）

> 本節對應 IEC 62304:7.1.1 — 建立軟體開發使用 SOUP 的需求

### 3.1 SOUP 功能需求

#### SOUP-001: Python（執行環境）

**功能需求**：
- **SHALL** 支援 Python 3.11.x 語法與標準庫
- **SHALL** 提供穩定的多執行緒與異步 I/O 支援
- **SHALL** 符合 PEP 484 類型註解規範

**效能需求**：
- 啟動時間 < 500 ms
- GC (Garbage Collection) 暫停時間 < 100 ms

**安全需求**：
- **MUST NOT** 有已知的 Critical/High CVE 漏洞
- **SHALL** 支援 HTTPS、TLS 1.2+ 安全協議

---

#### SOUP-006: OpenSSL（加密庫）

**功能需求**：
- **SHALL** 提供 AES-256 加密/解密功能
- **SHALL** 支援 TLS 1.2, TLS 1.3 協議
- **SHALL** 提供安全隨機數生成（CSPRNG）

**效能需求**：
- AES-256 加密速度 ≥ 100 MB/s

**安全需求**：
- **Critical**: OpenSSL 版本 **MUST** 修補所有已知的 CVE
- **SHALL** 定期更新（每季度檢查安全公告）

**已知限制**：
- 不支援某些舊版 SSL/TLS 協議（這是預期行為，符合安全要求）

---

### 3.2 SOUP 選擇理由

| SOUP | 選擇理由 | 替代方案考量 |
|------|---------|------------|
| **PostgreSQL** | 成熟穩定、ACID 支援、強大的查詢能力 | MySQL (缺少某些進階功能), MongoDB (NoSQL 不符需求) |
| **React** | 業界標準、豐富生態系、組件化架構 | Vue (生態較小), Angular (過於複雜) |
| **OpenSSL** | 業界標準、廣泛審計、高效能 | BoringSSL (文檔較少), LibreSSL (相容性問題) |
| **FastAPI** | 高效能、自動 API 文檔、類型安全 | Flask (缺少類型支援), Django (過重) |

---

## 4. SOUP 風險評估（SOUP Risk Assessment）

> 本節對應 ISO 14971 風險管理要求

### 4.1 SOUP 相關危害場景

| 危害場景 ID | SOUP | 故障模式 | 潛在傷害 | 嚴重性 | 發生可能性 | 風險等級 | 緩解措施 |
|-----------|------|---------|---------|--------|-----------|---------|---------|
| **HS-SOUP-001** | OpenSSL | 加密失敗 | 患者數據洩露 | 嚴重 | 低 | 中 | 輸入驗證、錯誤處理、定期更新 |
| **HS-SOUP-002** | PostgreSQL | 數據庫崩潰 | 服務中斷 | 嚴重 | 低 | 中 | 自動備份、容錯設計、健康檢查 |
| **HS-SOUP-003** | React | XSS 漏洞 | 未授權存取 | 中等 | 中 | 中 | 輸入驗證、Content Security Policy |
| **HS-SOUP-004** | Python | 解釋器故障 | 系統當機 | 嚴重 | 極低 | 低 | 異常捕獲、自動重啟 |

### 4.2 SOUP 已知問題與異常序列

> 本節對應 IEC 62304:7.1.3 — 記錄 SOUP 異常序列

#### SOUP-003: PostgreSQL

**已知問題**：
1. **連線池耗盡**
   - **異常序列**：高並發請求 → 連線池滿 → 新請求超時
   - **緩解措施**：
     - 設置連線超時（30 秒）
     - 監控連線池使用率（警報閾值 80%）
     - 實施連線重用與釋放機制
   - **對應 SR**: SR-PERF-003

2. **長時間執行查詢**
   - **異常序列**：複雜查詢 → 阻塞其他事務 → 性能下降
   - **緩解措施**：
     - 設置查詢超時（60 秒）
     - 異步處理長查詢
     - 查詢優化（索引、分頁）

---

#### SOUP-006: OpenSSL

**已知問題**：
1. **歷史 CVE 漏洞**（已修補）
   - **CVE-2023-XXXX**（Heartbleed-like）：已在 3.0.1 修補
   - **當前版本**：3.0.1（無已知 Critical CVE）
   - **更新策略**：每季度檢查安全公告，必要時更新

2. **API 回傳錯誤碼**
   - **異常序列**：加密/解密失敗 → OpenSSL 返回錯誤碼
   - **緩解措施**：
     - 檢查所有 OpenSSL API 返回值
     - 將錯誤碼轉換為應用層異常（`EncryptionFailureError`）
     - 記錄錯誤到安全日誌
   - **對應 SR**: SR-SEC-004

---

## 5. SOUP 隔離設計（SOUP Isolation）

> 本節對應 IEC 62304:5.3.4 — 隔離 SOUP

### 5.1 隔離策略

**Wrapper Pattern（封裝模式）**：
- 為每個 SOUP 建立隔離層（Wrapper/Adapter）
- 應用層僅通過 Wrapper 與 SOUP 互動
- Wrapper 負責異常轉換與錯誤處理

**隔離層架構**：
```
┌─────────────────────────────────────┐
│      應用層（Application Layer）      │
│   PatientService, AlarmManager...   │
└──────────────┬──────────────────────┘
               │ 使用抽象介面
┌──────────────┴──────────────────────┐
│        隔離層（Isolation Layer）      │
│   DatabaseWrapper, CacheWrapper...   │
└──────────────┬──────────────────────┘
               │ 隱藏 SOUP 細節
┌──────────────┴──────────────────────┐
│           SOUP 層（SOUP Layer）      │
│   PostgreSQL, Redis, OpenSSL...      │
└──────────────────────────────────────┘
```

### 5.2 SOUP Wrapper 設計範例

#### DatabaseWrapper（PostgreSQL 隔離）

```python
class DatabaseWrapper:
    """PostgreSQL SOUP 隔離層 (IEC 62304:5.3.4)"""

    def __init__(self, connection_string: str):
        self._engine = create_engine(connection_string)  # SOUP: SQLAlchemy
        self._session_factory = sessionmaker(bind=self._engine)
        self._logger = Logger("DatabaseWrapper")

    def execute_query(self, query: str) -> List[Dict]:
        """執行查詢（隔離 SOUP 異常）

        SOUP 異常轉換：
            - OperationalError → DatabaseConnectionError
            - IntegrityError → DataIntegrityError
            - ProgrammingError → QuerySyntaxError
            - All other → GenericDatabaseError

        IEC 62304:7.1.3 異常序列處理
        """
        session = None
        try:
            session = self._session_factory()
            result = session.execute(text(query))
            return [dict(row) for row in result]

        except OperationalError as e:
            # SOUP 異常：數據庫連線失敗
            self._logger.error(f"Database connection failed: {e}")
            raise DatabaseConnectionError(
                message="Unable to connect to database",
                original_error=str(e)
            )

        except IntegrityError as e:
            # SOUP 異常：數據完整性違反（如主鍵衝突）
            self._logger.warning(f"Data integrity violation: {e}")
            raise DataIntegrityError(
                message="Database constraint violated",
                original_error=str(e)
            )

        except ProgrammingError as e:
            # SOUP 異常：SQL 語法錯誤
            self._logger.error(f"Query syntax error: {e}")
            raise QuerySyntaxError(
                message="Invalid SQL query",
                query=query,
                original_error=str(e)
            )

        except Exception as e:
            # 未知 SOUP 異常（記錄詳細資訊以便分析）
            self._logger.critical(f"Unexpected database error: {e}")
            raise GenericDatabaseError(
                message="Unexpected database operation failed",
                original_error=str(e)
            )

        finally:
            if session:
                session.close()
```

**隔離驗證**（IEC 62304:5.6.5）：
- [ ] 單元測試：模擬所有已知 SOUP 異常，驗證轉換正確
- [ ] 整合測試：故意觸發 SOUP 錯誤，驗證系統穩定性
- [ ] 邊界測試：驗證異常不會洩露到應用層

---

## 6. SOUP 整合測試（SOUP Integration Testing）

> 本節對應 IEC 62304:5.6.5 — 測試 SOUP 整合

### 6.1 SOUP 整合測試計畫

| 測試 ID | SOUP | 測試目標 | 測試方法 | 狀態 |
|---------|------|---------|---------|------|
| **TC-SOUP-001** | PostgreSQL | 驗證連線失敗處理 | 停止數據庫服務，驗證異常轉換 | ✅ Pass |
| **TC-SOUP-002** | PostgreSQL | 驗證數據完整性錯誤 | 插入重複主鍵，驗證回滾 | ✅ Pass |
| **TC-SOUP-003** | OpenSSL | 驗證加密/解密正確性 | 加密數據後解密，驗證一致性 | ✅ Pass |
| **TC-SOUP-004** | Redis | 驗證快取失效降級 | 停止 Redis，驗證降級到數據庫 | 🔄 In Progress |
| **TC-SOUP-005** | React | 驗證 XSS 防護 | 注入惡意腳本，驗證無執行 | ✅ Pass |

### 6.2 SOUP 整合測試案例範例

#### TC-SOUP-001: PostgreSQL 連線失敗處理

**測試目的**：驗證 SOUP 隔離層正確處理數據庫連線失敗

**測試步驟**：
1. **GIVEN** PostgreSQL 服務正常運行
2. **WHEN** 停止 PostgreSQL 服務
3. **AND** 應用層嘗試執行查詢
4. **THEN**
   - 拋出 `DatabaseConnectionError`（非 PostgreSQL 原生異常）
   - 錯誤訊息清晰（"Unable to connect to database"）
   - 系統**不崩潰**（異常被正確處理）
   - 錯誤被記錄到日誌

**預期結果**：
- [ ] 異常類型正確（DatabaseConnectionError）
- [ ] 系統狀態穩定（無資料損壞）
- [ ] 錯誤日誌完整

---

## 7. SOUP 更新與變更管理

### 7.1 SOUP 更新策略

**定期檢查**：
- 安全更新：每月檢查 CVE 公告
- 功能更新：每季度評估新版本
- 重大版本升級：經過完整驗證後才升級

**更新觸發條件**：
- **Critical/High CVE 發布**：立即評估並更新
- **功能需求**：新版本提供必要功能
- **效能改進**：顯著效能提升（>20%）

### 7.2 SOUP 變更流程

```
發現新版本 → 評估影響 → 測試驗證 → 審查批准 → 更新部署 → 驗證上線
```

**變更記錄範例**：

| 變更日期 | SOUP | 舊版本 | 新版本 | 變更原因 | 影響評估 | 測試狀態 | 批准人 |
|---------|------|--------|--------|---------|---------|---------|--------|
| 2024-01-15 | OpenSSL | 3.0.0 | 3.0.1 | CVE-2023-XXXX 修補 | Critical 安全更新 | ✅ Pass | [QA 負責人] |
| 2024-02-01 | React | 18.2.0 | 18.2.1 | Bug 修復 | 低風險，相容性良好 | ✅ Pass | [架構師] |

---

## 8. SOUP 供應商評估

### 8.1 供應商可靠性評估

| SOUP | 供應商 | 社群活躍度 | 發布頻率 | 文檔品質 | 安全公告 | 評級 |
|------|--------|-----------|---------|---------|---------|------|
| PostgreSQL | PostgreSQL GDG | 極高 | 穩定（每年 1-2 次） | 優秀 | 及時且透明 | A+ |
| React | Meta (Facebook) | 極高 | 穩定（每月修補） | 優秀 | 良好 | A |
| OpenSSL | OpenSSL Project | 高 | 穩定（每季度） | 良好 | 及時 | A |
| FastAPI | Tiangolo | 高 | 頻繁（每月） | 優秀 | 良好 | B+ |

### 8.2 開源 SOUP 風險緩解

**風險**：開源 SOUP 可能缺乏商業支援

**緩解措施**：
- 選擇成熟且廣泛使用的 SOUP（社群大、文檔全）
- 建立 SOUP 隔離層（降低更換成本）
- 定期備份與離線儲存關鍵版本
- 考慮商業支援方案（如 PostgreSQL Enterprise）

---

## 9. 審查與批准（Review and Approval）

### 9.1 SOUP 清單審查記錄

| 審查日期 | 審查人 | 審查範圍 | 發現問題 | 解決狀態 |
|---------|--------|---------|---------|---------|
| YYYY-MM-DD | [架構師] | 完整 SOUP 清單 | 3 個 SOUP 版本過舊 | All Updated |
| YYYY-MM-DD | [安全負責人] | SOUP 安全評估 | OpenSSL CVE 待更新 | Updated to 3.0.1 |

### 9.2 批准簽名

| 角色 | 姓名 | 簽名 | 日期 |
|------|------|------|------|
| 專案經理 | [待填] | [待填] | YYYY-MM-DD |
| 架構師 | [待填] | [待填] | YYYY-MM-DD |
| 品質負責人 | [待填] | [待填] | YYYY-MM-DD |
| 安全負責人 | [待填] | [待填] | YYYY-MM-DD |

---

## 10. 附錄（Appendix）

### 10.1 SOUP 授權合規性

| SOUP | 授權 | 商業使用 | 醫療設備使用 | 合規性檢查 |
|------|------|---------|-------------|-----------|
| PostgreSQL | PostgreSQL License | ✅ 允許 | ✅ 允許 | ✅ 通過 |
| React | MIT | ✅ 允許 | ✅ 允許 | ✅ 通過 |
| OpenSSL | Apache 2.0 | ✅ 允許 | ✅ 允許 | ✅ 通過 |

**法務審查**：[法務部門簽名] YYYY-MM-DD

### 10.2 參考文件

- IEC 62304:2006+AMD1:2015 — 條款 5.3.4, 5.6.5, 7.1
- ISO 14971:2019 — Risk management for SOUP-related hazards
- Software Requirements Specification: `software-requirements-spec-template.md`
- Software Design Specification: `software-design-spec-template.md`

---

**文檔版本**: v1.0.0
**維護人**: [姓名]
**最後審核**: YYYY-MM-DD
**下次審核**: YYYY-MM-DD（建議每季度或 SOUP 更新時）
