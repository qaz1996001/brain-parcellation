# IEC 62304 軟體設計規格範本（Software Design Specification Template）

**文件 ID**: IEC62304-SDD-001
**標題**: [醫療設備名稱] — 軟體設計規格
**版本**: v1.0.0
**狀態**: Draft
**IEC 62304 安全級別**: [Class A / Class B / Class C]
**建立日期**: YYYY-MM-DD
**最後更新**: YYYY-MM-DD
**作者**: [待填]
**審核人**: [架構師 / 品質負責人 / 醫療顧問 待填]

---

## 變更歷史（Change History）

| 版本 | 日期 | 修改者 | 變更摘要 | IEC 62304 條款 |
|------|------|--------|---------|---------------|
| v0.1 | YYYY-MM-DD | [姓名] | 架構設計初稿 | 5.3.2 |
| v0.2 | YYYY-MM-DD | [姓名] | 詳細設計補充 | 5.4.2 |
| v1.0 | YYYY-MM-DD | [姓名] | 首次正式版本 | 5.3.6, 5.4.4 |

---

## 1. 設計概述（Design Overview）

### 1.1 設計目的

本文檔描述 [醫療設備軟體] 的：
- 軟體架構設計（Software Architectural Design）— IEC 62304:5.3
- 軟體詳細設計（Software Detailed Design）— IEC 62304:5.4

### 1.2 設計原則

**安全設計原則**：
- **Defence in Depth**: 多層次安全防護
- **Fail-Safe Design**: 故障安全設計
- **SOUP Isolation**: 第三方軟體隔離（IEC 62304:5.3.4）
- **Redundancy**: 關鍵功能冗餘設計

**品質設計原則**：
- **Modularity**: 模組化設計
- **Separation of Concerns**: 關注點分離
- **Testability**: 可測試性
- **Maintainability**: 可維護性

### 1.3 需求追溯

**設計來源**：
- 軟體需求規格（SRS）：`software-requirements-spec-template.md`
- 風險管理檔案（RMF）：`[文檔位置]`
- SOUP 管理計畫：`soup-management-template.md`

---

## 2. 架構設計（Architectural Design）

> 本節對應 IEC 62304:5.3 — 軟體架構設計

### 2.1 系統架構概覽

**架構圖**：

```
┌─────────────────────────────────────────────────────────┐
│                    使用者介面層 (UI Layer)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Web前端      │  │  行動應用     │  │  設備顯示器   │  │
│  │  (React)      │  │  (React Native)│  │  (Embedded) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                        │ HTTPS / REST API
┌────────────────────────┴────────────────────────────────┐
│                  應用層 (Application Layer)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  API Gateway  │  │  業務邏輯層   │  │  報警管理     │  │
│  │  (FastAPI)    │  │  (Service)    │  │  (Alarm Mgr) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                        │
┌────────────────────────┴────────────────────────────────┐
│               數據與持久層 (Data & Persistence Layer)     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  數據庫 ORM   │  │  檔案儲存     │  │  快取層       │  │
│  │  (SQLAlchemy) │  │  (S3/Local)   │  │  (Redis)     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                        │
┌────────────────────────┴────────────────────────────────┐
│                 設備與外部介面層 (Device & External Layer)│
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  感測器驅動   │  │  HIS 整合     │  │  PACS 整合   │  │
│  │  (Driver)     │  │  (HL7)        │  │  (DICOM)     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 架構層次劃分

| 層次 | 職責 | 主要組件 | 對應 SR | 安全考量 |
|------|------|---------|---------|---------|
| **UI Layer** | 使用者互動 | Web/Mobile UI | SR-UI-001 | XSS 防護、輸入驗證 |
| **Application Layer** | 業務邏輯 | Services, API | SR-FUNC-xxx | 授權控制、數據驗證 |
| **Data Layer** | 數據持久化 | Database, Cache | SR-DATA-xxx | 加密、備份、完整性 |
| **Device Layer** | 硬體/外部整合 | Drivers, Interfaces | SR-HW-xxx, SR-SW-xxx | 錯誤處理、隔離 |

### 2.3 關鍵架構決策（Architectural Decisions）

#### ADR-001: 採用微服務架構

**背景**：需要支援多平台（Web, Mobile, Embedded）且系統需高可用性

**決策**：採用微服務架構，各服務獨立部署與擴展

**理由**：
- 提高可維護性與可擴展性
- 容錯設計（單一服務故障不影響全系統）
- 符合 IEC 62304:5.3.5 軟體項目分離要求

**風險緩解**：
- 服務間通訊使用 HTTPS + 認證
- 實施 Circuit Breaker 防止級聯故障

**追溯關係**：
- Traces to: SR-REL-001 (可用性需求)
- Mitigates: HS-002 (系統故障風險)

---

#### ADR-002: SOUP 隔離策略

**背景**：系統使用多個第三方庫（PostgreSQL, Redis, React）

**決策**：使用 Wrapper Pattern 隔離所有 SOUP

**實施**：
- 數據庫存取：ORM 層封裝（SQLAlchemy）
- 快取：Cache Service 封裝（Redis Client）
- 前端框架：Component Library 封裝（React）

**理由**：
- 符合 IEC 62304:5.3.4 SOUP 隔離要求
- 便於未來更換 SOUP 版本或替代方案
- 集中處理 SOUP 異常與錯誤

**驗證**：
- SOUP 整合測試（IEC 62304:5.6.5）
- 邊界測試（驗證隔離有效性）

**追溯關係**：
- Traces to: IEC 62304:5.3.4, 7.1
- Mitigates: HS-025 (SOUP 故障風險)

---

### 2.4 軟體項目分解（Software Item Decomposition）

> 本節對應 IEC 62304:5.3.5 — 識別可分離的軟體項目

| 軟體項目 ID | 名稱 | 職責 | 對應 SR | Class | 測試策略 |
|-----------|------|------|---------|-------|---------|
| **SI-001** | 患者數據管理 | 患者資料 CRUD | SR-FUNC-001~003 | C | 完整單元+整合測試 |
| **SI-002** | 生命體徵監測 | 即時監測與警報 | SR-FUNC-010~015 | C | 實時性+容錯測試 |
| **SI-003** | 報警系統 | 警報生成與管理 | SR-ALARM-001~003 | C | 優先級+可靠性測試 |
| **SI-004** | 身份認證 | 使用者驗證與授權 | SR-SEC-001~003 | B | 安全+滲透測試 |
| **SI-005** | 數據存儲 | 持久化與備份 | SR-DATA-001~005 | B | 完整性+恢復測試 |
| **SI-006** | HL7 整合 | HIS 系統整合 | SR-SW-001 | B | 互通性測試 |
| **SI-007** | 報表生成 | 統計報表 | SR-FUNC-020 | A | 功能測試 |

---

## 3. 詳細設計（Detailed Design）

> 本節對應 IEC 62304:5.4 — 軟體詳細設計

### 3.1 模組詳細設計

#### 模組 DES-001: 患者數據管理模組

**模組 ID**: DES-001
**軟體項目**: SI-001
**追溯 SR**: SR-FUNC-001, SR-FUNC-002, SR-DATA-001

**職責**：
- 患者資料的建立、查詢、更新、刪除（CRUD）
- 數據驗證與完整性檢查
- 患者 ID 唯一性保證

**類別設計**：

```python
class PatientDataManager:
    """患者數據管理器 (IEC 62304:5.4.2)"""

    def __init__(self, db_connection: DatabaseConnection,
                 validator: DataValidator):
        """初始化患者數據管理器

        Args:
            db_connection: 數據庫連線（SOUP 隔離）
            validator: 數據驗證器
        """
        self._db = db_connection
        self._validator = validator
        self._logger = Logger("PatientDataManager")

    def create_patient(self, patient_data: PatientData) -> PatientID:
        """建立新患者記錄 (SR-FUNC-001)

        Args:
            patient_data: 患者資料對象

        Returns:
            PatientID: 新建立的患者 ID

        Raises:
            ValidationError: 數據驗證失敗
            DatabaseError: 數據庫操作失敗

        Safety:
            - 數據驗證（防止無效數據）
            - 交易機制（確保原子性）
            - 錯誤日誌記錄
        """
        # 1. 數據驗證
        if not self._validator.validate(patient_data):
            raise ValidationError("Invalid patient data")

        # 2. 檢查 ID 唯一性
        if self._db.patient_exists(patient_data.patient_id):
            raise DuplicatePatientError()

        # 3. 建立記錄（交易）
        try:
            with self._db.transaction():
                patient_id = self._db.insert_patient(patient_data)
                self._logger.info(f"Patient created: {patient_id}")
                return patient_id
        except DatabaseError as e:
            self._logger.error(f"Failed to create patient: {e}")
            raise

    def update_patient(self, patient_id: PatientID,
                       updated_data: PatientData) -> None:
        """更新患者記錄 (SR-FUNC-002)

        Safety:
            - 數據驗證
            - 審計日誌（記錄所有修改）
            - 樂觀鎖（防止併發衝突）
        """
        # 實施細節...
```

**介面設計**（IEC 62304:5.4.3）：

| 方法 | 輸入 | 輸出 | 錯誤處理 | SR ID |
|------|------|------|---------|-------|
| `create_patient` | `PatientData` | `PatientID` | ValidationError, DatabaseError | SR-FUNC-001 |
| `update_patient` | `PatientID`, `PatientData` | `None` | NotFoundError, ValidationError | SR-FUNC-002 |
| `delete_patient` | `PatientID` | `None` | NotFoundError, PermissionError | SR-FUNC-003 |
| `get_patient` | `PatientID` | `PatientData` | NotFoundError | SR-FUNC-004 |

**單元測試計畫**（IEC 62304:5.5.2）：
- 正常流程測試（有效數據）
- 邊界值測試（最小/最大值）
- 異常處理測試（無效數據、數據庫故障）
- 併發測試（多使用者同時操作）

---

#### 模組 DES-002: 生命體徵監測模組

**模組 ID**: DES-002
**軟體項目**: SI-002
**追溯 SR**: SR-FUNC-010, SR-ALARM-001, SR-DATA-001
**安全分級**: Class C（關鍵生命監測）

**職責**：
- 即時讀取生命體徵數據（心率、血壓、血氧等）
- 數據驗證與異常檢測
- 觸發警報（超出正常範圍）

**類別設計**：

```python
class VitalSignsMonitor:
    """生命體徵監測器 (IEC 62304:5.4.2, Class C)"""

    def __init__(self, sensor_driver: SensorDriver,
                 alarm_manager: AlarmManager,
                 data_validator: DataValidator):
        """初始化監測器

        Safety Design:
            - 感測器驅動隔離（SOUP isolation）
            - 冗餘檢測機制
            - 故障安全模式
        """
        self._sensor = sensor_driver
        self._alarm_mgr = alarm_manager
        self._validator = data_validator
        self._fail_safe_active = False

    def read_vital_signs(self) -> VitalSignsData:
        """讀取生命體徵數據 (SR-FUNC-010)

        Returns:
            VitalSignsData: 包含心率、血壓、血氧等

        Safety:
            - 數據範圍驗證（防止感測器錯誤）
            - 冗餘讀取（多次測量比對）
            - 故障檢測（感測器自檢）

        Risk Control:
            - Mitigates HS-001 (感測器數據錯誤)
            - IEC 62304:5.2.3 風險控制
        """
        # 1. 感測器自檢
        if not self._sensor.self_test():
            self._enter_fail_safe_mode()
            raise SensorFailureError()

        # 2. 讀取數據（冗餘讀取 3 次）
        readings = []
        for _ in range(3):
            data = self._sensor.read()
            if self._validator.validate_range(data):
                readings.append(data)

        # 3. 數據一致性檢查
        if len(readings) < 2:
            self._alarm_mgr.raise_alarm(AlarmLevel.HIGH,
                                       "Sensor data inconsistent")
            raise DataInconsistencyError()

        # 4. 取中位數（減少測量誤差）
        vital_signs = self._calculate_median(readings)

        # 5. 異常檢測
        if self._is_abnormal(vital_signs):
            self._alarm_mgr.raise_alarm(AlarmLevel.MEDIUM,
                                       f"Abnormal vital signs: {vital_signs}")

        return vital_signs

    def _enter_fail_safe_mode(self):
        """進入故障安全模式 (SR-RISK-001)

        Fail-Safe Behavior:
            - 停止所有自動化操作
            - 發出故障警報
            - 記錄故障事件
            - 保存當前狀態
        """
        self._fail_safe_active = True
        self._alarm_mgr.raise_alarm(AlarmLevel.CRITICAL,
                                   "System entering fail-safe mode")
        self._logger.critical("Fail-safe mode activated")
        # 停止所有可能造成傷害的操作...
```

**設計驗證**（IEC 62304:5.4.4）：
- [ ] 設計審查完成（架構師、醫療顧問）
- [ ] 故障模式分析（FMEA）
- [ ] 風險控制措施驗證
- [ ] 介面一致性檢查

---

### 3.2 數據模型設計

#### 患者數據模型

```python
@dataclass
class PatientData:
    """患者資料數據模型 (SR-DATA-001)"""

    patient_id: str  # 唯一識別碼（格式: P-XXXXXXXX）
    name: str  # 姓名
    date_of_birth: date  # 出生日期
    gender: Gender  # 性別（枚舉）
    blood_type: BloodType  # 血型
    allergies: List[str]  # 過敏史
    medical_history: List[MedicalRecord]  # 病史
    created_at: datetime  # 建立時間
    updated_at: datetime  # 最後更新時間

    def validate(self) -> bool:
        """數據驗證 (SR-DATA-001)

        Validation Rules:
            - patient_id 符合格式
            - name 非空且 ≤ 100 字元
            - date_of_birth 合理（年齡 0-150 歲）
            - 必填欄位不為空
        """
        # 驗證邏輯...
```

#### 生命體徵數據模型

```python
@dataclass
class VitalSignsData:
    """生命體徵數據模型 (SR-FUNC-010)"""

    timestamp: datetime  # 測量時間
    heart_rate: int  # 心率 (bpm)
    blood_pressure: BloodPressure  # 血壓 (mmHg)
    oxygen_saturation: float  # 血氧飽和度 (%)
    temperature: float  # 體溫 (°C)
    respiratory_rate: int  # 呼吸速率 (breaths/min)

    def is_within_normal_range(self) -> bool:
        """檢查是否在正常範圍 (SR-ALARM-001)

        Normal Ranges (成人):
            - heart_rate: 60-100 bpm
            - blood_pressure: SBP 90-140, DBP 60-90 mmHg
            - oxygen_saturation: ≥ 95%
            - temperature: 36.5-37.5°C
            - respiratory_rate: 12-20 breaths/min
        """
        # 檢查邏輯...
```

---

### 3.3 SOUP 整合設計

> 本節對應 IEC 62304:5.3.4, 7.1 — SOUP 隔離與整合

#### SOUP Wrapper: 數據庫存取層

```python
class DatabaseConnection:
    """數據庫連線 Wrapper (SOUP Isolation)

    Wraps: PostgreSQL (SOUP)
    Purpose: 隔離 SOUP 異常行為
    """

    def __init__(self, connection_string: str):
        self._engine = create_engine(connection_string)  # SQLAlchemy (SOUP)
        self._session_factory = sessionmaker(bind=self._engine)

    def execute_query(self, query: str) -> List[Dict]:
        """執行查詢（隔離 SOUP 異常）

        SOUP Error Handling:
            - OperationalError → DatabaseConnectionError
            - IntegrityError → DataIntegrityError
            - All other → GenericDatabaseError
        """
        try:
            session = self._session_factory()
            result = session.execute(text(query))
            return result.fetchall()
        except OperationalError as e:
            # SOUP 異常轉換
            raise DatabaseConnectionError(f"DB connection failed: {e}")
        except IntegrityError as e:
            raise DataIntegrityError(f"Data integrity violated: {e}")
        except Exception as e:
            # 未知 SOUP 異常
            raise GenericDatabaseError(f"Unexpected DB error: {e}")
        finally:
            session.close()
```

**SOUP 異常序列處理**（IEC 62304:7.1.3）：

| SOUP | 異常類型 | 轉換為 | 處理策略 | SR ID |
|------|---------|--------|---------|-------|
| PostgreSQL | OperationalError | DatabaseConnectionError | 重試 3 次，失敗則進入離線模式 | SR-REL-004 |
| PostgreSQL | IntegrityError | DataIntegrityError | 拒絕操作，記錄錯誤 | SR-DATA-002 |
| Redis | ConnectionError | CacheUnavailableError | 降級使用直接數據庫查詢 | SR-PERF-004 |
| OpenSSL | SSLError | EncryptionFailureError | 記錄錯誤，拒絕連線 | SR-SEC-004 |

---

## 4. 設計追溯矩陣（Design Traceability Matrix）

| SR ID | 設計 ID | 模組名稱 | 設計元素 | 驗證方式 | 狀態 |
|-------|---------|---------|---------|---------|------|
| SR-FUNC-001 | DES-001 | 患者數據管理 | `create_patient()` | 單元測試 | ✅ Verified |
| SR-FUNC-010 | DES-002 | 生命體徵監測 | `read_vital_signs()` | 整合測試 | 🔄 In Progress |
| SR-ALARM-001 | DES-005 | 報警系統 | `AlarmManager` | 系統測試 | ⏳ Pending |
| SR-SEC-001 | DES-010 | 身份認證 | `AuthenticationService` | 安全測試 | ✅ Verified |
| SR-DATA-001 | DES-001 | 數據驗證 | `DataValidator` | 單元測試 | ✅ Verified |

---

## 5. 設計審查記錄（Design Review）

### 5.1 架構設計審查（IEC 62304:5.3.6）

| 審查日期 | 審查人 | 審查範圍 | 發現問題 | 解決狀態 |
|---------|--------|---------|---------|---------|
| YYYY-MM-DD | [架構師] | 完整架構設計 | 3 Major, 5 Minor | All Resolved |
| YYYY-MM-DD | [風險管理負責人] | SOUP 隔離設計 | 2 Minor | Resolved |

### 5.2 詳細設計審查（IEC 62304:5.4.4）

| 審查日期 | 審查人 | 審查範圍 | 發現問題 | 解決狀態 |
|---------|--------|---------|---------|---------|
| YYYY-MM-DD | [開發主管] | DES-001~DES-005 | 8 Minor | All Resolved |
| YYYY-MM-DD | [品質負責人] | 介面設計 | 4 Minor | Resolved |

---

## 6. 附錄（Appendix）

### 6.1 設計模式應用

| 模式 | 應用位置 | 目的 |
|------|---------|------|
| **Wrapper Pattern** | SOUP 隔離 | 隔離第三方庫異常 |
| **Factory Pattern** | 物件創建 | 簡化依賴注入 |
| **Observer Pattern** | 報警系統 | 事件驅動通知 |
| **Strategy Pattern** | 數據驗證 | 可擴展驗證規則 |

### 6.2 參考文件

- IEC 62304:2006+AMD1:2015 — 條款 5.3, 5.4
- Software Requirements Specification: `software-requirements-spec-template.md`
- SOUP Management Plan: `soup-management-template.md`
- Risk Management File: `[文檔位置]`

---

**文檔版本**: v1.0.0
**維護人**: [姓名]
**最後審核**: YYYY-MM-DD
