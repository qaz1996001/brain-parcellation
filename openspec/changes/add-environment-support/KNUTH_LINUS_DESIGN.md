# 環境變數架構設計 - Knuth & Linus 哲學

## 設計哲學基礎

### Donald Knuth 原則
1. **精確性 (Precision)**: 每個配置項都有明確的型別、邊界和驗證規則
2. **文學式程式設計 (Literate Programming)**: 配置即文檔，自我解釋
3. **最佳化的正確性 (Correctness First)**: 先確保正確性，再考慮效能
4. **數學般的嚴謹 (Mathematical Rigor)**: 配置載入過程可證明正確

### Linus Torvalds 原則
1. **資料結構優先 (Data First)**: 好的資料結構使程式碼自然簡單
2. **品味 (Good Taste)**: 消除特殊情況，統一處理邏輯
3. **務實主義 (Pragmatism)**: 解決實際問題，不過度設計
4. **由下而上 (Bottom-Up)**: 從基礎元件向上構建

---

## 架構設計：三層配置分區

### 核心設計理念

**Linus: "資料結構決定一切"**
```
好的配置架構 = 清晰的資料結構 + 簡單的存取邏輯
```

**Knuth: "過早優化是萬惡之源，但好的結構不是優化"**
```
正確的分層 → 清晰的邊界 → 簡單的實作
```

### 配置分區設計

```
配置層次架構 (Hierarchical Configuration)
│
├─ 🌍 Common (通用配置)
│   ├─ ENV (production/testing)
│   ├─ LOG_LEVEL
│   └─ REDIS_* (跨系統共享)
│
├─ 🔧 Backend (後端專用)
│   ├─ AI_APP_* (FastAPI 配置)
│   ├─ DATABASE_URL
│   └─ API Endpoints
│
└─ 🧠 Code_AI (推理專用)
    ├─ PATH_* (推理路徑)
    ├─ GPU_N
    └─ Model Configs
```

**設計理由 (Knuth 精確性):**
- **明確邊界**: 每層配置有清晰的責任範圍
- **最小耦合**: Backend 不需要知道 GPU_N，Code_AI 不需要知道 DATABASE_URL
- **可測試性**: 每層可獨立測試和驗證

---

## 配置格式選擇：TOML (推薦)

### 格式比較分析

| 格式 | 優點 | 缺點 | Knuth 評分 | Linus 評分 |
|------|------|------|------------|-----------|
| **.env** | 簡單、Twelve-Factor | 無型別、無結構 | 6/10 | 7/10 |
| **YAML** | 人類可讀、豐富結構 | 複雜、易出錯 | 7/10 | 6/10 |
| **TOML** | 清晰、型別安全、分區明確 | 語法學習成本 | 9/10 | 9/10 |

**選擇 TOML 的理由:**

**Knuth 觀點:**
- ✅ 型別明確 (字串、數字、布林清晰區分)
- ✅ 結構清晰 (分區 `[section]` 自我文檔化)
- ✅ 註解豐富 (支援多行註解說明)

**Linus 觀點:**
- ✅ 簡單直觀 (比 YAML 少特殊情況)
- ✅ 資料結構清晰 (表 = 資料結構)
- ✅ Git 友好 (diff 清晰易讀)

### 配置檔案結構設計

```toml
# config.toml
# 遵循 Knuth 文學式程式設計：配置即文檔

# =============================================================================
# COMMON CONFIGURATION (通用配置)
# Linus: 資料結構優先 - 所有系統共享的基礎配置
# =============================================================================

[common]
# 環境選擇：production 或 testing
# Knuth: 明確的邊界條件，僅允許兩個值
env = "production"  # 預設值，符合安全原則

# 日誌級別：根據環境自動調整
# production: INFO, testing: DEBUG
log_level = "INFO"

[common.redis]
# Redis 共享快取配置
# Linus: 將相關資料組織在一起
host = "127.0.0.1"
port = 6379
username = ""
password = ""
db_fastapi_cache = 6

# =============================================================================
# BACKEND CONFIGURATION (後端配置)
# 遵循單一職責原則：僅包含 FastAPI 後端所需配置
# =============================================================================

[backend]
# FastAPI 應用配置
# Knuth: 精確的型別和預設值
app_title = "SHH AI API"
app_description = "API FOR SHH AI"
app_version = "1.0.0"
app_port = 8000

[backend.database]
# 資料庫連接配置
# Linus: 資料結構決定連接邏輯
connection_string = "postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom"

# 環境特定的資料庫名稱
# Knuth: 明確的環境隔離規則
db_name_production = "minio_backup"
db_name_testing = "minio_backup_testing"

[backend.api]
# 外部 API 端點配置
# Linus: 統一的 API 配置管理
upload_data_url = "http://example.com/api"
upload_dicom_seg_url = "http://example.com/dicom-seg"

[backend.api.orthanc]
# Orthanc DICOM 伺服器認證
# Knuth: 安全敏感資訊明確標記
username = "orthanc"
password = "orthanc"

# =============================================================================
# CODE_AI CONFIGURATION (推理引擎配置)
# 遵循 Linus 務實主義：解決實際推理任務的配置需求
# =============================================================================

[code_ai]
# Python 環境配置
python3_path = "/usr/bin/python3"
fsl_flirt_path = "/usr/local/fsl/bin/flirt"
local_db = "database.sqlite3"

# GPU 配置
# Knuth: 明確的數值範圍和預設值
gpu_number = 0  # 預設 GPU 編號，範圍 0-7

[code_ai.paths]
# 推理任務路徑配置
# Linus: 將路徑作為資料結構統一管理
# Knuth: 每個路徑都有明確的用途說明

# 原始 DICOM 輸入路徑
path_raw_dicom = "/data/raw_dicom"

# 重新命名後的 DICOM 路徑
path_rename_dicom = "/data/renamed_dicom"

# 重新命名後的 NIfTI 路徑
path_rename_nifti = "/data/renamed_nifti"

# 推理處理工作目錄
path_process = "/data/process"

# JSON 結果輸出路徑
path_json = "/data/json"

# 日誌檔案路徑
path_log = "/data/logs"

# 程式碼根目錄
path_code = "/app/code_ai"

# SynthSeg 模型路徑
path_synthseg = "/models/synthseg"

# FreeSurfer 安裝路徑
path_freesurfer = "/usr/local/freesurfer"

# ROOT 根路徑（向後相容）
path_root = "/data"

[code_ai.models]
# 模型配置路徑
# Knuth: 環境特定的模型版本管理
config_production = "/models/production/config.yaml"
config_testing = "/models/testing/config.yaml"

[code_ai.tensorflow]
# TensorFlow 特定配置
# Linus: 消除執行時環境變數設置的副作用
cpp_min_log_level = "3"  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
enable_auto_mixed_precision = false

[code_ai.dicom]
# DICOM 處理相關配置
# Knuth: 群組 ID 的明確定義和用途
group_id_cmb = 44
group_id_default = 44

# =============================================================================
# ENVIRONMENT OVERRIDES (環境特定覆蓋)
# Knuth: 精確的環境差異定義
# Linus: 消除重複，僅定義差異部分
# =============================================================================

[production]
# Production 環境特定配置
log_level = "INFO"
debug = false

[production.code_ai.paths]
# Production 路徑覆蓋
path_raw_dicom = "/mnt/production/raw_dicom"
path_rename_dicom = "/mnt/production/renamed_dicom"
path_rename_nifti = "/mnt/production/renamed_nifti"
path_process = "/mnt/production/process"

[testing]
# Testing 環境特定配置
log_level = "DEBUG"
debug = true

[testing.code_ai.paths]
# Testing 路徑覆蓋（與 production 完全隔離）
path_raw_dicom = "/mnt/testing/raw_dicom"
path_rename_dicom = "/mnt/testing/renamed_dicom"
path_rename_nifti = "/mnt/testing/renamed_nifti"
path_process = "/mnt/testing/process"
```

---

## 配置載入架構設計

### Knuth: 嚴謹的載入流程

```python
"""
配置載入器設計
遵循 Knuth 的精確性原則：每一步都可驗證和證明正確

載入流程：
1. 檢測環境 (ENV 環境變數)
2. 載入基礎配置 (config.toml)
3. 應用環境覆蓋 (production/testing section)
4. 驗證配置完整性
5. 凍結配置物件 (不可變)
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import tomli  # Python 3.11+ 內建 tomllib
from pathlib import Path
import os

Environment = Literal["production", "testing"]

@dataclass(frozen=True)  # Knuth: 不可變性保證
class CommonConfig:
    """通用配置 - 所有系統共享"""
    env: Environment
    log_level: str
    redis_host: str
    redis_port: int
    redis_username: str
    redis_password: str
    redis_db_fastapi_cache: int

    def __post_init__(self):
        """Knuth: 建構後驗證，確保配置正確性"""
        assert self.env in ("production", "testing"), f"Invalid env: {self.env}"
        assert self.log_level in ("DEBUG", "INFO", "WARNING", "ERROR")
        assert 1 <= self.redis_port <= 65535, f"Invalid port: {self.redis_port}"

@dataclass(frozen=True)
class BackendConfig:
    """Backend 專用配置"""
    app_title: str
    app_description: str
    app_version: str
    app_port: int
    database_connection_string: str
    database_name: str  # 根據環境動態選擇
    upload_data_url: str
    upload_dicom_seg_url: str
    orthanc_username: str
    orthanc_password: str

    def __post_init__(self):
        """Knuth: 驗證 Backend 配置"""
        assert 1 <= self.app_port <= 65535
        assert self.database_name.strip() != ""
        assert self.upload_data_url.startswith("http")

@dataclass(frozen=True)
class CodeAIPathConfig:
    """Code_AI 路徑配置 - Linus: 資料結構優先"""
    path_raw_dicom: Path
    path_rename_dicom: Path
    path_rename_nifti: Path
    path_process: Path
    path_json: Path
    path_log: Path
    path_code: Path
    path_synthseg: Path
    path_freesurfer: Path
    path_root: Path

    def __post_init__(self):
        """Knuth: 路徑驗證"""
        # 確保所有路徑都是 Path 物件
        for field_name, field_value in self.__dataclass_fields__.items():
            path = getattr(self, field_name)
            assert isinstance(path, Path), f"{field_name} must be Path"

@dataclass(frozen=True)
class CodeAIConfig:
    """Code_AI 專用配置"""
    python3_path: str
    fsl_flirt_path: str
    local_db: str
    gpu_number: int
    paths: CodeAIPathConfig
    model_config_production: str
    model_config_testing: str
    tf_cpp_min_log_level: str
    tf_enable_auto_mixed_precision: bool
    dicom_group_id_cmb: int
    dicom_group_id_default: int

    def __post_init__(self):
        """Knuth: Code_AI 配置驗證"""
        assert 0 <= self.gpu_number <= 7, f"Invalid GPU: {self.gpu_number}"
        assert self.tf_cpp_min_log_level in ("0", "1", "2", "3")

@dataclass(frozen=True)
class AppConfig:
    """
    應用全域配置

    Linus: 好的資料結構使程式碼自然簡單
    Knuth: 明確的型別和不可變性保證正確性
    """
    common: CommonConfig
    backend: BackendConfig
    code_ai: CodeAIConfig

    @classmethod
    def from_toml(cls, config_path: Path) -> "AppConfig":
        """
        從 TOML 檔案載入配置

        Knuth: 可證明正確的載入流程
        1. 讀取 TOML
        2. 檢測環境
        3. 合併基礎配置和環境覆蓋
        4. 驗證完整性
        5. 建立不可變物件

        Args:
            config_path: TOML 配置檔案路徑

        Returns:
            AppConfig: 驗證過的不可變配置物件

        Raises:
            FileNotFoundError: 配置檔案不存在
            ValueError: 配置驗證失敗
        """
        # Step 1: 讀取 TOML
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "rb") as f:
            toml_data = tomli.load(f)

        # Step 2: 檢測環境
        env = os.getenv("ENV", toml_data["common"]["env"])
        if env not in ("production", "testing"):
            raise ValueError(f"Invalid ENV: {env}. Must be production or testing")

        # Step 3: 合併基礎配置和環境覆蓋
        # Linus: 消除重複，統一處理邏輯
        base_config = toml_data
        env_overrides = toml_data.get(env, {})

        # 深度合併環境覆蓋
        def deep_merge(base: dict, override: dict) -> dict:
            """Linus: 簡單的合併邏輯，無特殊情況"""
            result = base.copy()
            for key, value in override.items():
                if isinstance(value, dict) and key in result:
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged = deep_merge(base_config, env_overrides)

        # Step 4: 建立配置物件
        common_cfg = CommonConfig(
            env=env,
            log_level=merged["common"]["log_level"],
            redis_host=merged["common"]["redis"]["host"],
            redis_port=merged["common"]["redis"]["port"],
            redis_username=merged["common"]["redis"]["username"],
            redis_password=merged["common"]["redis"]["password"],
            redis_db_fastapi_cache=merged["common"]["redis"]["db_fastapi_cache"],
        )

        # 根據環境選擇資料庫名稱
        db_name_key = f"db_name_{env}"
        db_name = merged["backend"]["database"][db_name_key]

        backend_cfg = BackendConfig(
            app_title=merged["backend"]["app_title"],
            app_description=merged["backend"]["app_description"],
            app_version=merged["backend"]["app_version"],
            app_port=merged["backend"]["app_port"],
            database_connection_string=merged["backend"]["database"]["connection_string"],
            database_name=db_name,
            upload_data_url=merged["backend"]["api"]["upload_data_url"],
            upload_dicom_seg_url=merged["backend"]["api"]["upload_dicom_seg_url"],
            orthanc_username=merged["backend"]["api"]["orthanc"]["username"],
            orthanc_password=merged["backend"]["api"]["orthanc"]["password"],
        )

        # Linus: 路徑作為資料結構
        path_cfg = CodeAIPathConfig(
            path_raw_dicom=Path(merged["code_ai"]["paths"]["path_raw_dicom"]),
            path_rename_dicom=Path(merged["code_ai"]["paths"]["path_rename_dicom"]),
            path_rename_nifti=Path(merged["code_ai"]["paths"]["path_rename_nifti"]),
            path_process=Path(merged["code_ai"]["paths"]["path_process"]),
            path_json=Path(merged["code_ai"]["paths"]["path_json"]),
            path_log=Path(merged["code_ai"]["paths"]["path_log"]),
            path_code=Path(merged["code_ai"]["paths"]["path_code"]),
            path_synthseg=Path(merged["code_ai"]["paths"]["path_synthseg"]),
            path_freesurfer=Path(merged["code_ai"]["paths"]["path_freesurfer"]),
            path_root=Path(merged["code_ai"]["paths"]["path_root"]),
        )

        code_ai_cfg = CodeAIConfig(
            python3_path=merged["code_ai"]["python3_path"],
            fsl_flirt_path=merged["code_ai"]["fsl_flirt_path"],
            local_db=merged["code_ai"]["local_db"],
            gpu_number=merged["code_ai"]["gpu_number"],
            paths=path_cfg,
            model_config_production=merged["code_ai"]["models"]["config_production"],
            model_config_testing=merged["code_ai"]["models"]["config_testing"],
            tf_cpp_min_log_level=merged["code_ai"]["tensorflow"]["cpp_min_log_level"],
            tf_enable_auto_mixed_precision=merged["code_ai"]["tensorflow"]["enable_auto_mixed_precision"],
            dicom_group_id_cmb=merged["code_ai"]["dicom"]["group_id_cmb"],
            dicom_group_id_default=merged["code_ai"]["dicom"]["group_id_default"],
        )

        # Step 5: 返回不可變配置
        return cls(
            common=common_cfg,
            backend=backend_cfg,
            code_ai=code_ai_cfg
        )

    def get_model_config_path(self) -> str:
        """
        根據環境返回對應的模型配置路徑

        Linus: Good Taste - 無 if/else 特殊情況
        """
        config_map = {
            "production": self.code_ai.model_config_production,
            "testing": self.code_ai.model_config_testing,
        }
        return config_map[self.common.env]
```

---

## 配置初始化策略

### 應用啟動時配置載入 (ONE TIME)

**Knuth: 明確的初始化順序**

```python
# backend/app/main.py
"""
Backend 應用入口
遵循 Knuth 精確性：明確的初始化順序
"""
import logging
from pathlib import Path
from backend.app.config import AppConfig

# ============================================================================
# STEP 1: 載入配置（唯一的環境變數讀取點）
# Linus: 資料優先 - 先建立好的資料結構
# ============================================================================

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
app_config = AppConfig.from_toml(CONFIG_PATH)

# ============================================================================
# STEP 2: 配置日誌系統
# Knuth: 根據配置精確設定
# ============================================================================

logging.basicConfig(
    level=app_config.common.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 3: 記錄環境資訊（審計需求）
# ============================================================================

logger.info(f"=" * 80)
logger.info(f"Environment: {app_config.common.env}")
logger.info(f"Log Level: {app_config.common.log_level}")
logger.info(f"Backend Port: {app_config.backend.app_port}")
logger.info(f"Database: {app_config.backend.database_name}")
logger.info(f"=" * 80)

# ============================================================================
# STEP 4: 初始化 FastAPI 應用
# Linus: 由下而上構建 - 配置 → 組件 → 應用
# ============================================================================

from fastapi import FastAPI

app = FastAPI(
    title=app_config.backend.app_title,
    description=app_config.backend.app_description,
    version=app_config.backend.app_version
)

# ============================================================================
# STEP 5: 依賴注入配置到服務層
# Knuth: 明確的依賴關係
# ============================================================================

from backend.app.sync.service import SyncService
from backend.app.database import get_session_manager

session_manager = get_session_manager(app_config.backend.database_connection_string)

# 注入配置到服務
sync_service = SyncService(
    session_manager=session_manager,
    api_config=app_config.backend,  # 注入 Backend 配置
    path_config=app_config.code_ai.paths  # 注入路徑配置
)

# ============================================================================
# STEP 6: 註冊路由
# ============================================================================

from backend.app.sync.router import router as sync_router
app.include_router(sync_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=app_config.backend.app_port)
```

**Code_AI 初始化:**

```python
# code_ai/__init__.py
"""
Code_AI 推理引擎入口
"""
import os
from pathlib import Path
from backend.app.config import AppConfig

# ============================================================================
# 載入配置（與 Backend 共享同一配置檔案）
# Linus: 單一真相來源 - config.toml
# ============================================================================

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
pipeline_config = AppConfig.from_toml(CONFIG_PATH)

# ============================================================================
# 設定 TensorFlow 環境（模組層級，但基於配置）
# Knuth: 明確的副作用控制
# ============================================================================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = pipeline_config.code_ai.tf_cpp_min_log_level

if pipeline_config.code_ai.tf_enable_auto_mixed_precision:
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# ============================================================================
# 導出配置供模組使用
# ============================================================================

__all__ = ['pipeline_config']
```

---

## 環境變數 → 配置檔案遷移策略

### Linus: 務實的漸進式遷移

**階段 1: 雙軌並行（向後相容）**

```python
def get_path_with_fallback(config: AppConfig, env_var_name: str) -> Path:
    """
    漸進式遷移輔助函數
    優先使用配置檔案，fallback 到環境變數

    Linus: 務實主義 - 不破壞現有系統
    """
    # 嘗試從配置獲取
    path_map = {
        "PATH_RAW_DICOM": config.code_ai.paths.path_raw_dicom,
        "PATH_RENAME_DICOM": config.code_ai.paths.path_rename_dicom,
        # ... 其他路徑
    }

    if env_var_name in path_map:
        return path_map[env_var_name]

    # Fallback 到環境變數（向後相容）
    env_value = os.getenv(env_var_name)
    if env_value:
        logger.warning(f"Using deprecated env var: {env_var_name}. Please migrate to config.toml")
        return Path(env_value)

    raise ValueError(f"Path not found in config or environment: {env_var_name}")
```

**階段 2: 棄用警告**

```python
# 在所有使用環境變數的地方添加警告
import warnings

def deprecated_env_var(env_var_name: str):
    warnings.warn(
        f"Environment variable {env_var_name} is deprecated. "
        f"Please migrate to config.toml",
        DeprecationWarning,
        stacklevel=2
    )
```

**階段 3: 完全移除環境變數依賴**

---

## .env 檔案作為補充（開發便利性）

**Knuth: 精確的用途定義**

`.env` 檔案僅用於：
1. 本地開發環境覆蓋
2. 敏感資訊（不提交到 Git）
3. 開發者個人偏好設定

```bash
# .env (本地開發，不提交 Git)
# 覆蓋 config.toml 的預設值

# 環境選擇
ENV=testing

# 本地開發路徑
PATH_RAW_DICOM=/Users/developer/data/raw_dicom
PATH_PROCESS=/Users/developer/data/process

# 敏感資訊
ORTHANC_PASSWORD=my_secret_password
```

**載入優先級 (Knuth: 明確的優先級規則):**

```
環境變數 (最高優先級)
    ↓
.env 檔案
    ↓
config.toml 環境覆蓋 ([production]/[testing])
    ↓
config.toml 基礎配置
```

---

## 總結：Knuth & Linus 統一的設計

### Knuth 貢獻
- ✅ 精確的型別和驗證
- ✅ 明確的邊界條件
- ✅ 文學式配置（自我文檔化）
- ✅ 可證明的載入流程

### Linus 貢獻
- ✅ 資料結構優先（TOML 分層結構）
- ✅ 消除特殊情況（統一的配置邏輯）
- ✅ 務實的遷移策略（向後相容）
- ✅ 由下而上構建（配置 → 組件 → 應用）

### 設計優勢
1. **單一真相來源**: `config.toml` 是所有配置的中心
2. **環境隔離**: Production/Testing 完全分離
3. **分層清晰**: Common/Backend/Code_AI 職責明確
4. **可測試性**: 直接注入測試配置，無需 mock
5. **可維護性**: 配置即文檔，Git 友好
6. **向後相容**: 漸進式遷移，不破壞現有系統