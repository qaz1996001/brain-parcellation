# 重構路線圖

## 概述
本文件提供詳細的重構計劃，將現有系統轉變為符合所有程式碼標準的高品質軟體。

## 重構原則
1. **漸進式重構**：避免大規模重寫，逐步改善
2. **測試驅動**：每次重構前先建立測試
3. **向後相容**：保持 API 相容性
4. **持續部署**：小步快跑，頻繁部署

## 階段一：緊急修復（第1週）

### 目標
修復致命的安全和配置問題

### 任務清單

#### 1.1 資料庫配置安全化
```python
# 建立 .env 檔案
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=${SECRET_PASSWORD}
DB_DATABASE=dicom

# 建立 config/database.py
from pydantic import BaseSettings, SecretStr

class DatabaseSettings(BaseSettings):
    host: str
    port: int = 5432
    username: str
    password: SecretStr
    database: str
    
    class Config:
        env_prefix = "DB_"
        env_file = ".env"
```

#### 1.2 修復 CORS 配置
```python
# backend/app/config/security.py
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://your-domain.com"
]

# backend/app/server.py
from .config.security import ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
```

#### 1.3 添加基本錯誤處理
```python
# backend/app/middleware/error_handler.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "伺服器內部錯誤",
            "path": str(request.url)
        }
    )

# 註冊處理器
app.add_exception_handler(Exception, global_exception_handler)
```

### 驗收標準
- [ ] 所有密碼從環境變數讀取
- [ ] CORS 只允許特定來源
- [ ] 所有異常都有適當處理
- [ ] 通過安全掃描

## 階段二：架構重構（第2-4週）

### 目標
將單體架構拆分為模組化設計

### 2.1 重構 pipeline/main.py

#### 現有結構
```
main.py (758行)
├── 參數解析
├── WMH 處理
├── CMB 處理
├── DWI 處理
└── 各種工具函數
```

#### 目標結構
```
pipeline/
├── __init__.py
├── config.py           # 配置管理
├── cli.py             # 命令列介面
├── processors/
│   ├── __init__.py
│   ├── base.py       # 基礎處理器
│   ├── wmh.py        # WMH 處理器
│   ├── cmb.py        # CMB 處理器
│   └── dwi.py        # DWI 處理器
├── utils/
│   ├── __init__.py
│   ├── file_utils.py
│   └── image_utils.py
└── main.py           # 入口點（< 50行）
```

#### 實作範例
```python
# processors/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

class BaseProcessor(ABC):
    """基礎處理器抽象類別"""
    
    @abstractmethod
    def validate_input(self, path: Path) -> bool:
        """驗證輸入"""
        pass
    
    @abstractmethod
    def process(self, path: Path, options: Dict[str, Any]) -> ProcessingResult:
        """處理邏輯"""
        pass
    
    @abstractmethod
    def save_output(self, result: ProcessingResult, output_path: Path) -> None:
        """儲存結果"""
        pass

# processors/wmh.py
class WMHProcessor(BaseProcessor):
    """WMH 處理器"""
    
    def validate_input(self, path: Path) -> bool:
        if not path.exists():
            raise FileNotFoundError(f"檔案不存在: {path}")
        
        if not path.suffix in ['.nii', '.nii.gz']:
            raise ValueError(f"不支援的檔案格式: {path.suffix}")
        
        return True
    
    def process(self, path: Path, options: Dict[str, Any]) -> ProcessingResult:
        # 實作 WMH 處理邏輯
        pass
```

### 2.2 實施策略模式

```python
# pipeline/factory.py
from typing import Dict, Type
from .processors.base import BaseProcessor
from .processors.wmh import WMHProcessor
from .processors.cmb import CMBProcessor
from .processors.dwi import DWIProcessor

class ProcessorFactory:
    """處理器工廠"""
    
    _processors: Dict[str, Type[BaseProcessor]] = {
        'WMH': WMHProcessor,
        'CMB': CMBProcessor,
        'DWI': DWIProcessor,
    }
    
    @classmethod
    def create(cls, processor_type: str) -> BaseProcessor:
        """建立處理器實例"""
        processor_class = cls._processors.get(processor_type)
        if not processor_class:
            raise ValueError(f"不支援的處理器類型: {processor_type}")
        return processor_class()

# 使用
processor = ProcessorFactory.create('WMH')
result = processor.process(input_path, options)
```

## 階段三：程式碼品質提升（第5-8週）

### 3.1 類型註解覆蓋

```bash
# 使用 mypy 檢查類型覆蓋率
uv run mypy --html-report mypy-report code_ai/
uv run mypy --html-report mypy-report backend/
```

### 3.2 測試覆蓋率提升

```python
# tests/test_processors.py
import pytest
from pathlib import Path
from code_ai.pipeline.processors.wmh import WMHProcessor

class TestWMHProcessor:
    @pytest.fixture
    def processor(self):
        return WMHProcessor()
    
    def test_validate_input_valid_file(self, processor, tmp_path):
        # 建立測試檔案
        test_file = tmp_path / "test.nii"
        test_file.touch()
        
        assert processor.validate_input(test_file) is True
    
    def test_validate_input_missing_file(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.validate_input(Path("nonexistent.nii"))
```

### 3.3 文檔完善

```python
# 使用 Sphinx 自動生成文檔
"""
WMH 處理模組

本模組提供白質高信號（WMH）檢測和量化功能。

Example:
    基本使用範例::
    
        from code_ai.pipeline.processors.wmh import WMHProcessor
        
        processor = WMHProcessor()
        result = processor.process(
            path=Path("path/to/image.nii"),
            options={"threshold": 0.5}
        )

Attributes:
    DEFAULT_THRESHOLD (float): 預設閾值 (0.3)
    SUPPORTED_FORMATS (list): 支援的檔案格式
"""
```

## 階段四：效能最佳化（第9-12週）

### 4.1 非同步化改造

```python
# 改造前
def process_multiple_files(files):
    results = []
    for file in files:
        result = process_file(file)  # 同步處理
        results.append(result)
    return results

# 改造後
async def process_multiple_files(files: List[Path]) -> List[ProcessingResult]:
    """非同步批次處理"""
    tasks = [process_file_async(file) for file in files]
    results = await asyncio.gather(*tasks)
    return results
```

### 4.2 快取策略實施

```python
# cache/manager.py
from functools import lru_cache
import hashlib

class CacheManager:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_or_compute(
        self,
        key: str,
        compute_func,
        ttl: int = 3600
    ):
        # 檢查快取
        cached = await self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        
        # 計算並快取
        result = await compute_func()
        await self.redis.setex(
            key,
            ttl,
            pickle.dumps(result)
        )
        return result
```

## 階段五：監控和維護（持續）

### 5.1 監控指標

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 定義指標
processing_requests = Counter(
    'processing_requests_total',
    'Total processing requests',
    ['processor_type', 'status']
)

processing_duration = Histogram(
    'processing_duration_seconds',
    'Processing duration in seconds',
    ['processor_type']
)

active_connections = Gauge(
    'database_connections_active',
    'Active database connections'
)
```

### 5.2 健康檢查

```python
# health/checks.py
from typing import Dict, Any

async def health_check() -> Dict[str, Any]:
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage(),
    }
    
    overall_status = "healthy" if all(
        check["status"] == "healthy"
        for check in checks.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

## 成功指標

### 程式碼品質
- [ ] 縮排不超過 3 層：95% 檔案符合
- [ ] 類型註解覆蓋率 > 80%
- [ ] 測試覆蓋率 > 70%
- [ ] 無 Linus 風格致命缺陷

### 效能
- [ ] API 回應時間 < 200ms (P95)
- [ ] 資料庫查詢時間 < 50ms (P95)
- [ ] 並發處理能力 > 100 req/s

### 可維護性
- [ ] 文檔覆蓋率 > 60%
- [ ] 模組耦合度 < 0.3
- [ ] 循環複雜度 < 10

## 風險管理

### 風險項目
1. **資料遺失風險**：重構過程可能影響資料完整性
   - 緩解：完整備份、漸進式遷移
   
2. **服務中斷風險**：部署新版本可能導致服務不可用
   - 緩解：藍綠部署、回滾機制
   
3. **相容性風險**：新版本可能破壞現有 API
   - 緩解：版本控制、棄用通知

## 時間表

| 階段 | 時間 | 主要任務 | 負責人 |
|-----|------|---------|--------|
| 階段一 | 第1週 | 安全修復 | DevOps |
| 階段二 | 第2-4週 | 架構重構 | 架構師 |
| 階段三 | 第5-8週 | 品質提升 | 開發團隊 |
| 階段四 | 第9-12週 | 效能最佳化 | 效能團隊 |
| 階段五 | 持續 | 監控維護 | SRE |

## 結論
通過系統性的重構計劃，我們將在 3 個月內將系統品質從目前的 3/10 提升到 8/10，達到生產環境標準。
