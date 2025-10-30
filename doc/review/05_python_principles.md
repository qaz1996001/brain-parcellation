# Python 一般原則審查報告

## 審查標準
基於專案的 Python 一般原則進行評估

## 1. 命名慣例審查

### ❌ 違規案例
```python
# code_ai/pipeline/main.py
# 問題：不一致的命名風格
def GetDWIData():  # ❌ 駝峰命名
    pass

def process_WMH_data():  # ❌ 混合風格
    pass

isValid = True  # ❌ 駝峰變數名
MAX_size = 100  # ❌ 混合大小寫常數
```

### ✅ 正確的命名慣例
```python
# 變數和函數：小寫加底線
def get_dwi_data():
    """取得 DWI 資料"""
    pass

def process_wmh_data():
    """處理 WMH 資料"""
    pass

is_valid = True
has_permission = False
created_at = datetime.now()

# 常數：大寫加底線
MAX_FILE_SIZE = 100 * 1024 * 1024
DEFAULT_PAGE_SIZE = 20
API_VERSION = "v1"

# 私有變數和函數
_internal_cache = {}
_SECRET_KEY = "secret"

def _validate_internal_data(data: dict) -> bool:
    """內部驗證函數"""
    return bool(data)
```

## 2. 函數式編程審查

### ❌ 過度使用類別
```python
# 發現問題：不必要的類別封裝
class FileProcessor:
    """不必要的類別 - 應該用函數"""
    def __init__(self):
        pass
    
    def process(self, file_path):
        return self._read_file(file_path)
    
    def _read_file(self, path):
        with open(path) as f:
            return f.read()
```

### ✅ 函數式方法
```python
# 使用純函數
def read_file(file_path: Path) -> str:
    """讀取檔案內容"""
    with open(file_path) as f:
        return f.read()

def process_file(file_path: Path) -> ProcessedData:
    """處理檔案"""
    content = read_file(file_path)
    return process_content(content)

# 高階函數
from functools import partial
from typing import Callable

def create_processor(
    transform_func: Callable[[str], str]
) -> Callable[[Path], str]:
    """建立檔案處理器"""
    def processor(file_path: Path) -> str:
        content = read_file(file_path)
        return transform_func(content)
    return processor

# 使用
uppercase_processor = create_processor(str.upper)
lowercase_processor = create_processor(str.lower)
```

## 3. 類型註解審查

### ❌ 缺少類型註解
```python
# 當前問題：大量函數缺少類型提示
def process_image(image_path, options=None):  # ❌
    if options:
        return apply_options(image_path, options)
    return load_image(image_path)

def calculate_metrics(data):  # ❌
    return {
        "mean": sum(data) / len(data),
        "max": max(data)
    }
```

### ✅ 完整的類型註解
```python
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np

def process_image(
    image_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """處理醫學影像
    
    Args:
        image_path: 影像檔案路徑
        options: 處理選項
        
    Returns:
        處理後的影像陣列
    """
    if options:
        return apply_options(image_path, options)
    return load_image(image_path)

def calculate_metrics(data: List[float]) -> Dict[str, float]:
    """計算統計指標
    
    Args:
        data: 數值列表
        
    Returns:
        包含統計指標的字典
    """
    if not data:
        raise ValueError("資料列表不能為空")
    
    return {
        "mean": sum(data) / len(data),
        "max": max(data),
        "min": min(data),
        "count": len(data)
    }
```

## 4. RORO 模式審查

### ❌ 缺少 RORO 模式
```python
# 當前問題：參數和返回值混亂
def process_study(study_id, user_id, options, validate=True):
    # 處理邏輯...
    if error:
        return None, error_message
    return result, None
```

### ✅ RORO 模式實作
```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class StudyProcessingRequest:
    """Study 處理請求"""
    study_id: str
    user_id: int
    options: Optional[Dict[str, Any]] = None
    validate: bool = True

@dataclass
class StudyProcessingResult:
    """Study 處理結果"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

def process_study(request: StudyProcessingRequest) -> StudyProcessingResult:
    """處理 Study - RORO 模式"""
    # 驗證輸入
    if not request.study_id:
        return StudyProcessingResult(
            success=False,
            message="Study ID 不能為空",
            errors=["INVALID_STUDY_ID"]
        )
    
    if request.validate:
        validation_errors = validate_study(request.study_id)
        if validation_errors:
            return StudyProcessingResult(
                success=False,
                message="驗證失敗",
                errors=validation_errors
            )
    
    try:
        # 處理邏輯
        processed_data = perform_processing(
            request.study_id,
            request.options
        )
        
        return StudyProcessingResult(
            success=True,
            data=processed_data,
            message="處理成功"
        )
    
    except Exception as e:
        return StudyProcessingResult(
            success=False,
            message="處理失敗",
            errors=[str(e)]
        )
```

## 5. 模組化和重用審查

### ❌ 程式碼重複問題
```python
# 發現多處重複的驗證邏輯
def process_t1_image(path):
    if not path:
        raise ValueError("路徑不能為空")
    if not os.path.exists(path):
        raise ValueError("檔案不存在")
    if not path.endswith('.nii'):
        raise ValueError("不是 NIfTI 檔案")
    # 處理 T1...

def process_t2_image(path):
    if not path:
        raise ValueError("路徑不能為空")
    if not os.path.exists(path):
        raise ValueError("檔案不存在")
    if not path.endswith('.nii'):
        raise ValueError("不是 NIfTI 檔案")
    # 處理 T2...
```

### ✅ 提取共同邏輯
```python
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar('T')

def validate_nifti_file(path: Path) -> None:
    """驗證 NIfTI 檔案"""
    if not path:
        raise ValueError("路徑不能為空")
    
    if not path.exists():
        raise ValueError(f"檔案不存在: {path}")
    
    if not path.suffix in ['.nii', '.nii.gz']:
        raise ValueError(f"不是 NIfTI 檔案: {path}")

def with_nifti_validation(
    processor: Callable[[Path], T]
) -> Callable[[Path], T]:
    """添加 NIfTI 驗證的裝飾器"""
    def wrapper(path: Path) -> T:
        validate_nifti_file(path)
        return processor(path)
    return wrapper

@with_nifti_validation
def process_t1_image(path: Path) -> ProcessedImage:
    """處理 T1 影像"""
    return T1Processor().process(path)

@with_nifti_validation
def process_t2_image(path: Path) -> ProcessedImage:
    """處理 T2 影像"""
    return T2Processor().process(path)
```

## 6. 錯誤處理最佳實踐

### ❌ 不明確的錯誤處理
```python
# 問題：捕獲所有異常且不處理
try:
    result = process_data(data)
except:  # ❌ 裸露的 except
    return None

try:
    value = int(user_input)
except Exception as e:  # ❌ 太廣泛
    print(f"Error: {e}")
```

### ✅ 明確的錯誤處理
```python
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def parse_user_age(age_str: str) -> Optional[int]:
    """解析使用者年齡"""
    try:
        age = int(age_str)
        if not 0 <= age <= 150:
            logger.warning(f"年齡超出合理範圍: {age}")
            return None
        return age
    except ValueError:
        logger.warning(f"無法解析年齡: {age_str}")
        return None
    except TypeError:
        logger.error(f"年齡類型錯誤: {type(age_str)}")
        return None

def process_medical_data(data: Dict[str, Any]) -> ProcessingResult:
    """處理醫學資料"""
    if not data:
        raise ValueError("資料不能為空")
    
    if "patient_id" not in data:
        raise KeyError("缺少必要欄位: patient_id")
    
    try:
        validated_data = validate_medical_data(data)
        processed = perform_processing(validated_data)
        return ProcessingResult.success(processed)
    
    except ValidationError as e:
        logger.warning(f"資料驗證失敗: {e}")
        return ProcessingResult.validation_error(str(e))
    
    except ProcessingError as e:
        logger.error(f"處理失敗: {e}")
        return ProcessingResult.processing_error(str(e))
```

## 7. 檔案和目錄結構

### ❌ 不一致的命名
```
code_ai/
├── DICOM2NII/      # ❌ 大寫目錄
├── Utils/          # ❌ 首字母大寫
├── task_Pipeline.py  # ❌ 混合命名
└── ProcessData.py    # ❌ 駝峰命名
```

### ✅ 正確的結構
```
code_ai/
├── dicom2nii/      # ✅ 小寫加底線
├── utils/          # ✅
├── task_pipeline.py  # ✅
└── process_data.py   # ✅
```

## Python 原則評分

| 類別 | 評分 | 說明 |
|-----|------|------|
| 命名慣例 | 4/10 | 不一致的命名風格 |
| 函數式編程 | 3/10 | 過度使用類別和狀態 |
| 類型註解 | 2/10 | 大部分函數缺少類型提示 |
| RORO 模式 | 1/10 | 幾乎沒有使用 |
| 模組化 | 3/10 | 大量程式碼重複 |
| 錯誤處理 | 3/10 | 不明確的異常處理 |
| 檔案結構 | 5/10 | 部分符合但不一致 |

## 總體評分：3.0/10

## 改進建議

### 第一階段（1週）
1. 統一命名慣例
2. 添加基本類型註解
3. 修復明顯的錯誤處理問題

### 第二階段（2週）
1. 重構為函數式風格
2. 實施 RORO 模式
3. 提取重複程式碼

### 第三階段（1個月）
1. 完整的類型覆蓋
2. 實施進階模式（裝飾器、高階函數）
3. 完善錯誤處理策略

## 程式碼品質工具建議

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.black]
line-length = 88
target-version = ['py39']

[tool.ruff]
line-length = 88
select = ["E", "W", "F", "I", "N", "UP", "B", "C4"]
ignore = []

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
```
