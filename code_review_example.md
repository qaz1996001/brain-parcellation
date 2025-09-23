# 程式碼審查範例 - 根據 Cursor 規則

本文件展示如何根據 `.cursor/rules/` 下的規則審查程式碼並提出改進建議。

## 1. 違反 Linus 風格的程式碼範例

### 🔴 問題：深度嵌套（code_ai/pipeline/main.py）

```python
# ❌ 垃圾程式碼 - 6層嵌套
def process_files(root_path, options):
    if os.path.exists(root_path):
        if os.path.isdir(root_path):
            for folder in os.listdir(root_path):
                folder_path = os.path.join(root_path, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith('.nii.gz'):
                            if 'T2_FLAIR' in file:
                                if options.get('WMH'):
                                    # 處理邏輯...
                                    process_wmh(file)
```

### ✅ 修復：早期返回，扁平結構

```python
# ✅ Good Taste - 最多3層嵌套
def process_files(root_path: Path, options: ProcessingOptions) -> List[ProcessResult]:
    """處理檔案 - 扁平結構，早期返回"""
    if not root_path.exists():
        raise ValueError(f"路徑不存在: {root_path}")
    
    if not root_path.is_dir():
        raise ValueError(f"不是目錄: {root_path}")
    
    results = []
    flair_files = find_flair_files(root_path)
    
    if not options.process_wmh:
        return results
    
    for file_path in flair_files:
        result = process_wmh_file(file_path)
        results.append(result)
    
    return results

def find_flair_files(root_path: Path) -> List[Path]:
    """尋找所有 FLAIR 檔案"""
    return list(root_path.rglob("*T2_FLAIR*.nii.gz"))
```

## 2. 違反 UV 專案管理規則

### 🔴 問題：使用 pip 和 requirements.txt

```bash
# ❌ 錯誤的依賴管理
pip install fastapi uvicorn
pip freeze > requirements.txt
```

### ✅ 修復：使用 UV

```bash
# ✅ 正確的 UV 管理
uv add fastapi uvicorn
uv sync --dev
uv lock
```

## 3. 違反 FastAPI 最佳實踐

### 🔴 問題：使用事件處理器和同步操作

```python
# ❌ 違反 FastAPI 最佳實踐
from fastapi import FastAPI
import psycopg2

app = FastAPI()

# 錯誤1：使用棄用的事件處理器
@app.on_event("startup")
def startup_event():
    global db_conn
    db_conn = psycopg2.connect(DATABASE_URL)

# 錯誤2：同步資料庫操作
@app.get("/users/{user_id}")
def get_user(user_id: int):  # 錯誤3：沒有類型註解
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    return cursor.fetchone()
```

### ✅ 修復：使用 lifespan 和非同步操作

```python
# ✅ 符合 FastAPI 最佳實踐
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db
from .schemas import UserResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    # 啟動
    await init_database()
    yield
    # 關閉
    await cleanup_database()

app = FastAPI(lifespan=lifespan)

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """取得使用者資訊 - 非同步操作"""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(404, "使用者不存在")
    return UserResponse.from_orm(user)
```

## 4. 違反 Python 一般原則

### 🔴 問題：過度使用類別

```python
# ❌ 不必要的類別
class FileProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def validate(self):
        return os.path.exists(self.file_path)
    
    def process(self):
        if self.validate():
            return self._do_process()
    
    def _do_process(self):
        # 處理邏輯
        pass

# 使用
processor = FileProcessor("file.nii.gz")
result = processor.process()
```

### ✅ 修復：函數式編程

```python
# ✅ 函數式方法
from pathlib import Path
from typing import Optional

def validate_file(file_path: Path) -> bool:
    """驗證檔案存在"""
    return file_path.exists() and file_path.is_file()

def process_file(file_path: Path) -> Optional[ProcessResult]:
    """處理檔案 - 純函數"""
    if not validate_file(file_path):
        return None
    
    # 使用函數組合
    return pipe(
        file_path,
        load_file,
        preprocess_data,
        run_inference,
        postprocess_results
    )

# 使用
result = process_file(Path("file.nii.gz"))
```

## 5. 違反效能優化規則

### 🔴 問題：同步操作和 N+1 查詢

```python
# ❌ 效能問題
@app.get("/studies")
def get_studies():
    studies = db.query(Study).all()
    result = []
    for study in studies:
        # N+1 查詢問題
        series = db.query(Series).filter(Series.study_id == study.id).all()
        result.append({
            "study": study,
            "series": series
        })
    return result
```

### ✅ 修復：非同步批次操作

```python
# ✅ 效能優化
@app.get("/studies")
async def get_studies(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100
) -> List[StudyResponse]:
    """取得研究列表 - 預載入關聯資料"""
    # 使用 selectinload 避免 N+1
    query = (
        select(Study)
        .options(selectinload(Study.series))
        .offset(skip)
        .limit(limit)
    )
    
    result = await db.execute(query)
    studies = result.scalars().all()
    
    return [StudyResponse.from_orm(study) for study in studies]
```

## 使用程式碼品質檢查工具

執行以下命令檢查程式碼品質：

```bash
# 1. 安裝 UV（如果尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 同步專案依賴
uv sync --dev

# 3. 執行程式碼品質檢查
python scripts/code_quality.py

# 4. 自動修復格式問題
uv run ruff format code_ai backend
uv run black code_ai backend

# 5. 執行測試
uv run pytest -v

# 6. 執行 pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

## 總結

遵循這些規則可以確保：
1. **程式碼品質**：符合 Linus 的 "Good Taste" 標準
2. **現代化工具**：使用 UV 而非 pip
3. **最佳實踐**：遵循 FastAPI 和 Python 社群標準
4. **效能優化**：非同步優先，避免常見效能陷阱
5. **可維護性**：清晰的結構，易於理解和修改
