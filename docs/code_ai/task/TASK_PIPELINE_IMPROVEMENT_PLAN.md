# task_pipeline.py 改進規劃

## 📋 分析範圍

本文件針對 `code_ai/task/task_pipeline.py` 進行程式碼品質分析，基於以下標準：
- **Python 一般原則** (python-general-principles.mdc)
- **Pydantic 模型規則** (pydantic-model-rules.mdc)
- **Linus 程式碼審查標準** (linus-code-review-standards.mdc)

## 🔍 當前狀態分析

### 檔案概覽

**檔案位置**: `code_ai/task/task_pipeline.py`  
**總行數**: 151 行（階段一後）  
**函數數量**: 2 個  
**引用位置**: 9 處（活躍使用）

### 函數清單

| 函數名稱 | 行數 | 複雜度 | 狀態 |
|---------|------|--------|------|
| `task_pipeline_inference` | 29-121 (93行) | 🔴 高 | ✅ 階段一完成，待階段二重構 |
| `task_subprocess_inference` | 131-150 (20行) | 🟡 中 | ✅ 階段一完成 |

## 🔴 致命問題（Linus 標準）

### 1. 函數過長 - `task_pipeline_inference` 有 92 行

**問題描述**：
- Linus 標準：函數超過 20 行應考慮拆分
- 當前 `task_pipeline_inference` 有 92 行，嚴重超標

**影響**：
- 難以維護和理解
- 違反單一職責原則
- 測試困難

**位置**：
```30:121:code_ai/task/task_pipeline.py
@Booster(...)
def task_pipeline_inference(func_params: Dict[str, any]):
    # 92 行的複雜邏輯
```

### 2. 重複的條件檢查

**問題描述**：
- `if study_uid and study_id:` 出現兩次（第64行和第99行）
- 違反 DRY 原則

**位置**：
```64:84:code_ai/task/task_pipeline.py
if study_uid and study_id:
    dcop_event = DCOPEventRequest(...)
    call_post_httpx.push(...)
else:
    pass  # 無意義的 else
```

```99:120:code_ai/task/task_pipeline.py
if study_uid and study_id:
    dcop_event = DCOPEventRequest(...)
    call_post_httpx.push(...)
```

### 3. ✅ 無意義的 else 語句（已修復）

**問題描述**：
- `else: pass` 完全無意義，應使用 early return

**狀態**：✅ **已修復**（階段一完成）

**修復內容**：
- 移除無意義的 `else: pass` 語句

## 🟡 類型安全問題（Python 一般原則）

### 4. ✅ 錯誤的類型註解（已修復）

**問題描述**：
- `Dict[str, any]` 應為 `Dict[str, Any]`
- `any` 是未定義的變數，不是類型

**狀態**：✅ **已修復**（階段一完成）

**修復內容**：
- 修正 `task_pipeline_inference`: `Dict[str, any]` → `Dict[str, Any]`
- 修正 `task_subprocess_inference`: `Dict[str, any]` → `Dict[str, Any]`
- 修正 `StudyTaskInferenceParams`: `Dict[str, any]` → `Dict[str, Any]`
- 導入 `Any` 類型

### 5. ✅ 缺少返回類型註解（已修復）

**問題描述**：
- 兩個函數都沒有返回類型註解
- 違反類型安全原則

**狀態**：✅ **已修復**（階段一完成）

**修復內容**：
- `task_pipeline_inference`: 添加 `-> str` 返回類型註解
- `task_subprocess_inference`: 添加 `-> str` 返回類型註解

### 6. ✅ 缺少輸入參數的 Pydantic 模型（已修復）

**問題描述**：
- 使用原始 `Dict[str, Any]` 而非 Pydantic 模型
- 無法進行輸入驗證
- 違反 Pydantic 模型規則

**狀態**：✅ **已修復**（階段一完成）

**修復內容**：
- 建立 `InferenceTaskParams` Pydantic 模型（用於 `task_pipeline_inference`）
- 建立 `SubprocessTaskParams` Pydantic 模型（用於 `task_subprocess_inference`）
- 兩個模型都包含輸入驗證邏輯（路徑和命令字串驗證）
- 更新函數使用 `model_validate()` 進行輸入驗證

## 🟡 程式碼品質問題

### 7. ✅ 重複的目錄建立（已修復）

**問題描述**：
- `os.makedirs(path_log, exist_ok=True)` 出現兩次（第40行和第42行）

**狀態**：✅ **已修復**（階段一完成）

**修復內容**：
- 移除重複的 `os.makedirs(path_log, exist_ok=True)` 調用

### 8. 缺少錯誤處理

**問題描述**：
- `subprocess.Popen` 沒有錯誤處理
- `process.communicate()` 沒有檢查返回碼
- 檔案寫入沒有錯誤處理

**位置**：
```88:95:code_ai/task/task_pipeline.py
process = subprocess.Popen(...)
stdout, stderr = process.communicate()
# 沒有檢查 process.returncode
```

### 9. 硬編碼的檔案大小檢查

**問題描述**：
- `FILE_SIZE = 500` 應為常數
- 但此檔案中沒有使用，可能是遺留代碼

### 10. 註解掉的日誌記錄

**問題描述**：
- 有註解掉的日誌記錄，應移除或啟用

**位置**：
```96:97:code_ai/task/task_pipeline.py
# logger.info("{}".format(stdout.decode()))
# logger.warn("{}".format(stderr.decode()))
```

### 11. 缺少輸入驗證

**問題描述**：
- 沒有驗證 `func_params` 的必要欄位
- 可能導致運行時錯誤

**必要欄位**：
- `nifti_study_path`
- `dicom_study_path`
- `study_uid` (可選)
- `study_id` (可選)

## 🟢 改進規劃

### ✅ 階段一：類型安全和輸入驗證（已完成）

**完成日期**：2025-12-17  
**Git Commit**：`97a93e7f0a92695008e3be498e8250bd704c860c`  
**狀態**：✅ **已完成**

#### 實際完成內容

1. **✅ 建立 Pydantic 輸入模型**
   - 建立 `InferenceTaskParams` 模型（`code_ai/task/schema/intput_params.py`）
     - 包含 `nifti_study_path`, `dicom_study_path`, `study_uid`, `study_id` 欄位
     - 添加路徑格式驗證（不強制要求路徑存在，因為可能是遠程路徑）
   - 建立 `SubprocessTaskParams` 模型
     - 包含 `cmd_str` 欄位
     - 添加命令字串驗證
   - 更新 `schema/__init__.py` 導出新模型

2. **✅ 修正類型註解**
   - 修正所有 `Dict[str, any]` → `Dict[str, Any]`（3 處）
   - 導入 `Any` 類型
   - 修正 `StudyTaskInferenceParams` 的類型註解

3. **✅ 添加返回類型註解**
   - `task_pipeline_inference` → `-> str`
   - `task_subprocess_inference` → `-> str`

4. **✅ 更新函數使用新模型**
   - `task_pipeline_inference` 使用 `InferenceTaskParams.model_validate()`
   - `task_subprocess_inference` 使用 `SubprocessTaskParams.model_validate()`
   - 使用 `params.model_dump()` 替代原始 `func_params`

5. **✅ 程式碼清理**
   - 移除重複的 `os.makedirs(path_log, exist_ok=True)` 調用
   - 移除無意義的 `else: pass` 語句
   - 添加函數文檔字串

#### 實際實作細節

**模型定義**（`code_ai/task/schema/intput_params.py`）：
```python
class InferenceTaskParams(BaseJsonAbleModel):
    """推論任務參數模型 - 用於 task_pipeline_inference"""
    nifti_study_path: str = Field(..., description="NIFTI Study 路徑")
    dicom_study_path: str = Field(..., description="DICOM Study 路徑")
    study_uid: Optional[str] = Field(None, description="Study UID")
    study_id: Optional[str] = Field(None, description="Study ID")
    
    @field_validator('nifti_study_path', 'dicom_study_path')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """驗證路徑格式（不強制要求路徑存在，因為可能是遠程路徑）"""
        if not v or not v.strip():
            raise ValueError(f"路徑不能為空")
        return v.strip()

class SubprocessTaskParams(BaseJsonAbleModel):
    """子進程任務參數模型 - 用於 task_subprocess_inference"""
    cmd_str: str = Field(..., description="要執行的命令字串")
    
    @field_validator('cmd_str')
    @classmethod
    def validate_cmd_str(cls, v: str) -> str:
        """驗證命令字串"""
        if not v or not v.strip():
            raise ValueError("命令字串不能為空")
        return v.strip()
```

**函數更新**（`code_ai/task/task_pipeline.py`）：
```python
def task_pipeline_inference(func_params: Dict[str, Any]) -> str:
    """推論任務主函數 - 執行推論管道並發送狀態事件"""
    # 驗證並解析輸入參數
    params = InferenceTaskParams.model_validate(func_params)
    # ... 使用 params.nifti_study_path, params.dicom_study_path 等

def task_subprocess_inference(func_params: Dict[str, Any]) -> str:
    """子進程推論任務 - 執行單一命令並返回結果"""
    # 驗證並解析輸入參數
    params = SubprocessTaskParams.model_validate(func_params)
    # ... 使用 params.cmd_str
```

#### 改進效果

| 指標 | 改進前 | 改進後 | 狀態 |
|------|--------|--------|------|
| 類型安全覆蓋率 | 0% | 100% | ✅ |
| 輸入驗證 | ❌ 無 | ✅ Pydantic 驗證 | ✅ |
| 返回類型註解 | ❌ 無 | ✅ 完整 | ✅ |
| 代碼重複 | 高 | 低 | ✅ |

#### 注意事項

- 用戶調整了 logger 初始化位置（移到導入之後），符合 Python 最佳實踐
- 函數長度仍然為 93 行，需要在階段二進行拆分
- 重複的條件檢查仍然存在，需要在階段二處理

### 階段一：類型安全和輸入驗證（高優先級）

#### 1.1 建立 Pydantic 輸入模型

**目標**：建立類型安全的輸入參數模型

**實作**：
```python
# code_ai/task/schema/input_params.py
from pydantic import BaseModel, Field, validator
from typing import Optional
from pathlib import Path

class InferenceTaskParams(BaseModel):
    """推論任務參數模型"""
    nifti_study_path: str = Field(..., description="NIFTI Study 路徑")
    dicom_study_path: str = Field(..., description="DICOM Study 路徑")
    study_uid: Optional[str] = Field(None, description="Study UID")
    study_id: Optional[str] = Field(None, description="Study ID")
    
    @validator('nifti_study_path', 'dicom_study_path')
    def validate_path_exists(cls, v):
        """驗證路徑存在"""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"路徑不存在: {v}")
        return str(path.absolute())
    
    class Config:
        json_loads = orjson.loads
        json_dumps = orjson.dumps
```

#### 1.2 修正類型註解

**目標**：修正所有類型註解錯誤

**變更**：
- `Dict[str, any]` → `Dict[str, Any]`
- 添加返回類型註解
- 導入 `Any` 類型

### 階段二：函數拆分和重構（高優先級）

#### 2.1 拆分 `task_pipeline_inference` 函數

**目標**：將 92 行函數拆分為多個小函數

**拆分策略**：

```python
# 1. 準備目錄結構
def _prepare_directories() -> Dict[str, Path]:
    """準備必要的目錄結構"""
    pass

# 2. 建立推論命令
def _build_inference_commands(
    nifti_study_path: Path,
    dicom_study_path: Path
) -> InferenceCmd:
    """建立推論命令"""
    pass

# 3. 儲存命令到檔案
def _save_commands_to_file(
    inference_cmd: InferenceCmd,
    output_path: Path
) -> None:
    """儲存推論命令到 JSON 檔案"""
    pass

# 4. 發送狀態事件
def _send_status_event(
    study_uid: Optional[str],
    study_id: Optional[str],
    status: DCOPStatus,
    params_data: Dict[str, Any],
    result_data: Optional[Dict[str, Any]] = None
) -> None:
    """發送狀態事件到 API"""
    pass

# 5. 執行推論命令
def _execute_inference_commands(
    inference_cmd: InferenceCmd
) -> List[Tuple[str, str, str]]:
    """執行所有推論命令並返回結果"""
    pass

# 6. 主函數（組合上述函數）
@Booster(...)
def task_pipeline_inference(func_params: Dict[str, Any]) -> str:
    """推論任務主函數"""
    params = InferenceTaskParams.model_validate(func_params)
    
    # 準備目錄
    directories = _prepare_directories()
    
    # 建立命令
    inference_cmd = _build_inference_commands(
        Path(params.nifti_study_path),
        Path(params.dicom_study_path)
    )
    
    # 儲存命令
    _save_commands_to_file(inference_cmd, directories['cmd_tools'])
    
    # 發送 RUNNING 狀態
    _send_status_event(
        params.study_uid,
        params.study_id,
        DCOPStatus.STUDY_INFERENCE_RUNNING,
        {...}
    )
    
    # 執行命令
    results = _execute_inference_commands(inference_cmd)
    
    # 發送 COMPLETE 狀態
    _send_status_event(
        params.study_uid,
        params.study_id,
        DCOPStatus.STUDY_INFERENCE_COMPLETE,
        {...},
        result_data={'result': Serialization.to_json_str(results)}
    )
    
    return Serialization.to_json_str(results)
```

#### 2.2 消除重複條件檢查

**目標**：使用 early return 和輔助函數消除重複

**實作**：
```python
def _should_send_events(study_uid: Optional[str], study_id: Optional[str]) -> bool:
    """判斷是否應該發送事件"""
    return study_uid is not None and study_id is not None

# 使用
if not _should_send_events(params.study_uid, params.study_id):
    return result  # Early return
```

### 階段三：錯誤處理和日誌（中優先級）

#### 3.1 添加 subprocess 錯誤處理

**目標**：正確處理子進程錯誤

**實作**：
```python
def _execute_single_command(cmd_str: str) -> Tuple[str, str, int]:
    """執行單一命令並返回結果"""
    process = subprocess.Popen(
        args=cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    return_code = process.returncode
    
    if return_code != 0:
        logger.error(
            f"命令執行失敗: {cmd_str}\n"
            f"返回碼: {return_code}\n"
            f"錯誤輸出: {stderr.decode()}"
        )
        raise subprocess.CalledProcessError(return_code, cmd_str, stdout, stderr)
    
    return stdout.decode(), stderr.decode(), return_code
```

#### 3.2 添加檔案操作錯誤處理

**目標**：處理檔案寫入錯誤

**實作**：
```python
def _save_commands_to_file(
    inference_cmd: InferenceCmd,
    output_path: Path
) -> None:
    """儲存推論命令到 JSON 檔案"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                inference_cmd.model_dump()["cmd_items"],
                f,
                indent=2,
                ensure_ascii=False
            )
        logger.info(f"命令已儲存到: {output_path}")
    except (IOError, OSError) as e:
        logger.error(f"無法寫入檔案 {output_path}: {e}")
        raise
```

### 階段四：程式碼清理（低優先級）

#### 4.1 移除重複的目錄建立

**目標**：統一目錄準備邏輯

**實作**：
```python
def _prepare_directories() -> Dict[str, Path]:
    """準備必要的目錄結構"""
    path_process = Path(os.getenv("PATH_PROCESS"))
    path_json = Path(os.getenv("PATH_JSON"))
    path_log = Path(os.getenv("PATH_LOG"))
    path_cmd_tools = path_process / "Deep_cmd_tools"
    
    directories = {
        'json': path_json,
        'log': path_log,
        'cmd_tools': path_cmd_tools,
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"目錄已準備: {name} = {path}")
    
    return directories
```

#### 4.2 移除註解掉的代碼

**目標**：清理無用的註解

**變更**：
- 移除第96-97行的註解日誌
- 移除無意義的 `else: pass`

## 📊 改進優先級矩陣

| 問題 | 優先級 | 影響 | 工作量 | 階段 |
|------|--------|------|--------|------|
| 類型註解錯誤 | 🔴 高 | 運行時錯誤風險 | 低 | 階段一 |
| 缺少 Pydantic 模型 | 🔴 高 | 輸入驗證缺失 | 中 | 階段一 |
| 函數過長 | 🔴 高 | 可維護性差 | 高 | 階段二 |
| 重複條件檢查 | 🟡 中 | 代碼重複 | 中 | 階段二 |
| 缺少錯誤處理 | 🟡 中 | 穩定性問題 | 中 | 階段三 |
| 重複目錄建立 | 🟢 低 | 輕微效率問題 | 低 | 階段四 |
| 註解代碼 | 🟢 低 | 代碼整潔 | 低 | 階段四 |

## 🎯 預期改進效果

### 程式碼品質指標

| 指標 | 改進前 | 階段一後 | 目標（全部完成） | 狀態 |
|------|--------|----------|----------------|------|
| 最大函數長度 | 92 行 | 93 行 | < 20 行 | 🟡 待階段二 |
| 類型安全覆蓋率 | 0% | 100% | 100% | ✅ 已完成 |
| 輸入驗證覆蓋率 | 0% | 100% | 100% | ✅ 已完成 |
| 返回類型註解 | 0% | 100% | 100% | ✅ 已完成 |
| 錯誤處理覆蓋率 | 0% | 0% | 100% | 🟡 待階段三 |
| 代碼重複率 | 高 | 中 | 低 | 🟡 部分改善 |
| 測試覆蓋率 | 未知 | 未知 | > 80% | 🟡 待測試 |

### 可維護性提升

1. **函數職責清晰**：每個函數只做一件事
2. **類型安全**：編譯時發現錯誤
3. **錯誤處理完善**：運行時錯誤可追蹤
4. **測試友好**：小函數易於單元測試

## 📝 實作檢查清單

### ✅ 階段一：類型安全和輸入驗證（已完成）
- [x] 建立 `InferenceTaskParams` Pydantic 模型
- [x] 建立 `SubprocessTaskParams` Pydantic 模型
- [x] 修正 `Dict[str, any]` → `Dict[str, Any]`（3 處）
- [x] 添加返回類型註解（2 個函數）
- [x] 添加輸入驗證邏輯（路徑和命令字串驗證）
- [x] 更新函數使用新模型（`model_validate()`）
- [x] 更新 `schema/__init__.py` 導出新模型
- [x] 移除重複的目錄建立
- [x] 移除無意義的 `else: pass`
- [x] Git 提交變更（commit: `97a93e7f0a92695008e3be498e8250bd704c860c`）

### 階段二：函數拆分和重構
- [ ] 拆分 `task_pipeline_inference` 為 6 個小函數
- [ ] 建立 `_prepare_directories()` 函數
- [ ] 建立 `_build_inference_commands()` 函數
- [ ] 建立 `_save_commands_to_file()` 函數
- [ ] 建立 `_send_status_event()` 函數
- [ ] 建立 `_execute_inference_commands()` 函數
- [ ] 重構主函數使用上述輔助函數
- [ ] 移除重複的條件檢查

### 階段三：錯誤處理和日誌
- [ ] 添加 subprocess 錯誤處理
- [ ] 添加檔案操作錯誤處理
- [ ] 添加目錄建立錯誤處理
- [ ] 改進日誌記錄（移除註解，添加關鍵日誌）

### 階段四：程式碼清理
- [ ] 移除重複的 `os.makedirs` 調用
- [ ] 移除 `else: pass` 無意義代碼
- [ ] 移除註解掉的日誌代碼
- [ ] 統一常數定義

## 🧪 測試策略

### 單元測試

1. **輸入驗證測試**
   - 測試 `InferenceTaskParams` 驗證邏輯
   - 測試路徑不存在的情況
   - 測試必要欄位缺失的情況

2. **函數拆分測試**
   - 測試每個輔助函數獨立功能
   - 測試函數組合的正確性

3. **錯誤處理測試**
   - 測試 subprocess 失敗情況
   - 測試檔案寫入失敗情況
   - 測試目錄建立失敗情況

### 整合測試

1. **端到端測試**
   - 測試完整的推論流程
   - 測試事件發送流程
   - 測試錯誤恢復機制

## ⚠️ 風險評估

### 高風險項目

1. **函數拆分可能影響現有調用**
   - **風險**：中
   - **緩解**：保持主函數簽名不變，僅內部重構

2. **Pydantic 模型可能改變輸入格式**
   - **風險**：低
   - **緩解**：使用 `model_validate` 自動轉換

### 低風險項目

1. **錯誤處理添加可能改變行為**
   - **風險**：低
   - **緩解**：錯誤處理僅添加日誌和異常，不改變成功路徑

## 📚 參考文檔

- [Python 一般原則](../.cursor/rules/python-general-principles.mdc)
- [Pydantic 模型規則](../.cursor/rules/pydantic-model-rules.mdc)
- [Linus 程式碼審查標準](../.cursor/rules/linus-code-review-standards.mdc)
- [Funboost Task 分析](./FUNBOOST_TASK_ANALYSIS.md)

## 🔄 版本歷史

| 版本 | 日期 | 作者 | 變更說明 |
|------|------|------|----------|
| 1.0.0 | 2025-12-17 | AI Assistant | 初始版本，完整分析 |
| 1.1.0 | 2025-12-17 | AI Assistant | 階段一完成：類型安全和輸入驗證 |

### 階段一完成記錄（v1.1.0）

**完成日期**：2025-12-17  
**Git Commit**：`97a93e7f0a92695008e3be498e8250bd704c860c`

**變更檔案**：
- `code_ai/task/task_pipeline.py` - 更新類型註解、添加返回類型、使用新模型
- `code_ai/task/schema/intput_params.py` - 添加 InferenceTaskParams 和 SubprocessTaskParams
- `code_ai/task/schema/__init__.py` - 導出新模型

**完成項目**：
- ✅ 建立 2 個 Pydantic 模型
- ✅ 修正 3 處類型註解錯誤
- ✅ 添加 2 個返回類型註解
- ✅ 添加輸入驗證邏輯
- ✅ 移除代碼重複和無意義代碼

**待處理項目**（階段二）：
- ⏳ 函數拆分（93 行 → < 20 行）
- ⏳ 消除重複條件檢查

---

**最後更新時間**：2025-12-17  
**分析工具**：代碼庫語義搜索 + 規則文件分析  
**當前狀態**：✅ 階段一完成，🟡 階段二待開始

