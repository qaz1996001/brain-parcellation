# Linus 式重構：Model ID 映射設計

## 🔴 當前設計的問題

### 違反 Linus 原則分析

```python
# 當前的「不優雅」設計
MODEL_UUID_MAPPING = {
    "48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6": InferenceEnum.CMB,
    "924d1538-597c-41d6-bc27-4b0b359111cf": InferenceEnum.Aneurysm,
    # ...
}

def _resolve_model_id_to_inference_enum(model_id: str):
    # 方式 1: UUID 映射
    if model_id in MODEL_UUID_MAPPING:
        return MODEL_UUID_MAPPING[model_id]
    # 方式 2: 直接名稱
    try:
        return InferenceEnum(model_id)
    except ValueError:
        pass
    # 方式 3: 不區分大小寫
    for enum_member in InferenceEnum:
        if enum_member.value.upper() == model_id_upper:
            return enum_member
```

### Linus 會怎麼批評？

**"Bad programmers worry about the code. Good programmers worry about data structures."**

#### 問題 1: 資料結構錯誤
- ❌ UUID 是數據庫的數據，不應該硬編碼在代碼裡
- ❌ 把配置數據混入代碼 → 違反「圍繞資料設計程式碼」原則
- ❌ 數據改變時代碼也要改 → 錯誤的依賴方向

#### 問題 2: 有特殊情況
- ❌ UUID 是「特殊情況」，ModelName 是「正常情況」
- ❌ **Good Taste**: "消除特殊情況，讓它變成正常情況"
- ❌ 3 種映射方式 = 3 種特殊情況處理

#### 問題 3: 過度設計
- ❌ Worker 不應該知道 UUID（這是 Backend 的事）
- ❌ "Don't over-design" → 試圖讓 Worker 變成萬能翻譯器
- ❌ "Intelligence is the ability to avoid doing work" → 現在在做不必要的工作

## ✅ Linus 式的正確設計

### 核心洞見：責任分離

**Linus: "git actually has a simple design, with stable and reasonably well-documented data structures."**

```
┌─────────────────────────────────────────────────┐
│ Backend Layer (負責數據庫交互)                     │
│   - 查詢數據庫：model.id → model.name            │
│   - 傳遞 model_name (如 "CMB") 給 Worker         │
└─────────────────────────────────────────────────┘
                    ↓ model_name
┌─────────────────────────────────────────────────┐
│ Worker Layer (只處理業務邏輯)                      │
│   - 接收 model_name: "CMB"                       │
│   - 簡單映射: InferenceEnum(model_name)           │
└─────────────────────────────────────────────────┘
```

### 重構步驟

#### Step 1: 簡化 Worker 函數（消除 UUID）

**Before (不優雅):**
```python
def _resolve_model_id_to_inference_enum(model_id: str):
    """支援 UUID、模型名稱、不區分大小寫..."""
    # 48 行代碼，3 種映射方式
    MODEL_UUID_MAPPING = {...}  # 硬編碼
    if model_id in MODEL_UUID_MAPPING: ...
    try: InferenceEnum(model_id) ...
    for enum_member in InferenceEnum: ...
```

**After (Linus 式 - Good Taste):**
```python
def _resolve_model_name_to_inference_enum(model_name: str) -> InferenceEnum:
    """
    將模型名稱映射到 InferenceEnum。

    Linus: "消除特殊情況" - 只接收模型名稱，不處理 UUID。
    UUID → ModelName 的映射應該在 Backend 層完成（查數據庫）。

    Args:
        model_name: 模型名稱（如 "CMB", "Aneurysm", "WMH"）

    Returns:
        InferenceEnum: 對應的推論枚舉值

    Raises:
        ValueError: 如果模型名稱不存在
    """
    try:
        return InferenceEnum(model_name)
    except ValueError:
        # Linus: "Fail fast and fail loud"
        valid_models = [e.value for e in InferenceEnum]
        raise ValueError(
            f"Unknown model: '{model_name}'. Valid: {valid_models}"
        )
```

**改進：**
- ✅ 從 48 行減少到 12 行
- ✅ 消除 UUID 硬編碼（特殊情況）
- ✅ 消除 3 種映射方式（只留 1 種）
- ✅ 函數做一件事，做好它

#### Step 2: Backend 負責 UUID → ModelName

**backend/app/inference/service.py:**
```python
async def dispatch_series_inference(
    inference_id: str,
    model_id: str,  # UUID from database
    series_uids: List[str],
    ...
):
    """Backend 負責 UUID → ModelName 映射"""

    # Linus: "圍繞資料設計程式碼"
    # 數據庫是 UUID 的單一數據源
    model = await db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise ValueError(f"Model not found: {model_id}")

    # 傳遞 model.name（不是 UUID）給 Worker
    task_params = {
        "series_uids": series_uids,
        "model_name": model.name,  # ✅ 傳遞 "CMB", 不傳遞 UUID
        "nifti_series_paths": nifti_paths,
        ...
    }

    task_pipeline_inference.push(task_params)
```

#### Step 3: 更新 Worker 接收參數

**code_ai/task/task_pipeline.py:**
```python
def _task_series_pipeline_inference(func_params: Dict[str, Any]):
    """Series Level 推論 - 接收 model_name（不是 UUID）"""

    # 提取參數
    series_uids = func_params["series_uids"]
    model_name = func_params["model_name"]  # ✅ 現在是 "CMB", 不是 UUID

    # Linus: 簡單映射（不需要 UUID 硬編碼）
    inference_enum = _resolve_model_name_to_inference_enum(model_name)

    # 根據 config.yaml 重新排序
    sorted_labels, sorted_uids, ... = _reorder_series_by_config(
        target_labels=target_labels,
        series_uids=series_uids,
        model_name=model_name,  # ✅ 傳遞 model_name
        ...
    )

    # ...推論邏輯
```

### 向後兼容處理

如果需要支持舊的 UUID 調用方式（過渡期）：

```python
def _resolve_model_id_or_name(model_id_or_name: str) -> InferenceEnum:
    """
    過渡期函數：支持 UUID 或 ModelName。

    Linus: "修坑洞而非仰望星空" - 先支持舊代碼，逐步遷移。
    TODO: 當所有 Backend 都改用 model_name 後，刪除這個函數。
    """
    # 判斷是否為 UUID（36 字元，包含 '-'）
    if len(model_id_or_name) == 36 and '-' in model_id_or_name:
        # Legacy mode: 查詢數據庫（或臨時映射）
        logger.warning(
            f"DEPRECATED: Worker received UUID '{model_id_or_name}'. "
            f"Backend should pass model_name instead."
        )
        # 臨時方案：從環境變數或配置文件讀取映射
        # （不在代碼中硬編碼）
        model_name = _lookup_model_name_from_uuid(model_id_or_name)
    else:
        # Modern mode: 直接使用 model_name
        model_name = model_id_or_name

    return _resolve_model_name_to_inference_enum(model_name)
```

## 📊 改進對比

### Before (不優雅)
```
代碼行數：48 行
硬編碼數據：4 個 UUID
映射方式：3 種（UUID / Name / Case-insensitive）
數據源：代碼中硬編碼
維護成本：高（UUID 改變需改代碼）
測試複雜度：高（3 種路徑）
```

### After (Linus 式)
```
代碼行數：12 行（減少 75%）
硬編碼數據：0（消除）
映射方式：1 種（Name only）
數據源：數據庫（單一數據源）
維護成本：低（只改數據庫）
測試複雜度：低（1 種路徑）
```

## 🎯 Linus 原則應用總結

### 第二原則：資料結構優先
- ✅ **Before**: 代碼包含數據（UUID 硬編碼）
- ✅ **After**: 資料在數據庫，代碼只處理邏輯

### Good Taste：消除特殊情況
- ✅ **Before**: UUID 是特殊情況，需要映射表
- ✅ **After**: 只有正常情況（ModelName），無特殊處理

### 第六原則：不要過度設計
- ✅ **Before**: 3 種映射方式，試圖支持所有可能
- ✅ **After**: 1 種映射方式，解決實際問題

### 第九原則：智慧是避免工作
- ✅ **Before**: Worker 做 UUID 映射（不必要的工作）
- ✅ **After**: Backend 做 UUID 映射（該做的人做該做的事）

## 🚀 遷移計劃

### Phase 1: Backend 改動（優先）
1. 修改 `backend/app/inference/service.py`
2. 查詢數據庫獲取 `model.name`
3. 傳遞 `model_name` 給 Worker（不傳遞 UUID）

### Phase 2: Worker 改動（同步）
1. 重命名函數：`_resolve_model_id_to_inference_enum` → `_resolve_model_name_to_inference_enum`
2. 移除 `MODEL_UUID_MAPPING` 硬編碼
3. 簡化映射邏輯（只保留 `InferenceEnum(model_name)`）

### Phase 3: 清理過渡代碼
1. 確認所有 Backend 都改用 `model_name`
2. 刪除向後兼容函數
3. 更新測試用例

## 💡 Linus 的智慧

**"Bad programmers worry about the code. Good programmers worry about data structures and their relationships."**

這個重構的核心不是改代碼，而是**修正數據流的架構**：
- **Before**: 數據（UUID）混入代碼 → 錯誤的資料結構
- **After**: 數據在數據庫，代碼只處理邏輯 → 正確的資料結構

**"If you write code that needs comments at the end of a line, your code is crap."**

- **Before**: 需要註解解釋為什麼有 3 種映射方式
- **After**: 代碼自解釋，不需要註解

**"Intelligence is the ability to avoid doing work, yet getting the work done."**

- **Before**: Worker 做 UUID 映射（不該它做的工作）
- **After**: Backend 做 UUID 映射（該誰做的工作誰做）

---

**結論**: 這不是簡單的代碼重構，而是**架構層面的資料流修正**。
