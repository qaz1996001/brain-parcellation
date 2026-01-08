# Inference ID 架構設計分析：Linus Torvalds 哲學洞見

> "Bad programmers worry about the code. Good programmers worry about data structures and their relationships."
> — Linus Torvalds

---

## 問題陳述

### 現況

`inference_id` 在三個不同的層級被生成或處理：

1. **Backend Service** (`backend/app/inference/service.py:1009`)
   ```python
   inference_id = uuid4()  # Backend 生成
   func_params["inference_id"] = str(inference_id)  # 傳給 worker
   ```

2. **Worker Layer** (`code_ai/task/task_pipeline.py:898`)
   ```python
   inference_id = func_params.get("inference_id", str(uuid4()))  # 接收或自生
   output_dir = os.path.join(path_json, inference_id)  # 使用
   ```

3. **Pipeline Layer** (`code_ai/pipeline/pipeline_cmb_tensorflow.py`)
   ```python
   # 目前使用 ID (study_id) 而非 inference_id
   path_output_dir = os.path.join(path_output, ID)

   # 僅在 upload 回應中記錄 inferenceId
   logging.info(f"inference_id={upload_result.inferenceId}")
   ```

### 目標需求

建立統一的輸出目錄結構：
```
path_inference_result/
└── <study_instance_uid>/
    └── <model_name>/
        └── <inference_id>/
            ├── prediction.json
            ├── <series_uid>_<label>.dcm
            └── <series_uid>_<label>.dcm
```

### 核心問題

**`inference_id` 到底應該由誰管理？**
- Backend Service 往下傳遞？（集中式管理）
- Pipeline 自行生成？（分散式管理）

---

## Linus 哲學分析

### 原則一：Data Structure Drives Behavior

> "I will, in fact, claim that the difference between a bad programmer and a good one is whether he considers his code or his data structures more important."

#### `inference_id` 的數據本質

讓我們從數據結構的角度分析 `inference_id`：

**`inference_id` 是什麼？**
- ✅ **任務標識符**：唯一標識一次推論任務的執行
- ✅ **生命週期標記**：從 READY → QUEUED → RUNNING → COMPLETE
- ✅ **結果聚合器**：所有輸出文件應該聚合在同一個 inference_id 目錄下
- ✅ **追蹤鏈條**：連接 Backend Request → Worker Execution → Pipeline Output → Database Event

**`inference_id` 不是什麼？**
- ❌ **臨時變數**：不是每個層級隨便生成的臨時 UUID
- ❌ **可選參數**：不是「有就用，沒有就自己生成」的 fallback
- ❌ **輸出產物**：不是 Pipeline 執行後才產生的結果

#### 數據結構決定答案

根據 `inference_id` 的數據本質，**它應該在任務創建時就存在，而非執行過程中產生**。

**Linus 洞見**：
```
inference_id 是任務的「出生證明」，不是「死亡證明」。
它應該在任務被創建（Backend）時簽發，而非任務執行完成（Pipeline）後追認。
```

---

### 原則二：Single Source of Truth

> "There should be one-- and preferably only one --obvious way to do it."

#### 當前設計的問題（Bad Taste）

**三處生成 = 三個 Source of Truth = 混亂**

```python
# Bad: Backend 生成
inference_id = uuid4()

# Bad: Worker 也能生成（fallback）
inference_id = func_params.get("inference_id", str(uuid4()))

# Bad: Pipeline 使用 ID (study_id) 而非 inference_id
path_output_dir = os.path.join(path_output, ID)
```

**問題症狀**：
1. **數據不一致**：Backend 生成的 inference_id 與 Pipeline 記錄的可能不同
2. **追蹤斷鏈**：Database 中的 inference_id 與文件系統的目錄名不對應
3. **測試困難**：無法預測哪個層級會生成 inference_id
4. **Debugging 地獄**：出問題時無法確定「真正的」inference_id 是哪個

#### 正確設計（Good Taste）

**Single Source of Truth: Backend Service**

```python
# ✅ Good: ONLY Backend generates
inference_id = uuid4()

# ✅ Good: Worker RECEIVES (no fallback!)
inference_id = func_params["inference_id"]  # KeyError if missing = fail fast

# ✅ Good: Pipeline RECEIVES (pure function parameter)
def pipeline_cmb(inference_id: str, ...):
    output_dir = os.path.join(path_inference_result, study_uid, model_name, inference_id)
```

**Linus 洞見**：
```
如果你需要在三個地方生成同一個 ID，那不是容錯設計，而是錯誤設計。
正確的做法是：生成一次，到處傳遞。
```

---

### 原則三：Pure Functions

> "Functions should be short and sweet, and do just one thing."

#### Pipeline 應該是 Pure Function

根據 CLAUDE.md 中的重構原則：

```python
# ❌ Bad: Pipeline 依賴環境或自行生成 ID
def pipeline_cmb(ID: str, ...):
    inference_id = generate_or_get_inference_id()  # Impure!
    output_dir = os.path.join(path_output, ID)  # 使用 study_id，不是 inference_id
```

```python
# ✅ Good: Pipeline 接收所有必要參數（Pure Function）
def pipeline_cmb(
    inference_id: str,           # 從 Backend 接收
    study_instance_uid: str,     # 從 Backend 接收
    model_name: str,             # 從 Backend 接收
    path_inference_result: str,  # 從 config 接收
    swan_file: str,
    t1_file: str,
    InputsDicomDir: str,
    gpu_n: int = 0
) -> CMBPipelineResult:
    # 純函數：相同輸入 → 相同輸出
    output_dir = os.path.join(
        path_inference_result,
        study_instance_uid,
        model_name,
        inference_id
    )
    os.makedirs(output_dir, exist_ok=True)
    # ... 執行推論 ...
```

**Pure Function 的好處**：
1. **可測試**：可以傳入 mock inference_id 進行單元測試
2. **可追蹤**：inference_id 在整個調用鏈中清晰可見
3. **無副作用**：不依賴環境變數或隨機生成
4. **確定性**：相同輸入保證相同輸出路徑

**Linus 洞見**：
```
如果你的函數會根據「今天的心情」（隨機 UUID）產生不同結果，
那它不是函數，是算命。
```

---

### 原則四：Fail Fast and Fail Loud

> "Don't ever try to be clever with fallbacks. If something is wrong, fail immediately and loudly."

#### 當前設計的 Silent Failure

```python
# ❌ Bad: 沉默的失敗，製造混亂
inference_id = func_params.get("inference_id", str(uuid4()))  # Fallback 掩蓋問題
```

**問題**：
- Backend 忘記傳 inference_id？沒關係，Worker 自己生成
- 但這樣 Database 記錄的是 Backend 的 ID，文件系統是 Worker 的 ID
- 結果：數據庫找不到文件，文件找不到數據庫記錄

#### 正確的 Fail Fast 設計

```python
# ✅ Good: 立即失敗，暴露問題
inference_id = func_params["inference_id"]  # KeyError if missing

# ✅ Good: 在最外層驗證，內層信任
if "inference_id" not in func_params:
    raise ValueError(
        "inference_id is REQUIRED. Backend must generate and pass it. "
        "DO NOT generate inference_id in worker or pipeline."
    )
```

**Linus 洞見**：
```
一個立即崩潰的程序比一個默默產生錯誤數據的程序好一千倍。
前者你能立即修復，後者你可能永遠不知道出了問題。
```

---

### 原則五：Eliminate Special Cases

> "Good taste is about understanding the problem well enough that the solution becomes obvious."

#### 當前設計的特殊情況

```python
# ❌ Bad: 三種不同的 ID 處理方式
# Backend: 生成 inference_id
# Worker: 接收或生成 inference_id
# Pipeline: 使用 study_id (ID) 作為目錄名
```

每一層都有自己的「特殊邏輯」，沒有統一的模式。

#### 統一設計（Good Taste）

```python
# ✅ Good: 單一模式，適用所有層級

# Backend: CREATE inference_id
inference_id = uuid4()

# Backend → Worker: PASS inference_id via func_params
func_params["inference_id"] = str(inference_id)

# Worker: RECEIVE inference_id
inference_id = func_params["inference_id"]

# Worker → Pipeline: PASS inference_id as argument
pipeline_func(inference_id=inference_id, ...)

# Pipeline: USE inference_id for output
output_dir = f"{path_inference_result}/{study_uid}/{model_name}/{inference_id}"
```

**統一模式**：CREATE → PASS → RECEIVE → USE

**Linus 洞見**：
```
如果你需要為 Backend、Worker、Pipeline 各寫一套 ID 管理邏輯，
那不是分層架構，而是分裂架構。
```

---

## 架構決策：推薦方案

### 決策：Backend 作為 Single Source of Truth

#### 理由

1. **數據本質**：`inference_id` 是任務的生命週期標識符，應該在任務創建時就存在
2. **追蹤需求**：Database、Cache、Event 都需要在任務啟動前記錄 inference_id
3. **Pure Function**：Worker 和 Pipeline 應該是純函數，接收參數而非生成狀態
4. **Fail Fast**：如果 Backend 沒傳 inference_id，應該立即失敗，而非 fallback

#### 數據流

```
Backend Service (CREATE)
    ↓ func_params["inference_id"]
Worker Layer (RECEIVE)
    ↓ pipeline_func(inference_id=...)
Pipeline Layer (USE)
    ↓ output_dir = .../inference_id/
File System
```

---

## 實施計劃

### Step 1: Backend Service（已完成）

**File**: `backend/app/inference/service.py`

✅ 已正確實施：
```python
inference_id = uuid4()  # Line 1009
func_params["inference_id"] = str(inference_id)  # Line 1058
```

### Step 2: Worker Layer（需修正）

**File**: `code_ai/task/task_pipeline.py`

**Before (Bad Taste)**:
```python
inference_id = func_params.get("inference_id", str(uuid4()))  # Line 898
```

**After (Good Taste)**:
```python
# Linus: "Fail fast - no fallback masking bugs"
if "inference_id" not in func_params:
    raise ValueError(
        "CRITICAL: inference_id is missing from func_params. "
        "Backend MUST generate and pass inference_id. "
        "DO NOT generate inference_id in worker layer."
    )

inference_id = func_params["inference_id"]
```

### Step 3: Pipeline Layer（需重構）

**File**: `code_ai/pipeline/pipeline_cmb_tensorflow.py`

#### 3.1 函數簽名修改

**Before**:
```python
def pipeline_cmb(
    ID: str,                      # study_id
    swan_file: str,
    t1_file: str,
    path_output: str,
    path_log: str,
    path_processModel: str,
    gpu_n: int = 0,
    ai_app_inference_complete: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    model_name: str = "cmb_model",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
```

**After**:
```python
def pipeline_cmb(
    inference_id: str,            # NEW: 從 Backend 接收
    study_id: str,                # 重命名：ID → study_id (更清楚)
    study_instance_uid: str,      # 改為必需參數
    model_name: str,              # 保持
    path_inference_result: str,   # NEW: 統一輸出根目錄
    swan_file: str,
    t1_file: str,
    InputsDicomDir: str,
    path_log: str,
    path_processModel: str,
    gpu_n: int = 0,
    ai_app_inference_complete: Optional[str] = None,
) -> CMBPipelineResult:
```

#### 3.2 輸出目錄結構

**Before**:
```python
path_output_dir = os.path.join(path_output, ID)
```

**After**:
```python
# Linus: "Data structure drives behavior"
# 目錄結構：path_inference_result/<study_uid>/<model_name>/<inference_id>/
output_dir = os.path.join(
    path_inference_result,
    study_instance_uid,
    model_name,
    inference_id
)
os.makedirs(output_dir, exist_ok=True)

# 在此目錄下放置所有輸出
prediction_json = os.path.join(output_dir, "prediction.json")
# DICOM-SEG 文件也放在同一目錄
```

#### 3.3 結果複製邏輯

**新增功能**：將所有結果複製到 `path_inference_result` 統一目錄

```python
def copy_results_to_inference_dir(
    source_files: Dict[str, str],  # {"prediction.json": "/tmp/xxx.json", ...}
    inference_dir: str,             # path_inference_result/<study>/<model>/<inference_id>
) -> bool:
    """
    將推論結果複製到統一的 inference_result 目錄。

    Linus: "Do one thing well" - 單一責任，只做文件複製
    """
    import shutil

    try:
        for dest_name, source_path in source_files.items():
            if not os.path.exists(source_path):
                logger.warning(f"Source file not found: {source_path}")
                continue

            dest_path = os.path.join(inference_dir, dest_name)
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied: {source_path} → {dest_path}")

        return True
    except Exception as e:
        logger.error(f"Failed to copy results: {e}")
        return False
```

### Step 4: Backend → Pipeline 參數傳遞

**File**: `code_ai/task/task_pipeline.py`

在調用 pipeline 函數時傳遞 `inference_id`：

**Before**:
```python
# _build_series_inference_cmd() 生成命令行參數
cmd = pipeline_config.generate_cmd(
    study_id=study_id,
    task=task,
    input_dicom_dir=dicom_dir,
    path_root=path_root
)
```

**After**:
```python
# 如果是 Python pipeline（如 CMB），直接調用函數
result = pipeline_cmb(
    inference_id=inference_id,          # 從 func_params 傳遞
    study_id=study_id,
    study_instance_uid=study_uid,
    model_name=model_id,
    path_inference_result=path_inference_result,
    swan_file=nifti_paths[0],
    t1_file=nifti_paths[1],
    InputsDicomDir=dicom_dir,
    path_log=path_log,
    path_processModel=path_process,
    gpu_n=gpu_n
)

# 如果是 Shell pipeline（如 Aneurysm），通過環境變數傳遞
env = os.environ.copy()
env["INFERENCE_ID"] = inference_id
subprocess.run(cmd, env=env, shell=True)
```

---

## 目錄結構示例

### 最終效果

```bash
path_inference_result/
├── 1.2.840.113619.2.55.3.123456789/              # study_instance_uid
│   ├── cmb_model/                                 # model_name
│   │   ├── a1b2c3d4-e5f6-4789-a0b1-c2d3e4f5g6h7/ # inference_id
│   │   │   ├── prediction.json
│   │   │   ├── 1.2.840...001_CMB.dcm
│   │   │   └── 1.2.840...002_CMB.dcm
│   │   └── x9y8z7w6-v5u4-3210-t9s8-r7q6p5o4n3m2/ # 另一次推論
│   │       ├── prediction.json
│   │       └── ...
│   └── aneurysm_model/
│       └── f1e2d3c4-b5a6-7890-c1d2-e3f4a5b6c7d8/
│           ├── prediction.json
│           └── 1.2.840...003_Aneurysm.dcm
└── 1.2.840.113619.2.55.3.987654321/              # 另一個 study
    └── ...
```

### 優點

1. **可追蹤性**：從 Database 的 inference_id 可以直接找到文件系統路徑
2. **組織性**：按 study → model → inference 層級組織，邏輯清晰
3. **可擴展**：新增 model 或 study 不會產生衝突
4. **可測試**：可以預測每次推論的輸出路徑

---

## Linus 哲學合規性檢查

### ✅ Data Structure Drives Behavior
- `inference_id` 的數據本質決定了它應該由 Backend 生成
- 目錄結構反映了數據的層級關係（study → model → inference）

### ✅ Single Source of Truth
- ONLY Backend generates `inference_id`
- Worker 和 Pipeline 只接收，不生成

### ✅ Pure Functions
- Pipeline 接收所有必要參數，無環境依賴
- 相同輸入 → 相同輸出路徑

### ✅ Fail Fast
- Worker 如果收不到 inference_id 立即拋出錯誤
- 不使用 fallback 掩蓋 Backend 的錯誤

### ✅ Eliminate Special Cases
- 統一的模式：CREATE → PASS → RECEIVE → USE
- 所有 Pipeline 使用相同的參數傳遞機制

### ✅ Do One Thing Well
- Backend: 管理任務生命週期
- Worker: 協調執行流程
- Pipeline: 執行推論並輸出結果

---

## 最終建議

### 明確回答：inference_id 由誰管理？

**答案：Backend Service 是 Single Source of Truth**

**理由**：
1. Backend 管理任務的完整生命週期（READY → COMPLETE）
2. Backend 需要在 Database 中記錄 inference_id
3. Backend 需要在 Cache 中使用 inference_id
4. Backend 返回 inference_id 給 API 調用者

**Worker 和 Pipeline 的角色**：
- **Worker**: 接收 inference_id，傳遞給 Pipeline
- **Pipeline**: 接收 inference_id，用於創建輸出目錄

### 實施優先順序

1. **高優先級**：修正 Worker fallback 邏輯（Step 2）
   - 改為 Fail Fast，不生成 fallback UUID

2. **高優先級**：重構 Pipeline 函數簽名（Step 3.1）
   - 添加 `inference_id` 參數
   - 使用統一的輸出目錄結構

3. **中優先級**：實作結果複製邏輯（Step 3.3）
   - 將 DICOM-SEG 和 JSON 複製到 inference_result 目錄

4. **低優先級**：清理舊的輸出目錄邏輯
   - 移除基於 study_id 的輸出目錄

---

## Linus 會說什麼？

> "If you're generating the same ID in three different places, you're not building a distributed system, you're building a distributed mess. Pick one place, do it right, and let everyone else just pass it along like a baton in a relay race. And if someone drops the baton? Let the race crash immediately so you know who to fire."

**翻譯**：
> 如果你在三個地方生成同一個 ID，你不是在構建分散式系統，而是在製造分散式混亂。選一個地方，做對它，讓其他所有人像接力賽一樣傳遞接力棒。如果有人掉了接力棒？讓比賽立即崩潰，這樣你就知道該開除誰了。

---

## 結論

**inference_id 管理的黃金法則**：

1. **Backend CREATES** - 生成 inference_id
2. **Worker PASSES** - 接收並傳遞（不生成）
3. **Pipeline USES** - 使用 inference_id 創建輸出目錄
4. **System FAILS FAST** - 缺少 inference_id 立即失敗

**Linus 的最終洞見**：
```
程式設計不是猜謎遊戲。
如果你不確定某個 ID 是誰生成的、什麼時候生成的、為什麼生成的，
那就是設計失敗。

好的設計應該讓答案顯而易見：
看一眼程式碼，就知道 inference_id 從哪來、往哪去、為什麼存在。
```

---

**文件版本**: 1.0
**日期**: 2026-01-07
**作者**: Claude Code (Linus Philosophy Analysis)
