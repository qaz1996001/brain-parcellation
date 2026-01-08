# ⚠️ [已棄用] GPU 資源競爭完整解決方案

> **⚠️ 文檔已棄用 (DEPRECATED)**
>
> 此文檔已被整合至新的統一部署指南。請改用：
> **[DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md](../../DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md)**
>
> **棄用原因**：
> - 此文檔與 QUICK_START_DUAL.md 分離，增加用戶認知負擔
> - 新指南提供單一真實來源 (Single Source of Truth)
> - 包含明確的 90% 使用場景決策樹（指向方案 B）
> - 整合了快速開始步驟與 GPU 資源保護
>
> **遷移建議**：
> - 方案 B（分布式控頻）現為預設推薦方案
> - 方案 A（Redis GPU 鎖）移至「進階選項」章節
> - 方案 C（優先級控制）移至「進階選項」章節
> - 使用新的 GPU 監控腳本 `scripts/monitor-gpu-usage.sh`
>
> ---
>
> 以下為歷史存檔內容，僅供參考：

# GPU 資源競爭完整解決方案

## 🎯 問題本質分析

### 您的觀察完全正確！

即使使用不同的隊列名稱（`task_pipeline_inference_queue` vs `task_pipeline_inference_queue_testing`），**兩個 worker 仍在同一台機器上運行，共用同一個 GPU**。

```
時間軸:  0s    1s    2s    3s    4s
         |-----|-----|-----|-----|
Prod Worker:  [==推理==]     [==推理==]
Test Worker:       [==推理==]     [==推理==]
                   ↑
              GPU 衝突！
```

### 為什麼 SOLO 模式不夠？

從 funboost 程式碼分析（`base_consumer.py`）發現：

**SOLO 模式只防止單一 worker 內的並發**
```python
# code_ai/task/params.py
class BoosterParamsMyAI(BoosterParamsMyRABBITMQ):
    concurrent_mode: str = ConcurrentModeEnum.SOLO  # ⚠️ 只在單個 worker 內生效
    qps: int = 1  # ⚠️ 每個 worker 各自 1 qps
```

**結果：**
- Production worker: 1 任務/秒
- Testing worker: 1 任務/秒
- **總計：2 任務/秒同時競爭 GPU** ❌

## ✅ Funboost 內建解決方案：分布式控頻

### 發現關鍵參數

從 `funboost/__init__old.py` 第 206-207 行：

```python
:param is_using_distributed_frequency_control: 是否使用分布式空頻（依賴redis統計消費者數量，然後頻率平分），默認只對當前實例化的消費者空頻有效。
假如實例化了2個qps為10的使用同一隊列名的消費者，並且都啟動，則每秒運行次數會達到20。
如果使用分布式空頻則所有消費者加起來的總運行次數是10。
```

### 工作原理

`base_consumer.py` 第 525-527 行：

```python
if self.consumer_params.is_using_distributed_frequency_control:  # 如果是需要分布式控頻
    active_num = self._distributed_consumer_statistics.active_consumer_num
    # 關鍵：將 QPS 除以活躍消費者數量
    self._frequency_control(self.consumer_params.qps / active_num,
                           self._msg_schedule_time_intercal * active_num)
else:
    self._frequency_control(self.consumer_params.qps,
                           self._msg_schedule_time_intercal)
```

**啟用分布式控頻後：**
- 2 個 worker 啟動
- Redis 追蹤活躍消費者數量：`active_num = 2`
- 每個 worker 的實際 QPS：`1 / 2 = 0.5`
- **全局總計：0.5 + 0.5 = 1 任務/秒** ✅

## 📋 三種解決方案

### 方案 A：Redis 分布式鎖（推薦）⭐

**適用場景：** 需要保持環境完全隔離（不同隊列）

**實作步驟：**

#### 1. 創建 GPU 鎖模組

`code_ai/utils/gpu_lock.py`：

```python
import redis
import uuid
import time
from typing import Optional
from contextlib import contextmanager
import os

class GPUDistributedLock:
    """Redis 分布式 GPU 鎖"""

    def __init__(self, redis_client: redis.Redis, lock_key: str = "gpu:inference:lock"):
        self.redis = redis_client
        self.lock_key = lock_key
        self.identifier = str(uuid.uuid4())

    def acquire(self, timeout: int = 300, blocking: bool = True,
                blocking_timeout: Optional[int] = None) -> bool:
        """
        獲取 GPU 鎖

        Args:
            timeout: 鎖超時時間（秒），防止死鎖
            blocking: 是否阻塞等待
            blocking_timeout: 阻塞等待超時時間（秒），None 表示無限等待
        """
        end_time = time.time() + blocking_timeout if blocking_timeout else None

        while True:
            # 使用 SET NX EX 原子操作
            acquired = self.redis.set(
                self.lock_key,
                self.identifier,
                nx=True,  # 只在 key 不存在時設置
                ex=timeout  # 過期時間
            )

            if acquired:
                return True

            if not blocking:
                return False

            if end_time and time.time() > end_time:
                return False

            # 等待 100ms 後重試
            time.sleep(0.1)

    def release(self) -> bool:
        """釋放鎖（只能釋放自己持有的鎖）"""
        # Lua 腳本確保原子性：只釋放自己的鎖
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        result = self.redis.eval(script, 1, self.lock_key, self.identifier)
        return bool(result)

    @contextmanager
    def __call__(self, timeout: int = 300, blocking_timeout: Optional[int] = None):
        """Context manager 用法"""
        acquired = self.acquire(timeout=timeout, blocking_timeout=blocking_timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()


# 全局單例
_gpu_lock: Optional[GPUDistributedLock] = None

def get_gpu_lock() -> GPUDistributedLock:
    """獲取全局 GPU 鎖實例"""
    global _gpu_lock
    if _gpu_lock is None:
        from funboost.utils.redis_manager import RedisMixin
        redis_client = RedisMixin.redis_db_frame
        _gpu_lock = GPUDistributedLock(redis_client)
    return _gpu_lock
```

#### 2. 修改推理任務

`code_ai/task/task_pipeline.py`：

```python
from backend.app.config import get_environment
from code_ai.utils.gpu_lock import get_gpu_lock

ENV = get_environment()
QUEUE_SUFFIX = "" if ENV == "production" else "_testing"

logger.info(f"Task pipeline initialized in {ENV} environment")
logger.info(f"Queue name: task_pipeline_inference_queue{QUEUE_SUFFIX}")

@Booster(BoosterParamsMyAI(
    queue_name=f'task_pipeline_inference_queue{QUEUE_SUFFIX}',
    user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
    qps=1,
))
def task_pipeline_inference(func_params: Dict[str, any]):
    """推理任務 - 使用分布式 GPU 鎖"""

    gpu_lock = get_gpu_lock()

    logger.info(f"[{ENV}] Waiting for GPU lock...")

    # 使用 GPU 鎖（最多等待 60 秒）
    with gpu_lock(timeout=300, blocking_timeout=60) as acquired:
        if not acquired:
            logger.error(f"[{ENV}] Failed to acquire GPU lock within 60 seconds")
            raise Exception("GPU lock acquisition timeout")

        logger.info(f"[{ENV}] GPU lock acquired, starting inference")

        try:
            # 原有的推理邏輯
            result = perform_inference(func_params)
            logger.info(f"[{ENV}] Inference completed")
            return result
        finally:
            logger.info(f"[{ENV}] GPU lock released")
```

**優點：**
- ✅ 保持環境完全隔離（不同隊列）
- ✅ 確保同一時間只有一個推理任務運行
- ✅ 自動處理鎖超時，防止死鎖
- ✅ 環境間自動排隊，不會衝突

**缺點：**
- ⚠️ 需要額外編寫鎖管理程式碼
- ⚠️ 依賴 Redis 可用性

---

### 方案 B：啟用分布式控頻（最簡單）⭐⭐

**適用場景：** 可接受統一隊列，通過任務參數區分環境

**實作步驟：**

#### 1. 修改任務參數模型

`code_ai/task/params.py`：

```python
class BoosterParamsMyAI(BoosterParamsMyRABBITMQ):
    concurrent_mode: str = ConcurrentModeEnum.SOLO
    concurrent_num: int  = 5
    qps: int = 1

    # ⭐ 啟用分布式控頻
    is_using_distributed_frequency_control: bool = True
```

#### 2. 使用統一隊列

`code_ai/task/task_pipeline.py`：

```python
from backend.app.config import get_environment

# ⭐ 所有環境使用相同隊列名稱
QUEUE_NAME = 'task_pipeline_inference_queue'

@Booster(BoosterParamsMyAI(
    queue_name=QUEUE_NAME,  # 統一隊列
    user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
    qps=1,  # 全局 1 qps
))
def task_pipeline_inference(func_params: Dict[str, any]):
    """推理任務 - 自動從參數讀取環境"""

    # 從任務參數讀取環境
    task_env = func_params.get('environment', 'production')

    logger.info(f"[{task_env}] Processing inference task")

    # 根據環境使用不同配置
    if task_env == 'production':
        # 使用 production 資料庫、模型路徑等
        db_name = 'dicom'
    else:
        # 使用 testing 資料庫、模型路徑等
        db_name = 'dicom_testing'

    # 原有推理邏輯
    result = perform_inference(func_params, db_name)
    return result
```

#### 3. 發布任務時添加環境標籤

在發布任務的地方（API endpoint）：

```python
from backend.app.config import get_environment

ENV = get_environment()

# 發布任務時添加環境標籤
task_pipeline_inference.push(
    study_id=study_id,
    model_name=model_name,
    environment=ENV,  # ⭐ 環境標籤
)
```

**工作原理：**
```
統一隊列: task_pipeline_inference_queue
├─ Production Worker (啟動)
├─ Testing Worker (啟動)
└─ Redis 追蹤: active_num = 2

實際 QPS 分配：
- Production Worker: qps / 2 = 0.5
- Testing Worker: qps / 2 = 0.5
- 全局總計: 1 任務/秒 ✅

任務處理：
- Worker 取出任務
- 讀取 func_params['environment']
- 根據環境選擇資料庫/配置
```

**優點：**
- ✅ 實作極簡，只需添加一行配置
- ✅ Funboost 內建功能，穩定可靠
- ✅ 自動處理多 worker 場景
- ✅ Redis 已配置，無需額外設置

**缺點：**
- ⚠️ 隊列層級無環境隔離
- ⚠️ 需要在任務參數中傳遞環境信息
- ⚠️ Production 和 Testing 共享 QPS 配額

---

### 方案 C：生產優先級控制（進階）

**適用場景：** Production 必須優先，Testing 只在空閒時運行

**實作步驟：**

#### 1. 使用不同的 QPS 配置

`code_ai/task/params.py`：

```python
from backend.app.config import get_environment

ENV = get_environment()

class BoosterParamsMyAI(BoosterParamsMyRABBITMQ):
    concurrent_mode: str = ConcurrentModeEnum.SOLO
    concurrent_num: int  = 5

    # Production 高優先級，Testing 低優先級
    qps: int = 1 if ENV == "production" else 0.2  # Testing 每 5 秒 1 個任務

    is_using_distributed_frequency_control: bool = True
```

#### 2. 設置不同隊列，但共享 GPU 鎖

結合方案 A 的 GPU 鎖 + 不同的 QPS：

```python
@Booster(BoosterParamsMyAI(
    queue_name=f'task_pipeline_inference_queue{QUEUE_SUFFIX}',
    qps=1 if ENV == "production" else 0.2,
))
def task_pipeline_inference(func_params: Dict[str, any]):
    gpu_lock = get_gpu_lock()

    # Production 鎖等待時間更長
    blocking_timeout = 120 if ENV == "production" else 30

    with gpu_lock(timeout=300, blocking_timeout=blocking_timeout) as acquired:
        if not acquired:
            if ENV == "testing":
                logger.warning("Testing task skipped due to GPU busy")
                return None  # Testing 任務可以跳過
            else:
                raise Exception("Production task GPU lock timeout")  # Production 必須執行

        return perform_inference(func_params)
```

**優點：**
- ✅ Production 絕對優先
- ✅ Testing 不影響 Production 性能
- ✅ Testing 任務可以優雅降級

**缺點：**
- ⚠️ 配置較複雜
- ⚠️ Testing 可能長時間無法執行

---

## 🎯 推薦方案選擇

### 如果您想要最簡單的解決方案：
**→ 方案 B（分布式控頻）**
- 只需修改 1 行配置
- Funboost 內建功能
- 無需額外程式碼

### 如果您想要最嚴格的環境隔離：
**→ 方案 A（GPU 鎖）**
- 不同隊列完全隔離
- 精確控制 GPU 訪問
- 適合生產環境

### 如果 Production 必須絕對優先：
**→ 方案 C（優先級控制）**
- Production 高優先級
- Testing 可降級

---

## 📊 方案對比表

| 特性 | 方案 A (GPU 鎖) | 方案 B (分布式控頻) | 方案 C (優先級) |
|------|----------------|-------------------|----------------|
| **實作難度** | 中等 | ⭐ 極簡 | 複雜 |
| **環境隔離** | ⭐ 完全隔離 | 任務級隔離 | 完全隔離 |
| **GPU 保護** | ⭐ 100% | ⭐ 100% | ⭐ 100% |
| **Production 優先** | 平等 | 平等 | ⭐ 絕對優先 |
| **程式碼修改** | 新增鎖模組 | ⭐ 1 行配置 | 較多修改 |
| **依賴** | Redis (已有) | Redis (已有) | Redis (已有) |

---

## 🚀 快速開始（推薦方案 B）

### 1. 修改配置

編輯 `code_ai/task/params.py`：

```python
class BoosterParamsMyAI(BoosterParamsMyRABBITMQ):
    concurrent_mode: str = ConcurrentModeEnum.SOLO
    concurrent_num: int  = 5
    qps: int = 1
    is_using_distributed_frequency_control: bool = True  # ⭐ 添加這一行
```

### 2. 統一隊列名稱

編輯 `code_ai/task/task_pipeline.py`：

```python
# 移除環境相關的隊列後綴
@Booster(BoosterParamsMyAI(
    queue_name='task_pipeline_inference_queue',  # 統一隊列
    user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
    qps=1,
))
def task_pipeline_inference(func_params: Dict[str, any]):
    # 從參數讀取環境
    env = func_params.get('environment', 'production')
    logger.info(f"[{env}] Processing task")
    # ... 原有邏輯
```

### 3. 驗證

```bash
# 啟動 Production worker
ENV=production python funboost_cli_user.py

# 啟動 Testing worker（另一個終端）
ENV=testing python funboost_cli_user.py

# 監控 GPU
watch -n 1 nvidia-smi

# 檢查 Redis 消費者
redis-cli keys "funboost*consumer*"
```

**預期結果：**
- ✅ 兩個 worker 同時運行
- ✅ 全局最多 1 個推理任務執行
- ✅ GPU 使用率穩定，無競爭
- ✅ 無 OOM 錯誤

---

## ✅ 總結

您的觀察是對的：**SOLO 模式 + 不同隊列 = 仍然會 GPU 競爭**

**根本原因：**
- SOLO 模式只防止單個 worker 內部並發
- 默認 QPS 控制是 per-worker，不是全局

**完美解決方案：**
- Funboost 內建 `is_using_distributed_frequency_control` 功能
- 啟用後自動在所有 worker 間共享 QPS 配額
- 確保全局最多 1 任務/秒 = 無 GPU 競爭 ✅
