# Funboost → PgQueuer 遷移指南

## 📋 **遷移概覽**

本指南詳細說明如何將專案從 **funboost** 遷移到 **pgqueuer**，實現：
- ✅ 移除 funboost、RabbitMQ、Redis 依賴
- ✅ 使用 PostgreSQL 作為單一資料源
- ✅ 保持所有業務邏輯不變
- ✅ 提升系統簡潔性和可維護性

---

## 🎯 **遷移步驟**

### **第一階段：環境準備**

#### 1.1 更新依賴配置
```bash
# 編輯 pyproject.toml，移除 funboost 相關依賴
uv remove funboost redis celery

# pgqueuer 已存在，無需額外安裝
uv sync
```

#### 1.2 初始化 PgQueuer 資料庫結構
```bash
# 安裝 pgqueuer 佇列表格
pgq install --connection-string="postgresql://postgres_n:postgres_p@127.0.0.1:15433/dicom"

# 驗證安裝
pgq dashboard --connection-string="postgresql://postgres_n:postgres_p@127.0.0.1:15433/dicom"
```

### **第二階段：程式碼遷移**

#### 2.1 佇列管理器設置
新的佇列管理器已建立在 `code_ai/queue/` 目錄：
- `manager.py` - 核心佇列管理器
- `database.py` - 任務追蹤和記錄
- `consumer.py` - 消費者啟動器

#### 2.2 遷移現有任務

**原始 funboost 任務：**
```python
from funboost import Booster
from code_ai.task.params import BoosterParamsMyAI

@Booster(BoosterParamsMyAI(queue_name='task_pipeline_inference_queue', qps=1))
def task_pipeline_inference(func_params: Dict[str, any]):
    # 業務邏輯
    pass
```

**遷移後的 pgqueuer 任務：**
```python
from code_ai.queue.manager import get_queue_manager

queue_manager = get_queue_manager()

@queue_manager.register_processor(
    queue_name='task_pipeline_inference_queue',
    qps=1,
    max_retry_times=3,
    concurrent_num=5
)
async def task_pipeline_inference(func_params: Dict[str, any]):
    # 相同的業務邏輯，無需修改
    pass
```

#### 2.3 批次遷移所有佇列任務

需要遷移的任務清單：
- [x] `task_pipeline_inference` - 已遷移至 `task_pipeline_new.py`
- [x] `task_subprocess_inference` - 已遷移至 `task_pipeline_new.py`
- [ ] `call_post_httpx` - 待遷移
- [ ] `call_dcm2niix` - 待遷移
- [ ] `dicom_2_nii_file` - 待遷移
- [ ] 其他 11 個任務...

### **第三階段：配置更新**

#### 3.1 移除 funboost 配置檔案
```bash
# 備份後移除
mv funboost_config.py funboost_config.py.bak
mv funboost_cli_user.py funboost_cli_user.py.bak
mv boosters_manager.py boosters_manager.py.bak
mv code_ai/task/params.py code_ai/task/params.py.bak
```

#### 3.2 更新啟動腳本
```bash
# 新的 start.sh
#!/bin/bash
export PYTHONPATH=$(pwd)/code_ai

# 啟動 pgqueuer 消費者 (替代 funboost start)
python -m code_ai.queue.consumer

# 或使用 pgqueuer CLI
# pgq run code_ai.queue.consumer:start_all_consumers
```

#### 3.3 更新 Docker Compose
```yaml
# docker-compose.yml - 移除 RabbitMQ 和 Redis 服務
services:
  # 移除這些服務：
  # rabbitmq_server:
  # redis_server:
  
  # 保留 PostgreSQL
  db_server:
    image: postgres
    # ... 保持原有配置
```

### **第四階段：任務排程整合**

#### 4.1 定時任務遷移
```python
# 原始 funboost 定時任務
from funboost.timing_job import ApsJobAdder

# 新的 pgqueuer 定時任務
from code_ai.queue.manager import get_queue_manager

queue_manager = get_queue_manager()

@queue_manager.pgq.schedule("delete_old_data", "0 2 * * *")  # 每日 2:00
async def delete_old_date_scheduled():
    await delete_old_date({})
```

### **第五階段：監控和除錯**

#### 5.1 佇列監控
```bash
# 使用 pgqueuer 內建監控
pgq dashboard --interval 10 --tail 25

# 查看佇列統計
python -c "
import asyncio
from code_ai.queue.manager import get_queue_manager
async def stats():
    qm = get_queue_manager()
    await qm.initialize()
    stats = await qm.get_queue_stats('task_pipeline_inference_queue')
    print(stats)
asyncio.run(stats())
"
```

#### 5.2 任務追蹤
```python
# 查看任務執行記錄
from code_ai.queue.database import JobTracker
import asyncpg

async def check_jobs():
    conn = await asyncpg.connect("postgresql://...")
    tracker = JobTracker()
    await tracker.initialize(conn)
    
    # 查看最近任務
    recent_jobs = await tracker.get_recent_jobs(limit=10)
    print(recent_jobs)
    
    # 查看失敗分析
    failures = await tracker.get_failure_analysis()
    print(failures)
```

---

## 🔧 **API 對應表**

| Funboost 功能 | PgQueuer 對應 | 說明 |
|---------------|---------------|------|
| `@Booster(...)` | `@queue_manager.register_processor(...)` | 佇列處理器註冊 |
| `task.push(params)` | `await queue_manager.enqueue_job(queue_name, params)` | 任務入隊 |
| `BoostersManager.consume_all()` | `await start_all_consumers()` | 啟動所有消費者 |
| `funboost start` | `python -m code_ai.queue.consumer` | CLI 啟動 |
| `Serialization.to_json_str()` | `json.dumps(..., default=str)` | 序列化 |
| `save_result_status_to_sqlalchemy` | `JobTracker.save_job_result()` | 結果記錄 |

---

## ⚠️ **注意事項**

### 相容性考量
1. **非同步轉換**: pgqueuer 是非同步的，部分同步函數需要轉換
2. **序列化變更**: 使用標準 `json` 模組替代 funboost 的序列化
3. **錯誤處理**: 重試和錯誤處理邏輯略有不同

### 測試建議
1. **並行測試**: 可以同時運行兩套系統進行對比測試
2. **逐步遷移**: 建議逐個佇列遷移，確保穩定性
3. **監控對比**: 比較遷移前後的效能和穩定性

### 回滾計劃
如果遇到問題，可以快速回滾：
```bash
# 恢復 funboost 依賴
uv add funboost==48.4 redis==5.2.1

# 恢復配置檔案
mv funboost_config.py.bak funboost_config.py
mv start.sh.bak start.sh

# 重啟服務
docker-compose up -d rabbitmq_server redis_server
```

---

## 🚀 **預期效益**

### 架構簡化
- 移除 3 個外部服務 (funboost, RabbitMQ, Redis)
- 統一使用 PostgreSQL 作為資料存儲
- 減少配置複雜度

### 效能提升
- PostgreSQL 的 `LISTEN/NOTIFY` 提供即時通知
- `FOR UPDATE SKIP LOCKED` 確保高併發效能
- 減少網路跳轉和序列化開銷

### 維護便利
- 單一資料庫連接管理
- 更好的監控和除錯工具
- 減少依賴管理複雜度

---

## 📞 **技術支援**

遷移過程中如遇問題，可以：
1. 查看 pgqueuer 官方文檔：https://pgqueuer.readthedocs.io/
2. 檢查任務執行日誌和資料庫記錄
3. 使用內建的監控和統計功能
4. 對比原始 funboost 實作確保邏輯一致性
