#!/bin/bash
# 快速移除任務: 77167315-6579-475f-8609-0f65b9f06a66
# 此腳本會同時處理 RabbitMQ 佇列和 Redis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="77167315-6579-475f-8609-0f65b9f06a66"

echo "=========================================="
echo "完整移除任務: $TASK_ID"
echo "=========================================="

# 方法 1: 使用完整移除工具 (推薦 - 處理 RabbitMQ + Redis)
echo -e "\n方法 1: 使用完整移除工具 (RabbitMQ + Redis)"
python "$SCRIPT_DIR/remove_task_complete.py" --task-id "$TASK_ID"

# 方法 2: 僅移除 Redis (不處理 RabbitMQ)
# echo -e "\n方法 2: 僅移除 Redis RPC 結果"
# python "$SCRIPT_DIR/quick_remove_task.py"

# 方法 3: 使用 redis-cli 直接刪除 (僅 Redis，需要安裝 redis-cli)
# echo -e "\n方法 3: 使用 redis-cli"
# REDIS_HOST=${REDIS_HOST:-127.0.0.1}
# REDIS_PORT=${REDIS_PORT:-6379}
# REDIS_DB=${REDIS_DB_FILTER_AND_RPC_RESULT:-3}
# 
# echo "連接 Redis: $REDIS_HOST:$REDIS_PORT (DB: $REDIS_DB)"
# redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB DEL "task_pipeline_inference_queue_result:$TASK_ID"

