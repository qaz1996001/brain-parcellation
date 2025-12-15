#!/bin/bash
# 定期清理 Redis 推理任務快取
# 建議透過 cron 每小時執行一次

source "$(dirname "$0")/../.env"

echo "[$(date)] 開始清理 Redis 推理任務快取"

# 檢查環境變數
if [ -z "$REDIS_HOST" ] || [ -z "$REDIS_PORT" ] || [ -z "$REDIS_PASSWORD" ]; then
    echo "[ERROR] Redis 環境變數未設定"
    exit 1
fi

# 計算 1 小時前的時間戳
ONE_HOUR_AGO=$(($(date +%s) - 3600))

# 取得所有推理任務 key
KEYS=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning --scan --pattern "inference_task:*")

if [ -z "$KEYS" ]; then
    echo "[INFO] 沒有發現推理任務 key"
    exit 0
fi

TOTAL_KEYS=$(echo "$KEYS" | wc -l)
DELETED_COUNT=0

echo "[INFO] 發現 $TOTAL_KEYS 個推理任務 key"

# 逐一檢查 TTL
for KEY in $KEYS; do
    TTL=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning TTL "$KEY")
    
    # TTL < 1800 秒（30 分鐘）表示可能卡住
    if [ "$TTL" -lt 1800 ] && [ "$TTL" -gt 0 ]; then
        echo "[WARN] Key $KEY TTL 異常低: ${TTL}s，正在刪除..."
        redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning DEL "$KEY" > /dev/null
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

echo "[INFO] 清理完成，共刪除 $DELETED_COUNT 個 key"
echo "[$(date)] 清理結束"

