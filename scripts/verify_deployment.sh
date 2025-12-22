#!/bin/bash
# 部署驗證腳本

source "$(dirname "$0")/../.env" 2>/dev/null || true

echo "========================================="
echo "🔍 部署驗證開始"
echo "========================================="

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 檢查服務狀態
echo -e "\n${YELLOW}1️⃣ 檢查服務狀態...${NC}"
if systemctl is-active --quiet brain-parcellation.service; then
    echo -e "${GREEN}✅ 服務運行中${NC}"
else
    echo -e "${RED}❌ 服務未運行${NC}"
    echo "   請執行: sudo systemctl start brain-parcellation.service"
fi

# 2. 檢查 Funboost consumer
echo -e "\n${YELLOW}2️⃣ 檢查 Funboost consumer...${NC}"
if ps aux | grep -v grep | grep "funboost" > /dev/null; then
    echo -e "${GREEN}✅ Consumer 運行中${NC}"
    CONSUMER_COUNT=$(ps aux | grep -v grep | grep "funboost" | wc -l)
    echo "   進程數量: $CONSUMER_COUNT"
else
    echo -e "${RED}❌ Consumer 未運行${NC}"
fi

# 3. 檢查 Redis
echo -e "\n${YELLOW}3️⃣ 檢查 Redis 連接...${NC}"
if [ ! -z "$REDIS_HOST" ] && [ ! -z "$REDIS_PORT" ]; then
    if timeout 5 redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning PING > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Redis 正常${NC}"
        echo "   連接: $REDIS_HOST:$REDIS_PORT"
    else
        echo -e "${RED}❌ Redis 連接失敗${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  未設定 Redis 環境變數${NC}"
fi

# 4. 檢查 RabbitMQ
echo -e "\n${YELLOW}4️⃣ 檢查 RabbitMQ 佇列...${NC}"
if command -v rabbitmqadmin > /dev/null 2>&1; then
    if [ ! -z "$RABBITMQ_HOST" ] && [ ! -z "$RABBITMQ_USER" ]; then
        QUEUE_INFO=$(timeout 5 rabbitmqadmin -H $RABBITMQ_HOST -P $RABBITMQ_PORT -u $RABBITMQ_USER -p $RABBITMQ_PASS list queues name messages consumers 2>/dev/null | grep task_pipeline_inference_queue)
        
        if [ ! -z "$QUEUE_INFO" ]; then
            MESSAGES=$(echo $QUEUE_INFO | awk '{print $2}')
            CONSUMERS=$(echo $QUEUE_INFO | awk '{print $3}')
            
            echo -e "${GREEN}✅ RabbitMQ 正常${NC}"
            echo "   佇列: task_pipeline_inference_queue"
            echo "   待處理訊息: $MESSAGES"
            echo "   消費者數量: $CONSUMERS"
            
            if [ "$CONSUMERS" -eq 0 ]; then
                echo -e "${RED}⚠️  沒有消費者連接！${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  無法取得佇列資訊${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  未設定 RabbitMQ 環境變數${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  未安裝 rabbitmqadmin${NC}"
fi

# 5. 檢查 GPU
echo -e "\n${YELLOW}5️⃣ 檢查 GPU 狀態...${NC}"
if command -v nvidia-smi > /dev/null 2>&1; then
    if nvidia-smi > /dev/null 2>&1; then
        echo -e "${GREEN}✅ GPU 正常${NC}"
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -n 1)
        echo "   $GPU_INFO"
    else
        echo -e "${RED}❌ GPU 異常${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  nvidia-smi 未安裝${NC}"
fi

# 6. 檢查 Redis key 數量
echo -e "\n${YELLOW}6️⃣ 檢查 Redis key 數量...${NC}"
if [ ! -z "$REDIS_HOST" ] && timeout 5 redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning PING > /dev/null 2>&1; then
    KEY_COUNT=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning --scan --pattern "inference_task:*" 2>/dev/null | wc -l)
    echo "   推理任務 key 數量: $KEY_COUNT"
    
    if [ "$KEY_COUNT" -lt 20 ]; then
        echo -e "${GREEN}✅ Key 數量正常${NC}"
    else
        echo -e "${YELLOW}⚠️  Key 數量過多 ($KEY_COUNT)，建議清理${NC}"
        echo "   執行: ./scripts/cleanup_redis_cache.sh"
    fi
    
    # 檢查狀態分佈
    if [ "$KEY_COUNT" -gt 0 ]; then
        echo -e "\n   狀態分佈:"
        QUEUED=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning --scan --pattern "inference_task:*" | xargs -I {} redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning GET {} 2>/dev/null | grep -c "queued")
        COMPLETED=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning --scan --pattern "inference_task:*" | xargs -I {} redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning GET {} 2>/dev/null | grep -c "completed")
        echo "   - queued: $QUEUED"
        echo "   - completed: $COMPLETED"
    fi
else
    echo -e "${YELLOW}⚠️  無法檢查 Redis key${NC}"
fi

# 7. 檢查最近日誌
echo -e "\n${YELLOW}7️⃣ 檢查最近日誌（最後 5 行）...${NC}"
if command -v journalctl > /dev/null 2>&1; then
    sudo journalctl -u brain-parcellation.service -n 5 --no-pager 2>/dev/null || echo "   無法讀取日誌"
else
    echo -e "${YELLOW}⚠️  journalctl 不可用${NC}"
fi

# 8. 檢查磁碟空間
echo -e "\n${YELLOW}8️⃣ 檢查磁碟空間...${NC}"
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
echo "   根目錄使用率: $DISK_USAGE%"
if [ "$DISK_USAGE" -lt 90 ]; then
    echo -e "${GREEN}✅ 磁碟空間充足${NC}"
else
    echo -e "${RED}⚠️  磁碟空間不足！${NC}"
fi

echo -e "\n========================================="
echo -e "${GREEN}🎉 驗證完成${NC}"
echo "========================================="
echo ""
echo "💡 建議操作："
echo "   - 查看實時日誌: sudo journalctl -u brain-parcellation.service -f"
echo "   - 執行完整診斷: python scripts/diagnose_inference_stuck.py"
echo "   - 快速修復: ./scripts/quick_fix_stuck.sh"
echo ""



