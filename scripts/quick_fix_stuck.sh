#!/bin/bash
# 快速修復 Study 300.50 卡住的問題

echo "========================================="
echo "🏥 Quick Fix: Study 300.50 卡住問題"
echo "========================================="

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 檢查 funboost consumer 是否在運行
echo -e "\n${YELLOW}1️⃣ 檢查 funboost consumer 狀態...${NC}"
if ps aux | grep -v grep | grep "funboost" > /dev/null; then
    echo -e "${GREEN}✅ Funboost consumer 正在運行${NC}"
    ps aux | grep -v grep | grep "funboost" | head -n 1
else
    echo -e "${RED}❌ Funboost consumer 未運行！${NC}"
    echo "   正在重啟服務..."
    sudo systemctl restart brain-parcellation.service
    sleep 3
    
    if ps aux | grep -v grep | grep "funboost" > /dev/null; then
        echo -e "${GREEN}✅ 服務重啟成功${NC}"
    else
        echo -e "${RED}❌ 服務重啟失敗，請手動檢查${NC}"
        exit 1
    fi
fi

# 2. 檢查 RabbitMQ 連接
echo -e "\n${YELLOW}2️⃣ 檢查 RabbitMQ 連接...${NC}"
source .env
if timeout 5 rabbitmqadmin -H $RABBITMQ_HOST -P $RABBITMQ_PORT -u $RABBITMQ_USER -p $RABBITMQ_PASS list queues > /dev/null 2>&1; then
    echo -e "${GREEN}✅ RabbitMQ 連接正常${NC}"
    
    # 檢查佇列狀態
    QUEUE_INFO=$(timeout 5 rabbitmqadmin -H $RABBITMQ_HOST -P $RABBITMQ_PORT -u $RABBITMQ_USER -p $RABBITMQ_PASS list queues name messages consumers | grep task_pipeline_inference_queue)
    
    if [ ! -z "$QUEUE_INFO" ]; then
        MESSAGES=$(echo $QUEUE_INFO | awk '{print $2}')
        CONSUMERS=$(echo $QUEUE_INFO | awk '{print $3}')
        
        echo "   佇列: task_pipeline_inference_queue"
        echo "   待處理訊息: $MESSAGES"
        echo "   消費者數量: $CONSUMERS"
        
        if [ "$CONSUMERS" -eq 0 ]; then
            echo -e "${RED}⚠️  沒有消費者連接！正在重啟服務...${NC}"
            sudo systemctl restart brain-parcellation.service
        fi
    fi
else
    echo -e "${YELLOW}⚠️  無法連接 RabbitMQ（可能未安裝 rabbitmqadmin）${NC}"
fi

# 3. 檢查 Redis 連接
echo -e "\n${YELLOW}3️⃣ 檢查 Redis 連接...${NC}"
if timeout 5 redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning PING > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis 連接正常${NC}"
    
    # 檢查是否有卡住的 key
    STUCK_KEYS=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning --scan --pattern "inference_task:*" | wc -l)
    
    if [ "$STUCK_KEYS" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  發現 $STUCK_KEYS 個推理任務 key${NC}"
        echo "   這些 key 可能導致任務被跳過"
        
        read -p "   是否要列出這些 key? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning --scan --pattern "inference_task:*"
        fi
        
        read -p "   是否要清除所有推理任務 key? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning --scan --pattern "inference_task:*" | xargs redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --no-auth-warning DEL
            echo -e "${GREEN}✅ 已清除所有推理任務 key${NC}"
        fi
    else
        echo "   沒有發現卡住的 key"
    fi
else
    echo -e "${YELLOW}⚠️  無法連接 Redis${NC}"
fi

# 4. 查看最近的日誌
echo -e "\n${YELLOW}4️⃣ 最近的日誌 (最後 20 行):${NC}"
sudo journalctl -u brain-parcellation.service -n 20 --no-pager

# 5. 總結
echo -e "\n========================================="
echo -e "${GREEN}✅ 快速檢查完成${NC}"
echo "========================================="
echo ""
echo "💡 如果問題仍未解決，請執行完整診斷:"
echo "   python scripts/diagnose_inference_stuck.py --study-uid <STUDY_UID> --study-id <STUDY_ID>"
echo ""
echo "📋 查看實時日誌:"
echo "   sudo journalctl -u brain-parcellation.service -f"
echo ""

