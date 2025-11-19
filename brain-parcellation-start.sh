#!/bin/bash

set -e

# 设置环境
WORK_DIR="/home/david/brain-parcellation"
CONDA_PATH="/home/david/miniconda3"
LOG_DIR="/var/log/brain-parcellation"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 激活 conda 环境
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate tf_2_14

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR"

# 启动 Docker Compose
echo "[$(date)] 启动 Docker Compose..." >> "$LOG_DIR/startup.log"
#docker compose restart -d
# 等待 Docker 容器就绪
sleep 3

# 启动后端服务
echo "[$(date)] 启动后端服务 (backend/app/main.py)..." >> "$LOG_DIR/startup.log"
python3 backend/app/main.py >> "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID" >> "$LOG_DIR/startup.log"

# 等待后端启动
sleep 2

# 启动 Funboost CLI 服务
echo "[$(date)] 启动 Funboost CLI (funboost_cli_user.py)..." >> "$LOG_DIR/startup.log"
python3 funboost_cli_user.py >> "$LOG_DIR/funboost.log" 2>&1 &
FUNBOOST_PID=$!
echo "Funboost PID: $FUNBOOST_PID" >> "$LOG_DIR/startup.log"

echo "[$(date)] 所有服务启动完成" >> "$LOG_DIR/startup.log"

# 等待任意进程终止
wait -n