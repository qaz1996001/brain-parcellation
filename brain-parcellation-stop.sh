#!/bin/bash

set -e

WORK_DIR="/home/david/brain-parcellation"
LOG_DIR="/var/log/brain-parcellation"

mkdir -p "$LOG_DIR"

echo "[$(date)] 开始关闭所有服务..." >> "$LOG_DIR/shutdown.log"

# 停止 Python 进程
echo "[$(date)] 停止后端服务..." >> "$LOG_DIR/shutdown.log"
pkill -f "python3 backend/app/main.py" || true
sleep 1

echo "[$(date)] 停止 Funboost CLI 服务..." >> "$LOG_DIR/shutdown.log"
pkill -f "python3 funboost_cli_user.py" || true
sleep 1

# 停止 Docker Compose
echo "[$(date)] 停止 Docker Compose..." >> "$LOG_DIR/shutdown.log"
cd "$WORK_DIR"
#docker-compose down

echo "[$(date)] 所有服务已关闭" >> "$LOG_DIR/shutdown.log"