# 同時部署 Production 和 Testing 實例指南

## 方案概述

本指南說明如何在同一台機器上同時運行 production 和 testing 兩個獨立實例。

## 🎯 設計原則

- **端口隔離**: 使用不同端口避免衝突
- **資料隔離**: 資料庫和資料路徑完全分離
- **容器隔離**: 使用 Docker Compose project name 區分容器
- **日誌隔離**: 不同環境使用不同日誌目錄

## 方案 1: 使用環境變數和不同端口（推薦）

### 1.1 準備配置文件

創建兩個獨立的 `.env` 檔案：

**`.env.production`**
```bash
# Production Environment
ENV=production

# Application Ports
APP_PORT=8000

# RabbitMQ Configuration
RABBITMQ_DEFAULT_VHOST=prod_vhost
RABBITMQ_DEFAULT_USER=admin
RABBITMQ_DEFAULT_PASS=prod_password
RABBITMQ_ERLANG_COOKIE=prod_erlang_cookie
RABBITMQ_PORT=5672
RABBITMQ_UI_PORT=15672

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_USERNAME=default
REDIS_PASSWORD=prod_redis_pass
REDIS_DB_FASTAPI_CACHE=0

# PostgreSQL Configuration
POSTGRES_DB=dicom
POSTGRES_USER=postgres_n
POSTGRES_PASSWORD=postgres_p
POSTGRES_PORT=15433
```

**`.env.testing`**
```bash
# Testing Environment
ENV=testing

# Application Ports (避免與 production 衝突)
APP_PORT=8001

# RabbitMQ Configuration (不同端口)
RABBITMQ_DEFAULT_VHOST=test_vhost
RABBITMQ_DEFAULT_USER=admin
RABBITMQ_DEFAULT_PASS=test_password
RABBITMQ_ERLANG_COOKIE=test_erlang_cookie
RABBITMQ_PORT=5673
RABBITMQ_UI_PORT=15673

# Redis Configuration (不同端口)
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_USERNAME=default
REDIS_PASSWORD=test_redis_pass
REDIS_DB_FASTAPI_CACHE=1

# PostgreSQL Configuration (相同端口但不同資料庫名稱)
POSTGRES_DB=dicom_testing
POSTGRES_USER=postgres_n
POSTGRES_PASSWORD=postgres_p
POSTGRES_PORT=15433
```

### 1.2 啟動 Production 實例

```bash
# 使用 production 配置啟動 Docker 服務
docker-compose --env-file .env.production --project-name brain-prod up -d

# 啟動 backend 服務
ENV=production ./brain-parcellation-start.sh production
```

### 1.3 啟動 Testing 實例

```bash
# 使用 testing 配置啟動 Docker 服務
docker-compose --env-file .env.testing --project-name brain-test up -d

# 啟動 backend 服務（使用不同端口）
ENV=testing APP_PORT=8001 ./brain-parcellation-start.sh testing
```

### 1.4 驗證兩個實例

```bash
# 檢查 production 健康狀態
curl http://localhost:8000/health
# 預期: {"status": "healthy", "environment": "production", "log_level": "INFO"}

# 檢查 testing 健康狀態
curl http://localhost:8001/health
# 預期: {"status": "healthy", "environment": "testing", "log_level": "DEBUG"}
```

## 方案 2: 使用獨立的 Docker Compose 檔案

### 2.1 創建分離的 Compose 檔案

**`docker-compose.production.yml`**
```yaml
services:
  rabbitmq_prod:
    restart: always
    container_name: rabbitmq_prod
    image: rabbitmq:3-management
    environment:
      - ENV=production
      - RABBITMQ_DEFAULT_VHOST=${RABBITMQ_DEFAULT_VHOST}
      - RABBITMQ_DEFAULT_USER=${RABBITMQ_DEFAULT_USER}
      - RABBITMQ_DEFAULT_PASS=${RABBITMQ_DEFAULT_PASS}
      - RABBITMQ_ERLANG_COOKIE=${RABBITMQ_ERLANG_COOKIE}
    volumes:
      - volume_rabbitmq_prod:/var/lib/rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"

  redis_prod:
    restart: always
    container_name: redis_prod
    image: redis:7.4-alpine
    environment:
      - ENV=production
    ports:
      - "6379:6379"
    volumes:
      - volume_redis_prod:/data

  db_prod:
    image: postgres
    restart: always
    container_name: db_prod
    volumes:
      - volume_db_prod:/var/lib/postgresql/data
    environment:
      - ENV=production
      - POSTGRES_DB=dicom
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    ports:
      - "15433:5432"

volumes:
  volume_rabbitmq_prod:
  volume_redis_prod:
  volume_db_prod:
```

**`docker-compose.testing.yml`**
```yaml
services:
  rabbitmq_test:
    restart: always
    container_name: rabbitmq_test
    image: rabbitmq:3-management
    environment:
      - ENV=testing
      - RABBITMQ_DEFAULT_VHOST=${RABBITMQ_DEFAULT_VHOST}
      - RABBITMQ_DEFAULT_USER=${RABBITMQ_DEFAULT_USER}
      - RABBITMQ_DEFAULT_PASS=${RABBITMQ_DEFAULT_PASS}
      - RABBITMQ_ERLANG_COOKIE=${RABBITMQ_ERLANG_COOKIE}
    volumes:
      - volume_rabbitmq_test:/var/lib/rabbitmq
    ports:
      - "5673:5672"
      - "15673:15672"

  redis_test:
    restart: always
    container_name: redis_test
    image: redis:7.4-alpine
    environment:
      - ENV=testing
    ports:
      - "6380:6379"
    volumes:
      - volume_redis_test:/data

  db_test:
    image: postgres
    restart: always
    container_name: db_test
    volumes:
      - volume_db_test:/var/lib/postgresql/data
    environment:
      - ENV=testing
      - POSTGRES_DB=dicom_testing
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    ports:
      - "15434:5432"

volumes:
  volume_rabbitmq_test:
  volume_redis_test:
  volume_db_test:
```

### 2.2 啟動命令

```bash
# Production
docker-compose -f docker-compose.production.yml --env-file .env.production up -d
ENV=production APP_PORT=8000 python backend/app/main.py

# Testing
docker-compose -f docker-compose.testing.yml --env-file .env.testing up -d
ENV=testing APP_PORT=8001 python backend/app/main.py
```

## 方案 3: 使用啟動腳本自動化（最簡單）

### 3.1 創建統一管理腳本

**`deploy-dual.sh`**
```bash
#!/bin/bash

set -e

ACTION=$1  # start|stop|restart|status

case $ACTION in
  start)
    echo "Starting Production instance..."
    ENV=production APP_PORT=8000 docker-compose --env-file .env.production --project-name brain-prod up -d
    ENV=production APP_PORT=8000 ./brain-parcellation-start.sh production &

    echo "Starting Testing instance..."
    ENV=testing APP_PORT=8001 docker-compose --env-file .env.testing --project-name brain-test up -d
    ENV=testing APP_PORT=8001 ./brain-parcellation-start.sh testing &

    echo "Both instances started successfully"
    ;;

  stop)
    echo "Stopping both instances..."
    docker-compose --project-name brain-prod down
    docker-compose --project-name brain-test down
    ./brain-parcellation-stop.sh
    echo "Both instances stopped"
    ;;

  restart)
    $0 stop
    sleep 3
    $0 start
    ;;

  status)
    echo "=== Production Instance ==="
    curl -s http://localhost:8000/health | python -m json.tool
    docker-compose --project-name brain-prod ps

    echo ""
    echo "=== Testing Instance ==="
    curl -s http://localhost:8001/health | python -m json.tool
    docker-compose --project-name brain-test ps
    ;;

  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
```

### 3.2 使用方式

```bash
# 給予執行權限
chmod +x deploy-dual.sh

# 啟動兩個實例
./deploy-dual.sh start

# 檢查狀態
./deploy-dual.sh status

# 停止兩個實例
./deploy-dual.sh stop

# 重啟兩個實例
./deploy-dual.sh restart
```

## 🔍 資源隔離清單

| 資源 | Production | Testing | 隔離方式 |
|------|-----------|---------|---------|
| **Backend Port** | 8000 | 8001 | 環境變數 APP_PORT |
| **Database** | dicom | dicom_testing | 資料庫名稱後綴 |
| **RabbitMQ Port** | 5672 | 5673 | 不同端口映射 |
| **RabbitMQ UI** | 15672 | 15673 | 不同端口映射 |
| **Redis Port** | 6379 | 6380 | 不同端口映射 |
| **PostgreSQL Port** | 15433 | 15434 (選項) | 可共用或分離 |
| **Docker Containers** | brain-prod_* | brain-test_* | Project name |
| **Docker Volumes** | *_prod | *_test | Volume 名稱後綴 |
| **日誌目錄** | /var/log/brain-parcellation/prod/ | /var/log/brain-parcellation/test/ | 路徑分離 |

## 🚨 注意事項

### 1. 端口衝突檢查
```bash
# 檢查端口佔用
netstat -tuln | grep -E '8000|8001|5672|5673|6379|6380|15433|15434'
```

### 2. 資料庫連接字串調整

如果 Testing 使用獨立的 PostgreSQL 端口：

修改 `backend/app/database.py`:
```python
import os

_ENV = get_environment()
_CONFIG = get_config()

# 根據環境使用不同端口
_DB_PORT = "15433" if _ENV == "production" else "15434"
_DB_NAME = "dicom" if _ENV == "production" else "dicom_testing"

_CONNECTION_STRING = f"postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:{_DB_PORT}/{_DB_NAME}"
```

### 3. 日誌隔離

修改 `brain-parcellation-start.sh`:
```bash
# 根據環境設定不同日誌目錄
if [[ "$ENVIRONMENT" == "testing" ]]; then
    LOG_DIR="/var/log/brain-parcellation/test"
else
    LOG_DIR="/var/log/brain-parcellation/prod"
fi

mkdir -p "$LOG_DIR"
```

### 4. 資源監控

```bash
# 監控兩個實例的資源使用
docker stats brain-prod_rabbitmq_1 brain-prod_redis_1 brain-prod_db_1 \
              brain-test_rabbitmq_1 brain-test_redis_1 brain-test_db_1
```

## 📊 驗證檢查清單

```bash
# 1. 檢查 Docker 容器
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. 檢查 Production 健康狀態
curl http://localhost:8000/health

# 3. 檢查 Testing 健康狀態
curl http://localhost:8001/health

# 4. 檢查資料庫連接
docker exec brain-prod_db_1 psql -U postgres_n -c "\l" | grep dicom
docker exec brain-test_db_1 psql -U postgres_n -c "\l" | grep dicom_testing

# 5. 檢查日誌
tail -f /var/log/brain-parcellation/prod/startup.log
tail -f /var/log/brain-parcellation/test/startup.log
```

## 🔄 切換和維護

### 僅重啟 Production
```bash
docker-compose --project-name brain-prod restart
ENV=production ./brain-parcellation-start.sh production
```

### 僅重啟 Testing
```bash
docker-compose --project-name brain-test restart
ENV=testing ./brain-parcellation-start.sh testing
```

### 清理 Testing 環境（保留 Production）
```bash
docker-compose --project-name brain-test down -v
```

## 🎯 最佳實踐

1. **使用 systemd 服務管理**（適用於 Linux）
2. **配置監控告警**（Prometheus + Grafana）
3. **定期備份 Production 資料**
4. **Testing 環境可定期重置**
5. **使用 Nginx 反向代理統一入口**

## 📚 參考

- OpenSpec 提案: `openspec/changes/add-environment-support/`
- 環境配置: `backend/app/config/environments.py`
- Docker Compose: `docker-compose.yml`
