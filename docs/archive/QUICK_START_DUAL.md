# ⚠️ [已棄用] 快速開始：同時運行 Production 和 Testing 實例

> **⚠️ 文檔已棄用 (DEPRECATED)**
>
> 此文檔已被整合至新的統一部署指南。請改用：
> **[DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md](../../DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md)**
>
> **棄用原因**：
> - 此文檔僅涵蓋快速啟動，缺少 GPU 資源管理方案
> - 新指南整合了完整的 GPU 資源保護策略（分布式控頻）
> - 提供明確的決策樹和驗證腳本
> - 遵循 Linus Torvalds 和 Donald Knuth 原則
>
> **遷移建議**：
> - 閱讀新指南的「快速開始（推薦方案 B）」章節
> - 按照新的驗證步驟確保 GPU 資源不競爭
> - 使用提供的驗證腳本 `scripts/verify-dual-deployment.sh`
>
> ---
>
> 以下為歷史存檔內容，僅供參考：

# 快速開始：同時運行 Production 和 Testing 實例

## 🚀 快速啟動

### 1. 準備配置檔案（已提供）
```bash
# 檢查配置檔案是否存在
ls -la .env.production .env.testing
```

### 2. 啟動兩個實例
```bash
# 給予腳本執行權限（首次使用）
chmod +x deploy-dual.sh

# 啟動兩個實例
./deploy-dual.sh start
```

### 3. 驗證運行狀態
```bash
# 檢查狀態
./deploy-dual.sh status

# 或手動檢查
curl http://localhost:8000/health  # Production
curl http://localhost:8001/health  # Testing
```

## 📊 端口分配

| 服務 | Production | Testing | 用途 |
|------|-----------|---------|------|
| **Backend API** | 8000 | 8001 | FastAPI 應用 |
| **RabbitMQ** | 5672 | 5673 | 訊息佇列 |
| **RabbitMQ UI** | 15672 | 15673 | 管理介面 |
| **Redis** | 6379 | 6380 | 快取服務 |
| **PostgreSQL** | 15433 | 15433 | 資料庫（共用端口，不同資料庫名稱） |

## 🔧 常用命令

### 啟動/停止

```bash
# 啟動所有實例
./deploy-dual.sh start

# 停止所有實例
./deploy-dual.sh stop

# 重啟所有實例
./deploy-dual.sh restart

# 只啟動 Production
./deploy-dual.sh start-prod

# 只啟動 Testing
./deploy-dual.sh start-test

# 只停止 Production
./deploy-dual.sh stop-prod

# 只停止 Testing
./deploy-dual.sh stop-test
```

### 狀態檢查

```bash
# 查看完整狀態
./deploy-dual.sh status

# 查看 Docker 容器
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 查看日誌
docker-compose --project-name brain-prod logs -f
docker-compose --project-name brain-test logs -f
```

## 🗄️ 資料庫管理

### 連接資料庫

```bash
# Production 資料庫
docker exec -it brain-prod-db_server-1 psql -U postgres_n -d dicom

# Testing 資料庫
docker exec -it brain-test-db_server-1 psql -U postgres_n -d dicom_testing
```

### 備份資料

```bash
# 備份 Production 資料庫
docker exec brain-prod-db_server-1 pg_dump -U postgres_n dicom > backup_prod_$(date +%Y%m%d).sql

# 備份 Testing 資料庫
docker exec brain-test-db_server-1 pg_dump -U postgres_n dicom_testing > backup_test_$(date +%Y%m%d).sql
```

## 🔍 監控和除錯

### 檢查端口佔用

```bash
# Windows
netstat -ano | findstr "8000 8001 5672 5673 6379 6380"

# Linux/Mac
netstat -tuln | grep -E '8000|8001|5672|5673|6379|6380'
```

### 查看應用日誌

```bash
# Backend 日誌（如果使用啟動腳本）
tail -f /var/log/brain-parcellation/prod/backend.log
tail -f /var/log/brain-parcellation/test/backend.log

# Docker 容器日誌
docker logs -f brain-prod-rabbitmq_server-1
docker logs -f brain-test-redis_server-1
```

### 檢查環境變數

```bash
# 檢查 Production 容器的環境變數
docker exec brain-prod-rabbitmq_server-1 env | grep ENV

# 檢查 Testing 容器的環境變數
docker exec brain-test-rabbitmq_server-1 env | grep ENV
```

## 🧪 測試驗證

### API 測試

```bash
# Production 健康檢查
curl -X GET http://localhost:8000/health

# Testing 健康檢查
curl -X GET http://localhost:8001/health

# 測試 API 端點（根據實際 API 調整）
curl -X GET http://localhost:8000/api/v1/your-endpoint
curl -X GET http://localhost:8001/api/v1/your-endpoint
```

### 資料庫測試

```bash
# 驗證資料隔離
# Production 應該不會看到 testing 資料
docker exec brain-prod-db_server-1 psql -U postgres_n -c "\l" | grep dicom

# Testing 應該有獨立的資料庫
docker exec brain-test-db_server-1 psql -U postgres_n -c "\l" | grep dicom_testing
```

## ⚠️ 常見問題

### 1. 端口已被佔用

**問題**: `Error: port is already allocated`

**解決方案**:
```bash
# 檢查佔用端口的程序
netstat -ano | findstr "8000"  # Windows
lsof -i :8000                  # Linux/Mac

# 停止衝突的服務
./deploy-dual.sh stop

# 或修改 .env.testing 中的端口
```

### 2. 容器啟動失敗

**問題**: Docker 容器無法啟動

**解決方案**:
```bash
# 查看詳細錯誤
docker-compose --project-name brain-prod logs

# 檢查配置檔案
cat .env.production
cat .env.testing

# 重新建立容器
docker-compose --project-name brain-prod up -d --force-recreate
```

### 3. 資料庫連接失敗

**問題**: Backend 無法連接到資料庫

**解決方案**:
```bash
# 確認資料庫容器運行中
docker ps | grep postgres

# 檢查資料庫日誌
docker logs brain-prod-db_server-1

# 測試資料庫連接
docker exec brain-prod-db_server-1 psql -U postgres_n -d dicom -c "SELECT 1"
```

### 4. 環境變數未生效

**問題**: 應用仍然使用錯誤的環境

**解決方案**:
```bash
# 確認環境變數正確設置
docker exec brain-prod-rabbitmq_server-1 env | grep ENV
# 應該顯示: ENV=production

docker exec brain-test-rabbitmq_server-1 env | grep ENV
# 應該顯示: ENV=testing

# 重新啟動容器
./deploy-dual.sh restart
```

## 🔄 更新和維護

### 更新配置

```bash
# 1. 修改配置檔案
nano .env.production
nano .env.testing

# 2. 重新啟動服務
./deploy-dual.sh restart
```

### 清理 Testing 環境

```bash
# 停止並刪除 testing 容器和資料
docker-compose --project-name brain-test down -v

# 重新啟動乾淨的 testing 環境
./deploy-dual.sh start-test
```

### 僅更新程式碼（不重啟 Docker）

```bash
# 停止 Python 應用但保持 Docker 服務運行
# （根據實際啟動方式調整）
pkill -f "python.*backend.app.main"

# 重新啟動應用
ENV=production APP_PORT=8000 python backend/app/main.py &
ENV=testing APP_PORT=8001 python backend/app/main.py &
```

## 📚 相關文檔

- 詳細部署指南: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- 環境配置說明: [.env.example](.env.example)
- OpenSpec 設計文檔: [openspec/changes/add-environment-support/](openspec/changes/add-environment-support/)

## 🎯 檢查清單

部署前檢查：
- [ ] `.env.production` 和 `.env.testing` 已配置
- [ ] 所需端口未被佔用（8000, 8001, 5672, 5673, 6379, 6380）
- [ ] Docker 已安裝並運行
- [ ] 有足夠的系統資源（建議至少 4GB RAM）

部署後驗證：
- [ ] `./deploy-dual.sh status` 顯示所有容器運行中
- [ ] `curl http://localhost:8000/health` 返回 production 環境
- [ ] `curl http://localhost:8001/health` 返回 testing 環境
- [ ] 資料庫可以正常連接
- [ ] RabbitMQ 管理介面可訪問（http://localhost:15672 和 http://localhost:15673）

---

**提示**: 如果這是首次部署，建議先使用 `./deploy-dual.sh start-prod` 啟動 production，驗證無誤後再啟動 testing 實例。
