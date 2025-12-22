# 雙實例部署摘要

## ✅ 已完成配置

您現在可以同時運行 **Production** 和 **Testing** 兩個獨立實例。

## 📦 提供的文件

### 配置文件
1. **`.env.production`** - Production 環境配置
2. **`.env.testing`** - Testing 環境配置
3. **`.env.example`** - 配置範例和說明（已更新）

### 腳本工具
4. **`deploy-dual.sh`** - 雙實例管理腳本

### 文檔
5. **`DEPLOYMENT_GUIDE.md`** - 完整部署指南（3種方案）
6. **`QUICK_START_DUAL.md`** - 快速開始指南

## 🚀 最快啟動方式

```bash
# 1. 給予腳本執行權限（首次使用）
chmod +x deploy-dual.sh

# 2. 啟動兩個實例
./deploy-dual.sh start

# 3. 檢查狀態
./deploy-dual.sh status
```

## 📊 資源隔離

| 項目 | Production | Testing |
|------|-----------|---------|
| **Backend** | localhost:8000 | localhost:8001 |
| **Database** | dicom | dicom_testing |
| **RabbitMQ** | :5672 | :5673 |
| **RabbitMQ UI** | :15672 | :15673 |
| **Redis** | :6379 | :6380 |
| **環境變數** | ENV=production | ENV=testing |
| **日誌級別** | INFO | DEBUG |

## 🎯 健康檢查

```bash
# Production
curl http://localhost:8000/health
# 預期: {"status": "healthy", "environment": "production", "log_level": "INFO"}

# Testing
curl http://localhost:8001/health
# 預期: {"status": "healthy", "environment": "testing", "log_level": "DEBUG"}
```

## 🔧 常用命令

```bash
./deploy-dual.sh start        # 啟動所有
./deploy-dual.sh stop         # 停止所有
./deploy-dual.sh restart      # 重啟所有
./deploy-dual.sh status       # 查看狀態

./deploy-dual.sh start-prod   # 僅啟動 Production
./deploy-dual.sh start-test   # 僅啟動 Testing
./deploy-dual.sh stop-prod    # 僅停止 Production
./deploy-dual.sh stop-test    # 僅停止 Testing
```

## 📝 注意事項

### 端口檢查
部署前確保端口未被佔用：
```bash
# Windows
netstat -ano | findstr "8000 8001 5672 5673 6379 6380"

# Linux/Mac
netstat -tuln | grep -E '8000|8001|5672|5673|6379|6380'
```

### 資料庫隔離
- Production 和 Testing 使用**完全獨立**的資料庫
- Production: `dicom`
- Testing: `dicom_testing`
- **絕對不會互相污染資料**

### Docker 容器命名
- Production: `brain-prod-*`
- Testing: `brain-test-*`

## 🔍 故障排除

### 問題：端口衝突
```bash
# 檢查佔用
netstat -ano | findstr "8000"  # Windows
lsof -i :8000                  # Linux/Mac

# 解決：停止衝突的服務或修改 .env.testing 中的端口
```

### 問題：容器無法啟動
```bash
# 查看日誌
docker-compose --project-name brain-prod logs
docker-compose --project-name brain-test logs

# 重建容器
./deploy-dual.sh stop
./deploy-dual.sh start
```

### 問題：環境變數未生效
```bash
# 檢查容器環境變數
docker exec brain-prod-rabbitmq_server-1 env | grep ENV
docker exec brain-test-rabbitmq_server-1 env | grep ENV

# 應該分別顯示 production 和 testing
```

## 📚 詳細資訊

- **完整部署指南**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **快速開始**: [QUICK_START_DUAL.md](QUICK_START_DUAL.md)
- **環境配置原理**: [openspec/changes/add-environment-support/](openspec/changes/add-environment-support/)

## ✨ 優勢

✅ **完全隔離** - 資料庫、端口、容器完全分離
✅ **簡單管理** - 一個腳本控制所有
✅ **靈活切換** - 可獨立啟停任一環境
✅ **安全測試** - Testing 環境完全不影響 Production
✅ **快速驗證** - 同時驗證兩個環境的運行狀態

---

**下一步**: 閱讀 [QUICK_START_DUAL.md](QUICK_START_DUAL.md) 開始部署！
