# Brain Parcellation 單一服務安裝指南

## 📋 方案說明

使用一個 systemd 服務管理所有元件：
- ✅ Docker Compose
- ✅ FastAPI 後端 (backend/app/main.py)
- ✅ Funboost CLI (funboost_cli_user.py)

啟動一個服務，所有元件自動啟動；停止一個服務，所有元件自動關閉。

## 🔧 安裝步驟

### 1. 檢查配置路徑

確認以下路徑是否正確：

```bash
# 檢查 conda 路徑
which conda
# 輸出應該類似：/home/david/miniconda3/bin/cond

# 檢查 docker-compose
which docker-compose
# 輸出應該類似：/usr/bin/docker-compose
```

如果路徑不同，需要編輯以下檔中的相應路徑。

### 2. 創建啟動和停止腳本

```bash
# 複製腳本到系統目錄
sudo cp  /home/david/brain-parcellation/brain-parcellation-start.sh /usr/local/bin/
sudo cp  /home/david/brain-parcellation/brain-parcellation-stop.sh /usr/local/bin/

# 添加執行許可權
sudo chmod +x /usr/local/bin/brain-parcellation-start.sh
sudo chmod +x /usr/local/bin/brain-parcellation-stop.sh
```

### 3. 創建日誌目錄

```bash
sudo mkdir -p /var/log/brain-parcellation
sudo chown root:root /var/log/brain-parcellation
sudo chmod 755 /var/log/brain-parcellation
```

### 4. 複製服務檔

```bash
# 使用推薦的改進版本
sudo cp  /home/david/brain-parcellation/brain-parcellation-improved.service /etc/systemd/system/brain-parcellation.service
```

### 5. 重新載入 systemd 配置

```bash
sudo systemctl daemon-reload
```

### 6. 啟用和啟動服務

```bash
# 啟用開機自啟
sudo systemctl enable brain-parcellation.service

# 啟動服務
sudo systemctl start brain-parcellation.service
```

## 📊 使用命令

### 查看服務狀態

```bash
# 查看當前狀態
sudo systemctl status brain-parcellation.service

# 查看詳細狀態
systemctl show brain-parcellation.service
```

### 查看日誌

```bash
# 即時日誌（systemd 日誌）
sudo journalctl -u brain-parcellation.service -f

# 查看最後 100 行
sudo journalctl -u brain-parcellation.service -n 100

# 查看開機記錄
sudo tail -f /var/log/brain-parcellation/startup.log

# 查看後端日誌
sudo tail -f /var/log/brain-parcellation/backend.log

# 查看 Funboost 日誌
sudo tail -f /var/log/brain-parcellation/funboost.log

# 查看關閉日誌
sudo tail -f /var/log/brain-parcellation/shutdown.log
```

### 啟動、停止、重啟

```bash
# 啟動服務
sudo systemctl start brain-parcellation.service

# 停止服務
sudo systemctl stop brain-parcellation.service

# 重啟服務
sudo systemctl restart brain-parcellation.service

# 重新載入（不中斷服務）
sudo systemctl reload brain-parcellation.service
```

### 禁用或移除

```bash
# 禁用開機自啟
sudo systemctl disable brain-parcellation.service

# 停止並禁用
sudo systemctl disable --now brain-parcellation.service
```

## 🎯 便捷別名

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
# Brain Parcellation 快捷命令
alias brain-start="sudo systemctl start brain-parcellation.service"
alias brain-stop="sudo systemctl stop brain-parcellation.service"
alias brain-restart="sudo systemctl restart brain-parcellation.service"
alias brain-status="sudo systemctl status brain-parcellation.service"
alias brain-logs="sudo journalctl -u brain-parcellation.service -f"
alias brain-backend-log="sudo tail -f /var/log/brain-parcellation/backend.log"
alias brain-funboost-log="sudo tail -f /var/log/brain-parcellation/funboost.log"
```

然後執行：

```bash
source ~/.bashrc
brain-start
brain-status
```

## 🔍 故障排查

### 問題 1：服務啟動失敗

```bash
# 查看詳細錯誤
sudo journalctl -u brain-parcellation.service --no-pager

# 查看腳本錯誤
sudo /usr/local/bin/brain-parcellation-start.sh
```

### 問題 2：Docker Compose 找不到

確保在正確的目錄運行：

```bash
cd /home/david/brain-parcellation
docker-compose ps
```

### 問題 3：conda 命令找不到

檢查 conda 路徑：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate tf_2_14
```

如果路徑不同，編輯腳本中的 `CONDA_PATH` 變數。

### 問題 4：許可權問題

```bash
# 檢查腳本許可權
ls -la /usr/local/bin/brain-parcellation-*.sh

# 檢查日誌目錄許可權
ls -la /var/log/brain-parcellation

# 檢查項目目錄許可權
ls -la /home/david/brain-parcellation
```

### 問題 5：埠佔用

如果服務啟動但無法訪問，檢查是否有之前的進程仍在運行：

```bash
# 查找所有 Python 進程
ps aux | grep python3

# 查找佔用特定埠的進程
sudo lsof -i :8000  # FastAPI 默認埠
```

## 📝 日誌管理建議

### 自動輪轉日誌

編輯 `/etc/logrotate.d/brain-parcellation`：

```bash
sudo cat > /etc/logrotate.d/brain-parcellation << 'EOF'
/var/log/brain-parcellation/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 root root
    sharedscripts
}
EOF
```

## 🔐 安全建議

### 1. 改用特定用戶運行

編輯服務檔，改為使用普通用戶（如 `www-data`）：

```ini
User=www-data
```

同時更新腳本中的許可權設置。

### 2. 限制日誌大小

在啟動腳本中添加日誌輪轉：

```bash
# 如果日誌超過 100MB，進行輪轉
for log in /var/log/brain-parcellation/*.log; do
    size=$(stat -f%z "$log" 2>/dev/null || stat -c%s "$log")
    if [ "$size" -gt 104857600 ]; then  # 100MB
        mv "$log" "$log.$(date +%s)"
        gzip "$log".* 2>/dev/null || true
    fi
done
```

## 📋 完整工作流

### 首次安裝

```bash
# 1. 檢查路徑
which conda
which docker-compose

# 2. 編輯腳本（如需要）
sudo vim /usr/local/bin/brain-parcellation-start.sh
sudo vim /usr/local/bin/brain-parcellation-stop.sh

# 3. 設置許可權
sudo chmod +x /usr/local/bin/brain-parcellation-*.sh

# 4. 創建日誌目錄
sudo mkdir -p /var/log/brain-parcellation

# 5. 安裝服務
sudo cp  /home/david/brain-parcellation/brain-parcellation-improved.service /etc/systemd/system/brain-parcellation.service
sudo systemctl daemon-reload

# 6. 啟用和啟動
sudo systemctl enable brain-parcellation.service
sudo systemctl start brain-parcellation.service

# 7. 驗證
sudo systemctl status brain-parcellation.service
sudo tail -f /var/log/brain-parcellation/startup.log
```

### 日常使用

```bash
# 啟動
brain-start

# 查看狀態
brain-status

# 查看日誌
brain-logs

# 停止
brain-stop
```

## ✅ 驗證清單

啟動後檢查以下內容：

- [ ] 服務狀態為 `active (running)`
- [ ] Docker 容器已啟動：`docker ps`
- [ ] 後端服務可訪問（通常是 http://localhost:8000）
- [ ] 沒有錯誤日誌：`sudo journalctl -u brain-parcellation.service --no-pager | grep -i error`
- [ ] 啟動腳本已正確執行：`cat /var/log/brain-parcellation/startup.log`
