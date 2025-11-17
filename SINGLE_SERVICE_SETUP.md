# Brain Parcellation 单一服务安装指南

## 📋 方案说明

使用一个 systemd 服务管理所有组件：
- ✅ Docker Compose
- ✅ FastAPI 后端 (backend/app/main.py)
- ✅ Funboost CLI (funboost_cli_user.py)

启动一个服务，所有组件自动启动；停止一个服务，所有组件自动关闭。

## 🔧 安装步骤

### 1. 检查配置路径

确认以下路径是否正确：

```bash
# 检查 conda 路径
which conda
# 输出应该类似：/opt/miniconda3/bin/conda

# 检查 docker-compose
which docker-compose
# 输出应该类似：/usr/bin/docker-compose
```

如果路径不同，需要编辑以下文件中的相应路径。

### 2. 创建启动和停止脚本

```bash
# 复制脚本到系统目录
sudo cp  /var/www/brain-parcellation/brain-parcellation-start.sh /usr/local/bin/
sudo cp  /var/www/brain-parcellation/brain-parcellation-stop.sh /usr/local/bin/

# 添加执行权限
sudo chmod +x /usr/local/bin/brain-parcellation-start.sh
sudo chmod +x /usr/local/bin/brain-parcellation-stop.sh
```

### 3. 创建日志目录

```bash
sudo mkdir -p /var/log/brain-parcellation
sudo chown root:root /var/log/brain-parcellation
sudo chmod 755 /var/log/brain-parcellation
```

### 4. 复制服务文件

```bash
# 使用推荐的改进版本
sudo cp  /var/www/brain-parcellation/brain-parcellation-improved.service /etc/systemd/system/brain-parcellation.service
```

### 5. 重新加载 systemd 配置

```bash
sudo systemctl daemon-reload
```

### 6. 启用和启动服务

```bash
# 启用开机自启
sudo systemctl enable brain-parcellation.service

# 启动服务
sudo systemctl start brain-parcellation.service
```

## 📊 使用命令

### 查看服务状态

```bash
# 查看当前状态
sudo systemctl status brain-parcellation.service

# 查看详细状态
systemctl show brain-parcellation.service
```

### 查看日志

```bash
# 实时日志（systemd 日志）
sudo journalctl -u brain-parcellation.service -f

# 查看最后 100 行
sudo journalctl -u brain-parcellation.service -n 100

# 查看启动日志
sudo tail -f /var/log/brain-parcellation/startup.log

# 查看后端日志
sudo tail -f /var/log/brain-parcellation/backend.log

# 查看 Funboost 日志
sudo tail -f /var/log/brain-parcellation/funboost.log

# 查看关闭日志
sudo tail -f /var/log/brain-parcellation/shutdown.log
```

### 启动、停止、重启

```bash
# 启动服务
sudo systemctl start brain-parcellation.service

# 停止服务
sudo systemctl stop brain-parcellation.service

# 重启服务
sudo systemctl restart brain-parcellation.service

# 重新加载（不中断服务）
sudo systemctl reload brain-parcellation.service
```

### 禁用或移除

```bash
# 禁用开机自启
sudo systemctl disable brain-parcellation.service

# 停止并禁用
sudo systemctl disable --now brain-parcellation.service
```

## 🎯 便捷别名

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

然后执行：

```bash
source ~/.bashrc
brain-start
brain-status
```

## 🔍 故障排查

### 问题 1：服务启动失败

```bash
# 查看详细错误
sudo journalctl -u brain-parcellation.service --no-pager

# 查看脚本错误
sudo /usr/local/bin/brain-parcellation-start.sh
```

### 问题 2：Docker Compose 找不到

确保在正确的目录运行：

```bash
cd /var/www/brain-parcellation
docker-compose ps
```

### 问题 3：conda 命令找不到

检查 conda 路径：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate tf_2_14
```

如果路径不同，编辑脚本中的 `CONDA_PATH` 变量。

### 问题 4：权限问题

```bash
# 检查脚本权限
ls -la /usr/local/bin/brain-parcellation-*.sh

# 检查日志目录权限
ls -la /var/log/brain-parcellation

# 检查项目目录权限
ls -la /var/www/brain-parcellation
```

### 问题 5：端口占用

如果服务启动但无法访问，检查是否有之前的进程仍在运行：

```bash
# 查找所有 Python 进程
ps aux | grep python3

# 查找占用特定端口的进程
sudo lsof -i :8000  # FastAPI 默认端口
```

## 📝 日志管理建议

### 自动轮转日志

编辑 `/etc/logrotate.d/brain-parcellation`：

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

## 🔐 安全建议

### 1. 改用特定用户运行

编辑服务文件，改为使用普通用户（如 `www-data`）：

```ini
User=www-data
```

同时更新脚本中的权限设置。

### 2. 限制日志大小

在启动脚本中添加日志轮转：

```bash
# 如果日志超过 100MB，进行轮转
for log in /var/log/brain-parcellation/*.log; do
    size=$(stat -f%z "$log" 2>/dev/null || stat -c%s "$log")
    if [ "$size" -gt 104857600 ]; then  # 100MB
        mv "$log" "$log.$(date +%s)"
        gzip "$log".* 2>/dev/null || true
    fi
done
```

## 📋 完整工作流

### 首次安装

```bash
# 1. 检查路径
which conda
which docker-compose

# 2. 编辑脚本（如需要）
sudo vim /usr/local/bin/brain-parcellation-start.sh
sudo vim /usr/local/bin/brain-parcellation-stop.sh

# 3. 设置权限
sudo chmod +x /usr/local/bin/brain-parcellation-*.sh

# 4. 创建日志目录
sudo mkdir -p /var/log/brain-parcellation

# 5. 安装服务
sudo cp  /var/www/brain-parcellation/brain-parcellation-improved.service /etc/systemd/system/brain-parcellation.service
sudo systemctl daemon-reload

# 6. 启用和启动
sudo systemctl enable brain-parcellation.service
sudo systemctl start brain-parcellation.service

# 7. 验证
sudo systemctl status brain-parcellation.service
sudo tail -f /var/log/brain-parcellation/startup.log
```

### 日常使用

```bash
# 启动
brain-start

# 查看状态
brain-status

# 查看日志
brain-logs

# 停止
brain-stop
```

## ✅ 验证清单

启动后检查以下内容：

- [ ] 服务状态为 `active (running)`
- [ ] Docker 容器已启动：`docker ps`
- [ ] 后端服务可访问（通常是 http://localhost:8000）
- [ ] 没有错误日志：`sudo journalctl -u brain-parcellation.service --no-pager | grep -i error`
- [ ] 启动脚本已正确执行：`cat /var/log/brain-parcellation/startup.log`
