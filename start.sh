#!/bin/bash

# 腦部分割專案啟動腳本
# Author: [Your Name]
# Date: $(date +%Y-%m-%d)

# 設定變數 (根據你的環境調整)
#PROJECT_DIR="/mnt/d/00_Chen/Task04_git"
CONDA_ENV="tf_2_14"

# 顏色輸出設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 錯誤處理函數
error_exit() {
    echo -e "${RED}錯誤: $1${NC}" >&2
    exit 1
}

# 資訊輸出函數
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 檢查目錄是否存在
check_directory() {
    if [ ! -d "$PROJECT_DIR" ]; then
        error_exit "專案目錄不存在: $PROJECT_DIR"
    fi
}

# 檢查並初始化 conda
init_conda() {
    # 嘗試找到 conda 的初始化腳本
    local conda_paths=(
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/anaconda3/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
    )

    local conda_found=false
    for path in "${conda_paths[@]}"; do
        if [ -f "$path" ]; then
            info "找到 conda 初始化腳本: $path"
            source "$path"
            conda_found=true
            break
        fi
    done

    if [ "$conda_found" = false ]; then
        # 嘗試直接使用 conda 命令
        if command -v conda &> /dev/null; then
            info "使用系統中的 conda 命令"
            # 初始化 conda for bash
            eval "$(conda shell.bash hook)" 2>/dev/null || true
        else
            error_exit "找不到 conda 安裝，請確認 conda 已正確安裝"
        fi
    fi
}

# 檢查 conda 環境
check_conda_env() {
    if ! conda info --envs | grep -q "$CONDA_ENV" 2>/dev/null; then
        warning "Conda 環境 '$CONDA_ENV' 可能不存在，將嘗試繼續執行"
        echo "可用的 conda 環境："
        conda info --envs 2>/dev/null || echo "無法列出 conda 環境"
    fi
}

# 設定環境函數
setup_environment() {
    info "切換到專案目錄: $PROJECT_DIR"
    cd "$PROJECT_DIR" || error_exit "無法切換到專案目錄"

    info "初始化 conda..."
    init_conda

    info "激活 conda 環境: $CONDA_ENV"
    # 嘗試多種方式激活環境
    if ! conda activate "$CONDA_ENV" 2>/dev/null; then
        if ! source activate "$CONDA_ENV" 2>/dev/null; then
            warning "無法激活 conda 環境 '$CONDA_ENV'，使用當前環境"
        fi
    fi

    info "設定 PYTHONPATH"
    export PYTHONPATH=$(pwd)

    # 顯示當前環境資訊
    info "當前 Python 版本: $(python3 --version 2>/dev/null || echo '未知')"
    info "當前工作目錄: $(pwd)"
}

# 啟動後端服務
start_backend() {
    info "啟動後端服務..."
    if [ -f "backend/app/main.py" ]; then
        python3 backend/app/main.py &
        BACKEND_PID=$!
        info "後端服務已啟動 (PID: $BACKEND_PID)"
    else
        warning "找不到後端主程式: backend/app/main.py"
    fi
}

# 啟動 CLI 使用者介面
start_cli() {
    info "啟動 CLI 使用者介面..."
    if [ -f "funboost_cli_user.py" ]; then
        python3 funboost_cli_user.py &
        CLI_PID=$!
        info "CLI 使用者介面已啟動 (PID: $CLI_PID)"
    else
        warning "找不到 CLI 程式: funboost_cli_user.py"
    fi
}

# 清理函數（當腳本被中斷時）
cleanup() {
    info "正在清理程序..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        info "後端服務已停止"
    fi
    if [ ! -z "$CLI_PID" ]; then
        kill $CLI_PID 2>/dev/null
        info "CLI 使用者介面已停止"
    fi
    exit 0
}

# 設定訊號處理
trap cleanup SIGINT SIGTERM

# 主要執行流程
main() {
    info "開始啟動腦部分割系統..."

    # 檢查先決條件
    check_directory
    init_conda
    check_conda_env

    # 設定環境
    setup_environment

    # 啟動服務
    start_backend
    sleep 2  # 給後端一些時間啟動
    start_cli

    info "所有服務已啟動完成"
    info "按 Ctrl+C 停止所有服務"

    # 等待所有背景程序
    wait
}

# 顯示使用說明
show_help() {
    echo "腦部分割系統啟動腳本"
    echo "用法: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  -h, --help     顯示此幫助訊息"
    echo "  -b, --backend  只啟動後端服務"
    echo "  -c, --cli      只啟動 CLI 介面"
    echo ""
}

# 處理命令列參數
case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
    -b|--backend)
        info "只啟動後端服務模式"
        check_directory
        check_conda_env
        setup_environment
        start_backend
        wait
        ;;
    -c|--cli)
        info "只啟動 CLI 介面模式"
        check_directory
        check_conda_env
        setup_environment
        start_cli
        wait
        ;;
    "")
        main
        ;;
    *)
        echo "未知選項: $1"
        show_help
        exit 1
        ;;
esac