#!/bin/bash

# 新的啟動腳本 - PgQueuer 版本
# 替代原始的 funboost 啟動方式

# 設定 PYTHONPATH 為當前目錄
export PYTHONPATH=$(pwd)/code_ai

# 載入環境變數
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 顯示啟動資訊
echo "======================================"
echo "🚀 啟動醫學影像處理系統 (PgQueuer 版本)"
echo "======================================"
echo "Python Path: $PYTHONPATH"
echo "Database: $DATABASE_URL"
echo "======================================"

# 檢查 PostgreSQL 連接
echo "📡 檢查 PostgreSQL 連接..."
python -c "
import asyncio
import asyncpg
async def test_db():
    try:
        conn = await asyncpg.connect('postgresql://postgres_n:postgres_p@127.0.0.1:15433/dicom')
        print('✅ PostgreSQL 連接成功')
        await conn.close()
    except Exception as e:
        print(f'❌ PostgreSQL 連接失敗: {e}')
        exit(1)
asyncio.run(test_db())
"

# 初始化 PgQueuer 佇列表格 (如果尚未初始化)
echo "🔧 初始化 PgQueuer 佇列表格..."
pgq install --connection-string="postgresql://postgres_n:postgres_p@127.0.0.1:15433/dicom" || echo "⚠️  佇列表格可能已存在"

# 啟動 PgQueuer 消費者
echo "🎯 啟動 PgQueuer 佇列消費者..."
echo "替代原始命令: funboost start -m funboost consume_all_queues"
echo "新命令: python -m code_ai.queue.consumer"
echo "======================================"

# 方法一：直接使用 Python 模組啟動
python -m code_ai.queue.consumer

# 方法二：使用 pgqueuer CLI 啟動 (備選)
# pgq run code_ai.queue.consumer:start_all_consumers --connection-string="postgresql://postgres_n:postgres_p@127.0.0.1:15433/dicom"

echo "🏁 PgQueuer 消費者已停止"
