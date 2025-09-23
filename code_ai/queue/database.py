"""
PgQueuer 資料庫追蹤模組
替代 funboost 的資料庫記錄功能
"""
import json
from datetime import datetime
from typing import Any, Dict, Optional

import asyncpg


class JobTracker:
    """
    任務追蹤器 - 替代 funboost 的 FunboostConsumeResult
    """
    
    def __init__(self):
        self.conn: Optional[asyncpg.Connection] = None
    
    async def initialize(self, conn: asyncpg.Connection):
        """初始化資料庫連接和表格"""
        self.conn = conn
        await self._create_tables()
    
    async def _create_tables(self):
        """建立任務追蹤表格"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS pgqueue_job_results (
            id SERIAL PRIMARY KEY,
            job_id VARCHAR(255) NOT NULL,
            queue_name VARCHAR(255) NOT NULL,
            params JSONB,
            result JSONB,
            success BOOLEAN NOT NULL,
            error_message TEXT,
            start_time TIMESTAMP WITH TIME ZONE,
            end_time TIMESTAMP WITH TIME ZONE,
            execution_time FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- 索引優化
            UNIQUE(job_id)
        );
        
        -- 建立索引
        CREATE INDEX IF NOT EXISTS idx_pgqueue_job_results_queue_name 
            ON pgqueue_job_results(queue_name);
        CREATE INDEX IF NOT EXISTS idx_pgqueue_job_results_success 
            ON pgqueue_job_results(success);
        CREATE INDEX IF NOT EXISTS idx_pgqueue_job_results_created_at 
            ON pgqueue_job_results(created_at);
        """
        
        await self.conn.execute(create_table_sql)
    
    async def save_job_result(
        self,
        job_id: str,
        queue_name: str,
        params: Dict[str, Any],
        result: Any,
        success: bool,
        error_message: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        execution_time: Optional[float] = None
    ):
        """
        儲存任務執行結果
        替代 funboost 的 save_result_status_to_sqlalchemy
        """
        insert_sql = """
        INSERT INTO pgqueue_job_results (
            job_id, queue_name, params, result, success,
            error_message, start_time, end_time, execution_time
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (job_id) DO UPDATE SET
            result = EXCLUDED.result,
            success = EXCLUDED.success,
            error_message = EXCLUDED.error_message,
            end_time = EXCLUDED.end_time,
            execution_time = EXCLUDED.execution_time
        """
        
        # 序列化複雜物件
        params_json = json.dumps(params, ensure_ascii=False, default=str)
        result_json = json.dumps(result, ensure_ascii=False, default=str) if result is not None else None
        
        await self.conn.execute(
            insert_sql,
            job_id, queue_name, params_json, result_json, success,
            error_message, start_time, end_time, execution_time
        )
    
    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """取得任務結果"""
        query_sql = """
        SELECT * FROM pgqueue_job_results WHERE job_id = $1
        """
        
        row = await self.conn.fetchrow(query_sql, job_id)
        if row:
            return dict(row)
        return None
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """取得佇列統計資訊"""
        stats_sql = """
        SELECT 
            COUNT(*) as total_jobs,
            COUNT(*) FILTER (WHERE success = true) as successful_jobs,
            COUNT(*) FILTER (WHERE success = false) as failed_jobs,
            AVG(execution_time) FILTER (WHERE success = true) as avg_execution_time,
            MAX(created_at) as last_job_time
        FROM pgqueue_job_results 
        WHERE queue_name = $1
        """
        
        row = await self.conn.fetchrow(stats_sql, queue_name)
        return dict(row) if row else {}
    
    async def get_recent_jobs(
        self, 
        queue_name: Optional[str] = None, 
        limit: int = 100,
        success_only: Optional[bool] = None
    ) -> list[Dict[str, Any]]:
        """取得最近的任務記錄"""
        conditions = []
        params = []
        param_count = 0
        
        if queue_name:
            param_count += 1
            conditions.append(f"queue_name = ${param_count}")
            params.append(queue_name)
        
        if success_only is not None:
            param_count += 1
            conditions.append(f"success = ${param_count}")
            params.append(success_only)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query_sql = f"""
        SELECT * FROM pgqueue_job_results 
        {where_clause}
        ORDER BY created_at DESC 
        LIMIT {limit}
        """
        
        rows = await self.conn.fetch(query_sql, *params)
        return [dict(row) for row in rows]
    
    async def cleanup_old_records(self, days: int = 30):
        """清理舊記錄"""
        cleanup_sql = """
        DELETE FROM pgqueue_job_results 
        WHERE created_at < NOW() - INTERVAL '%s days'
        """
        
        result = await self.conn.execute(cleanup_sql, days)
        return result
    
    async def get_failure_analysis(self, queue_name: Optional[str] = None) -> Dict[str, Any]:
        """分析失敗任務"""
        conditions = "WHERE success = false"
        params = []
        
        if queue_name:
            conditions += " AND queue_name = $1"
            params.append(queue_name)
        
        analysis_sql = f"""
        SELECT 
            error_message,
            COUNT(*) as error_count,
            MAX(created_at) as last_occurrence
        FROM pgqueue_job_results 
        {conditions}
        GROUP BY error_message 
        ORDER BY error_count DESC
        LIMIT 20
        """
        
        rows = await self.conn.fetch(analysis_sql, *params)
        return [dict(row) for row in rows]
