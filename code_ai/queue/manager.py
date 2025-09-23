"""
PgQueuer 佇列管理器 - 替代 funboost 的核心功能
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Callable, Optional
from datetime import datetime

import asyncpg
from pgqueuer import PgQueuer
from pgqueuer.db import AsyncpgDriver
from pgqueuer.models import Job

from .database import JobTracker

logger = logging.getLogger(__name__)


class DicomQueueManager:
    """
    醫學影像處理佇列管理器
    替代 funboost 的 @Booster 裝飾器功能
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pgq: Optional[PgQueuer] = None
        self.conn: Optional[asyncpg.Connection] = None
        self.job_tracker = JobTracker()
        self._processors: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> PgQueuer:
        """初始化佇列連接"""
        if self.pgq is not None:
            return self.pgq
            
        self.conn = await asyncpg.connect(self.connection_string)
        driver = AsyncpgDriver(self.conn)
        self.pgq = PgQueuer(driver)
        
        # 初始化任務追蹤器
        await self.job_tracker.initialize(self.conn)
        
        logger.info("PgQueuer 佇列管理器初始化完成")
        return self.pgq
    
    async def close(self):
        """關閉連接"""
        if self.conn:
            await self.conn.close()
            logger.info("PgQueuer 連接已關閉")
    
    def register_processor(
        self, 
        queue_name: str, 
        qps: int = 1,
        max_retry_times: int = 3,
        retry_interval: int = 20,
        concurrent_num: int = 10,
        user_custom_record_func: Optional[Callable] = None
    ):
        """
        註冊處理器裝飾器 - 替代 @Booster 裝飾器
        
        Args:
            queue_name: 佇列名稱
            qps: 每秒處理任務數
            max_retry_times: 最大重試次數  
            retry_interval: 重試間隔(秒)
            concurrent_num: 並發數量
            user_custom_record_func: 自定義記錄函數
        """
        def decorator(func: Callable):
            # 儲存處理器配置
            self._processors[queue_name] = {
                'func': func,
                'qps': qps,
                'max_retry_times': max_retry_times,
                'retry_interval': retry_interval,
                'concurrent_num': concurrent_num,
                'user_custom_record_func': user_custom_record_func
            }
            
            @self.pgq.entrypoint(queue_name)
            async def wrapper(job: Job) -> None:
                """任務包裝器 - 處理錯誤和記錄"""
                job_start_time = datetime.utcnow()
                
                try:
                    # 解析任務參數
                    func_params = job.payload
                    
                    logger.info(f"開始處理任務: {queue_name}, Job ID: {job.id}")
                    
                    # 執行原始函數
                    if asyncio.iscoroutinefunction(func):
                        result = await func(func_params)
                    else:
                        result = func(func_params)
                    
                    # 記錄成功狀態
                    await self._save_job_result(
                        job, func_params, result, True, 
                        job_start_time, user_custom_record_func
                    )
                    
                    logger.info(f"任務完成: {queue_name}, Job ID: {job.id}")
                    return result
                    
                except Exception as e:
                    # 記錄失敗狀態
                    await self._save_job_result(
                        job, func_params, None, False,
                        job_start_time, user_custom_record_func, str(e)
                    )
                    
                    logger.error(f"任務失敗: {queue_name}, Job ID: {job.id}, 錯誤: {str(e)}")
                    raise
            
            return wrapper
        return decorator
    
    async def _save_job_result(
        self, 
        job: Job, 
        func_params: Dict[str, Any],
        result: Any, 
        success: bool,
        start_time: datetime,
        user_custom_record_func: Optional[Callable] = None,
        error_message: str = None
    ):
        """
        儲存任務執行結果到 PostgreSQL
        替代 funboost 的 save_result_status_to_sqlalchemy
        """
        try:
            # 計算執行時間
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            # 儲存到資料庫
            await self.job_tracker.save_job_result(
                job_id=str(job.id),
                queue_name=job.entrypoint,
                params=func_params,
                result=result,
                success=success,
                error_message=error_message,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time
            )
            
            # 如果有自定義記錄函數，也執行它
            if user_custom_record_func:
                # 建立類似 FunctionResultStatus 的狀態物件
                status_dict = {
                    'job_id': str(job.id),
                    'queue_name': job.entrypoint,
                    'params': func_params,
                    'result': result,
                    'success': success,
                    'error_message': error_message,
                    'start_time': start_time,
                    'end_time': end_time,
                    'execution_time': execution_time
                }
                
                if asyncio.iscoroutinefunction(user_custom_record_func):
                    await user_custom_record_func(status_dict)
                else:
                    user_custom_record_func(status_dict)
                    
        except Exception as e:
            logger.error(f"儲存任務結果失敗: {str(e)}")
    
    async def enqueue_job(
        self, 
        queue_name: str, 
        params: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """
        將任務加入佇列
        替代 funboost 的 .push() 方法
        """
        if not self.pgq:
            await self.initialize()
            
        # 使用 pgqueuer 的 enqueue 方法
        queries = self.pgq.queries
        job_ids = await queries.enqueue(
            [queue_name],  # entrypoints
            [json.dumps(params).encode()],  # payloads  
            [priority]  # priorities
        )
        
        job_id = job_ids[0] if job_ids else None
        logger.info(f"任務已加入佇列: {queue_name}, Job ID: {job_id}")
        return str(job_id)
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """取得佇列統計資訊"""
        return await self.job_tracker.get_queue_stats(queue_name)
    
    def get_registered_processors(self) -> Dict[str, Dict[str, Any]]:
        """取得已註冊的處理器資訊"""
        return self._processors.copy()


# 全域佇列管理器實例
_global_queue_manager: Optional[DicomQueueManager] = None


def get_queue_manager(connection_string: str = None) -> DicomQueueManager:
    """取得全域佇列管理器實例"""
    global _global_queue_manager
    
    if _global_queue_manager is None:
        if connection_string is None:
            # 從環境變數或配置獲取
            connection_string = "postgresql://postgres_n:postgres_p@127.0.0.1:15433/dicom"
        _global_queue_manager = DicomQueueManager(connection_string)
    
    return _global_queue_manager


@asynccontextmanager
async def queue_manager_context(connection_string: str = None):
    """佇列管理器上下文管理器"""
    manager = get_queue_manager(connection_string)
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.close()
