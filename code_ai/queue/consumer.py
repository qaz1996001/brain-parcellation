"""
PgQueuer 消費者啟動器
替代 funboost 的 BoostersManager 和 CLI 功能
"""
import asyncio
import logging
import os
import signal
from typing import Optional

from .manager import get_queue_manager

logger = logging.getLogger(__name__)


class QueueConsumerManager:
    """佇列消費者管理器"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL", 
            "postgresql://postgres_n:postgres_p@127.0.0.1:15433/dicom"
        )
        self.queue_manager = get_queue_manager(self.connection_string)
        self.running = False
    
    async def start_all_consumers(self):
        """
        啟動所有佇列消費者
        替代 funboost 的 BoostersManager.consume_all()
        """
        logger.info("正在啟動 PgQueuer 消費者...")
        
        try:
            # 初始化佇列管理器
            pgq = await self.queue_manager.initialize()
            
            # 自動發現並註冊所有處理器
            await self._register_all_processors()
            
            # 設置信號處理器
            self._setup_signal_handlers()
            
            # 顯示已註冊的處理器
            processors = self.queue_manager.get_registered_processors()
            logger.info(f"已註冊 {len(processors)} 個佇列處理器:")
            for queue_name, config in processors.items():
                logger.info(f"  - {queue_name} (QPS: {config['qps']}, 並發: {config['concurrent_num']})")
            
            # 啟動消費者
            self.running = True
            logger.info("PgQueuer 消費者已啟動，等待任務...")
            await pgq.run()
            
        except Exception as e:
            logger.error(f"啟動消費者失敗: {str(e)}")
            raise
        finally:
            await self.queue_manager.close()
    
    async def _register_all_processors(self):
        """自動發現並註冊所有處理器"""
        # 匯入所有包含 @queue_manager.register_processor 的模組
        try:
            # 匯入任務模組 - 這會觸發裝飾器註冊
            from code_ai.task import task_pipeline_new, task_dicom2nii_new
            from code_ai.scheduler import scheduler_database_new
            
            logger.info("已匯入所有任務模組")
            
        except ImportError as e:
            logger.warning(f"匯入任務模組時發生警告: {str(e)}")
            # 如果新模組還未建立，嘗試匯入原始模組並動態註冊
            await self._register_legacy_processors()
    
    async def _register_legacy_processors(self):
        """註冊舊版 funboost 處理器 (過渡期使用)"""
        logger.info("正在註冊舊版處理器...")
        
        # 這裡可以動態載入並轉換舊的 @Booster 裝飾器
        # 實際實作時需要根據具體情況調整
        pass
    
    def _setup_signal_handlers(self):
        """設置信號處理器以便優雅關閉"""
        def signal_handler(signum, frame):
            logger.info(f"收到信號 {signum}，正在關閉消費者...")
            self.running = False
            # 觸發 asyncio 事件循環停止
            asyncio.create_task(self._graceful_shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _graceful_shutdown(self):
        """優雅關閉"""
        logger.info("正在執行優雅關閉...")
        await self.queue_manager.close()
        logger.info("消費者已關閉")


async def start_all_consumers():
    """
    啟動所有消費者的便捷函數
    可以直接在命令行中使用: pgq run code_ai.queue.consumer:start_all_consumers
    """
    consumer_manager = QueueConsumerManager()
    await consumer_manager.start_all_consumers()


def main():
    """命令行入口點"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(start_all_consumers())
    except KeyboardInterrupt:
        logger.info("收到中斷信號，正在關閉...")
    except Exception as e:
        logger.error(f"消費者運行失敗: {str(e)}")
        raise


if __name__ == "__main__":
    main()
