"""
DICOM 同步模組設定管理。

此模組提供 DICOM 同步服務的配置管理，包括快取前綴、API 端點等設定。
使用 Pydantic BaseSettings 從環境變數載入配置。

Attributes
----------
SyncSettings
    同步服務設定類，包含所有配置選項。
    
get_sync_settings()
    取得同步服務設定實例的函數。

Notes
-----
配置優先順序：
    1. 環境變數
    2. 預設值
    
所有配置都可以通過環境變數覆蓋。

Examples
--------
基本使用：

>>> from backend.app.sync.settings import get_sync_settings
>>> settings = get_sync_settings()
>>> cache_prefix = settings.cache.inference_prefix
>>> print(cache_prefix)
inference_task

使用環境變數覆蓋：

>>> # 設置環境變數
>>> # export SYNC_CACHE_INFERENCE_PREFIX=custom_prefix
>>> settings = get_sync_settings()
>>> print(settings.cache.inference_prefix)
custom_prefix
"""

import os
from pydantic import BaseModel, Field
from functools import lru_cache


class CacheConfig(BaseModel):
    """
    快取配置類。
    
    Attributes
    ----------
    inference_prefix : str
        推論任務快取鍵前綴，預設 "inference_task"。
        用於構建 Redis 鍵：{inference_prefix}:{study_uid},{study_id}
    """
    inference_prefix: str = Field(
        default="inference_task",
        description="推論任務快取鍵前綴"
    )


class SyncSettings(BaseModel):
    """
    DICOM 同步服務設定類。
    
    此類包含所有 DICOM 同步服務的配置選項。
    配置可以通過環境變數或直接傳入參數設置。
    
    Attributes
    ----------
    cache : CacheConfig
        快取相關配置。
    
    Environment Variables
    ---------------------
    SYNC_CACHE_INFERENCE_PREFIX : str, optional
        推論任務快取鍵前綴，預設 "inference_task"。
    
    Examples
    --------
    使用預設值：
    
    >>> settings = SyncSettings()
    >>> print(settings.cache.inference_prefix)
    inference_task
    
    使用環境變數：
    
    >>> import os
    >>> os.environ["SYNC_CACHE_INFERENCE_PREFIX"] = "custom_prefix"
    >>> settings = SyncSettings()
    >>> print(settings.cache.inference_prefix)
    custom_prefix
    """
    
    cache: CacheConfig = Field(
        default_factory=lambda: CacheConfig(
            inference_prefix=os.getenv(
                "SYNC_CACHE_INFERENCE_PREFIX",
                "inference_task"
            )
        ),
        description="快取配置"
    )


@lru_cache(maxsize=1)
def get_sync_settings() -> SyncSettings:
    """
    取得 DICOM 同步服務設定實例。
    
    此函數使用 LRU 快取確保只創建一個設定實例，
    提高性能並確保配置一致性。
    
    Returns
    -------
    SyncSettings
        同步服務設定實例。
    
    Notes
    -----
    使用 @lru_cache 裝飾器確保：
    - 只創建一個設定實例（單例模式）
    - 提高性能（避免重複創建）
    - 配置一致性（所有呼叫回傳同一實例）
    
    注意：環境變數變更不會自動反映，需要重啟應用程式。
    
    Examples
    --------
    >>> settings = get_sync_settings()
    >>> cache_prefix = settings.cache.inference_prefix
    >>> print(cache_prefix)
    inference_task
    
    >>> # 多次呼叫回傳同一實例
    >>> settings1 = get_sync_settings()
    >>> settings2 = get_sync_settings()
    >>> assert settings1 is settings2  # True
    """
    return SyncSettings()
