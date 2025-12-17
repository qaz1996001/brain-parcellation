"""
DICOM 同步服務層 - 業務邏輯核心實現。

此模組包含 DCOPEventDicomService，負責 DICOM 同步系統的所有業務邏輯，包括：
1. Study/Series 狀態轉遷
2. 事件驅動的工作流程
3. 與外部工具的集成（Orthanc、NIFTI_TOOL、推論引擎）
4. 多輸出序列的特殊處理（如 DWI）
5. 日誌記錄和審計追蹤

核心概念：
- 事件驅動: 每個狀態變化都產生一條事件紀錄
- 檢查點 API: 推動狀態轉遷而非完全自動化
- 鬆散耦合: 通過配置表和事件而非直接 API 調用
- 完整審計: 完整的時間戳記和參數追蹤

Dependencies
-----------
code_ai : 代碼轉換和推理任務框架
pyorthanc : Orthanc DICOM 伺服器 API 客戶端
sqlalchemy : 非同步 ORM
fastapi_cache : Redis 快取層
funboost : 非同步任務隊列

Notes
-----
此服務層建立在 Good Taste 設計原則之上：
- 消除特殊情況：統一的事件處理流程
- 資料結構驅動：使用 DCOPStatus 列舉而非魔術字符串
- 鬆散耦合：配置驅動而非硬編碼邏輯
"""

import json
import logging
import os
import pathlib
import traceback
from typing import List, Optional, Tuple, Dict, Any
import re
import httpx
import pandas as pd
import pydicom
from datetime import datetime
from logging.handlers import RotatingFileHandler
from advanced_alchemy.extensions.fastapi import repository
from advanced_alchemy.service import OffsetPagination
from funboost import AsyncResult
from pyorthanc import Study, Orthanc
from sqlalchemy import text, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi_cache import FastAPICache

from code_ai.task.schema.intput_params import Dicom2NiiParams
from backend.app.service import BaseRepositoryService
from .model import DCOPEventModel
from .schemas import DCOPStatus, DCOPEventRequest, DCOPEventNIFTITOOLRequest, StydySeriesOpeNoStatus,OpeNo,OrthancID
from .urls import SYNC_PROT_OPE_NO, SYNC_PROT_STUDY_NIFTI_TOOL, SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID, \
    SYNC_PROT_STUDY_TRANSFER_COMPLETE

def setup_dcop_event_logger() -> logging.Logger:
    """
    為 DCOPEventDicomService 設置日誌記錄器。
    
    此函數配置一個獨立的日誌系統，用於記錄 DICOM 同步系統的所有活動。
    具有自動日誌輪轉和過期清理功能。
    
    特性
    ----
    - 檔案大小限制: 100MB
    - 備份保留: 5 個檔案
    - 日期保留: 7 天
    - 同時輸出到檔案和控制台
    - 不向 root logger 傳播（隔離）
    
    Returns
    -------
    logging.Logger
        配置完成的 logger 實例。
    
    Notes
    -----
    此函數檢查是否已有 handlers，避免重複初始化。
    日誌檔案位置由環境變數 LOG_PATH 決定，預設為 ./logs
    
    Examples
    --------
    >>> logger = setup_dcop_event_logger()
    >>> logger.info("Study 123 started processing")
    
    日誌檔案位置 (LOG_PATH/YYYY-MM-DD.0001.DCOPEventDicomService.log):
    2025-12-17 14:30:45 - DCOPEventDicomService - INFO - [service.py:100] - Study 123 started
    """
    # 獲取或創建 logger 實例
    dcop_logger = logging.getLogger('DCOPEventDicomService')
    
    # 避免重複添加 handler（多次調用時的冪等性）
    if dcop_logger.handlers:
        return dcop_logger
    
    # 設置日誌級別為 DEBUG（記錄最詳細的資訊）
    dcop_logger.setLevel(logging.DEBUG)
    
    # 設置日誌目錄
    log_dir = os.getenv('LOG_PATH', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 清理 7 天前的舊檔案（自動維護）
    _cleanup_old_logs(log_dir, days=7)
    
    # 生成日誌檔案名稱（按日期和序號）
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_filename = f'{date_str}.0001.DCOPEventDicomService.log'
    log_filepath = os.path.join(log_dir, log_filename)
    
    # 創建文件 handler (最大 100MB，輪轉時保留 5 個備份)
    file_handler = RotatingFileHandler(
        log_filepath,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 設置日誌格式（包含時間、模組、行號等）
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 添加文件 handler
    dcop_logger.addHandler(file_handler)
    
    # 同時輸出到控制台（用於實時監控）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    dcop_logger.addHandler(console_handler)
    
    # 避免日誌向上傳播到 root logger（隔離日誌系統）
    dcop_logger.propagate = False
    
    return dcop_logger


def _cleanup_old_logs(log_dir: str, days: int = 7) -> None:
    """
    清理指定日期之前的日誌檔案。
    
    此函數自動清理過期的日誌文件，防止磁盤空間耗盡。
    使用檔案修改時間判斷是否過期。
    
    Parameters
    ----------
    log_dir : str
        日誌目錄路徑。
    days : int, optional
        保留天數，預設 7 天。
        在此之前修改的檔案會被刪除。
    
    Returns
    -------
    None
    
    Notes
    -----
    此函數具有容錯能力：
    - 刪除失敗時會被捕捉並忽略
    - 遍歷過程中的異常也會被忽略
    - 確保不影響主業務流程
    
    檔案匹配規則：
    - 主檔案: YYYY-MM-DD.0001.DCOPEventDicomService.log
    - 備份檔案: .log.1, .log.2, ..., .log.5
    
    Examples
    --------
    >>> _cleanup_old_logs('./logs', days=7)
    # 刪除所有 7 天前修改的 DCOPEventDicomService 日誌檔案
    """
    from datetime import timedelta
    
    # 計算截止時間戳記
    cutoff_date = datetime.now() - timedelta(days=days)
    cutoff_timestamp = cutoff_date.timestamp()
    
    try:
        # 遍歷日誌目錄中的所有檔案
        for filename in os.listdir(log_dir):
            # 檢查是否為 DCOPEventDicomService 日誌檔案（包括備份）
            if filename.endswith('.DCOPEventDicomService.log') or \
               filename.endswith('.DCOPEventDicomService.log.1') or \
               filename.endswith('.DCOPEventDicomService.log.2') or \
               filename.endswith('.DCOPEventDicomService.log.3') or \
               filename.endswith('.DCOPEventDicomService.log.4') or \
               filename.endswith('.DCOPEventDicomService.log.5'):
                filepath = os.path.join(log_dir, filename)
                
                # 獲取檔案修改時間
                file_mtime = os.path.getmtime(filepath)
                
                # 如果檔案早於截止時間，刪除它
                if file_mtime < cutoff_timestamp:
                    try:
                        os.remove(filepath)
                        # 可選：記錄刪除行為（但避免循環依賴）
                    except Exception as e:
                        # 刪除失敗時忽略（例如檔案被鎖定）
                        pass
    except Exception as e:
        # 清理過程中的任何異常都被忽略（避免影響主流程）
        pass


# 初始化 DCOPEventDicomService logger
_dcop_event_logger = setup_dcop_event_logger()


class DCOPEventDicomService(BaseRepositoryService[DCOPEventModel]):
    """
    DICOM 同步事件服務 - 核心業務邏輯層。
    
    此服務類負責管理 DICOM 同步系統的所有業務邏輯，包括：
    1. Study/Series 狀態轉遷管理
    2. 事件記錄和審計
    3. 與 Orthanc DICOM 伺服器的集成
    4. NIFTI 轉檔工具的協調
    5. 推論任務的排隊和監控
    6. 多輸出序列（如 DWI）的特殊處理
    
    狀態轉遷流程
    -----------
    完整的生命週期遵循嚴格的狀態機：
    
    1. 傳輸階段:
       STUDY_NEW → STUDY_TRANSFERRING → (Series 傳輸) → STUDY_TRANSFER_COMPLETE
    
    2. 轉檔階段:
       STUDY_CONVERTING → SERIES_CONVERTING → SERIES_CONVERSION_COMPLETE → STUDY_CONVERSION_COMPLETE
    
    3. 推論階段:
       STUDY_INFERENCE_READY → QUEUED → RUNNING → COMPLETE
    
    多輸出序列處理
    ---------------
    對於 DWI (Diffusion Weighted Imaging) 等多輸出序列，系統：
    - 自動檢測序列類型
    - 掃描文件系統找尋所有輸出（如 DWI0、DWI1000）
    - 為每個輸出創建獨立的轉檔任務
    - 等待所有輸出完成後才進入下一階段
    
    Attributes
    ----------
    logger : logging.Logger
        類級別的日誌記錄器，所有實例共享。
    pattern_str : str
        用於匹配 Series 轉檔完成序列的正則表達式。
    can_inference_pattern : re.Pattern
        編譯後的正則表達式，用於快速判斷 Series 是否可進入推論。
    
    Nested Classes
    ---------------
    Repo : SQLAlchemyAsyncRepository
        非同步 ORM 儲存庫，處理資料庫操作。
    
    Notes
    -----
    Good Taste 設計特點：
    - 消除特殊情況：統一的事件驅動流程
    - 資料結構驅動：使用 DCOPStatus 列舉而非字符串
    - 檢查點 API：支援推動式轉遷而非完全自動化
    - 鬆散耦合：通過事件而非直接 API 調用
    
    Examples
    --------
    基本使用流程：
    
    >>> service = DCOPEventDicomService()
    >>> 
    >>> # 1. 排程新 Study 同步
    >>> study_ids = ["study-uid-123"]
    >>> await service.schedule_new_studies(study_ids)
    >>> 
    >>> # 2. 擷取 Series 資訊
    >>> events = await service.dicom_tool_get_series_info(result_list)
    >>> 
    >>> # 3. 檢查傳輸是否完成
    >>> await service.check_study_series_transfer_complete()
    >>> 
    >>> # 4. 排程轉檔任務
    >>> await service.nifti_tool_get_series_info(study_uid, session)
    """

    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        """
        DICOM 事件儲存庫 - 非同步 ORM 操作。
        
        使用 SQLAlchemy 異步引擎進行資料庫操作，提供事件的
        CRUD 操作和複雜查詢。
        """
        model_type = DCOPEventModel

    repository_type = Repo
    
    # 正則表達式：Series 必須經歷的狀態序列
    # 格式: (NEW), (TRANSFERRING), (TRANSFER_COMPLETE), (CONVERTING), (CONVERSION_COMPLETE | CONVERSION_SKIP)
    pattern_str = '({}),({}),({}),({}),({}|{})'.format(
        DCOPStatus.SERIES_NEW.value,
        DCOPStatus.SERIES_TRANSFERRING.value,
        DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
        DCOPStatus.SERIES_CONVERTING.value,
        DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
        DCOPStatus.SERIES_CONVERSION_SKIP.value
    )
    # 編譯後的正則模式，用於快速檢查 Series 是否可進入推論
    can_inference_pattern = re.compile(pattern_str)
    
    # 類級別的日誌記錄器，所有實例共享
    logger = _dcop_event_logger

    async def get_check_url_by_ope_no(self, ope_no: str) -> Optional[str]:
        """
        根據操作編號取得對應的檢查點 API URL。
        
        此方法實現了操作碼到 API 端點的映射，支援四個檢查點：
        1. Series 傳輸完成 → 觸發 Study 傳輸檢查
        2. Series 轉檔完成 → 觸發 Study 轉檔檢查
        3. Study 傳輸完成 → 觸發轉檔初始化
        4. Study 轉檔完成 → 觸發推論初始化
        
        Parameters
        ----------
        ope_no : str
            操作編號，格式為 xxx.xxx（e.g. "100.095"）。
        
        Returns
        -------
        Optional[str]
            對應的檢查點 API URL，若無對應則返回 None。
        
        Examples
        --------
        >>> url = await service.get_check_url_by_ope_no("100.095")
        >>> print(url)
        http://api.server/sync/study/transfer
        
        >>> url = await service.get_check_url_by_ope_no("200.195")
        >>> print(url)
        http://api.server/sync/study/convert
        
        Notes
        -----
        Good Taste 設計：使用 Python match-case 語句而非 if-elif 鏈，
        消除了條件邏輯中的重複。
        """
        from code_ai import load_dotenv
        load_dotenv()
        UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
        
        # 根據操作編號映射到對應的檢查點 API
        match ope_no:
            case DCOPStatus.STUDY_TRANSFER_COMPLETE.value:
                # Study 傳輸完成 → 觸發轉檔初始化
                url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_TRANSFER_COMPLETE}"
            case DCOPStatus.STUDY_CONVERSION_COMPLETE.value:
                # Study 轉檔完成 → 觸發推論初始化
                url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
            case DCOPStatus.SERIES_TRANSFER_COMPLETE.value:
                # Series 傳輸完成 → 觸發 Study 傳輸檢查
                url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_TRANSFER_COMPLETE}"
            case DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
                # Series 轉檔完成 → 觸發 Study 轉檔檢查
                url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
            case _:
                # 未知的操作編號
                url = None
        
        self.logger.debug(f'get_check_url_by_ope_no: ope_no={ope_no}, url={url}')
        return url

    async def post_ope_no_task(self, data: List[DCOPEventRequest]) -> None:
        """
        批次寫入事件記錄並觸發相應的檢查點 API。
        
        此方法用於處理外部系統（如 Orthanc、NIFTI_TOOL）批次上報的事件。
        它會：
        1. 逐一寫入事件到資料庫
        2. 識別觸發檢查點的事件類型
        3. 收集所有需要觸發的檢查點 URL
        4. 批次執行所有檢查點
        
        Parameters
        ----------
        data : list[DCOPEventRequest]
            外部系統上報的事件列表。
        
        Returns
        -------
        None
        
        Side Effects
        -----------
        - 每個事件寫入資料庫並立即提交
        - 觸發所有相關的檢查點 API（去重）
        
        事件與檢查點的對應
        -------------------
        - SERIES_TRANSFER_COMPLETE → 檢查 Study 傳輸是否完成
        - SERIES_CONVERSION_COMPLETE → 檢查 Study 轉檔是否完成
        
        Examples
        --------
        處理 Orthanc 的 Series 傳輸完成上報：
        
        >>> events = [
        ...     DCOPEventRequest(
        ...         study_uid="abc-123",
        ...         series_uid="def-456",
        ...         ope_no="100.095",
        ...         tool_id="DICOM_TOOL"
        ...     ),
        ...     DCOPEventRequest(
        ...         study_uid="abc-123",
        ...         series_uid="ghi-789",
        ...         ope_no="100.095",
        ...         tool_id="DICOM_TOOL"
        ...     )
        ... ]
        >>> await service.post_ope_no_task(events)
        # 結果: 2 個事件寫入，1 次檢查點 API 調用
        
        Notes
        -----
        檢查點 URL 去重：
        - 若同一檢查點被多個事件觸發，只調用一次
        - 例如 2 個 Series 都完成轉檔，只調用一次 check_conversion
        
        原子性：
        - 每個事件單獨提交，確保原子性
        - 某個事件寫入失敗不影響其他事件
        """
        from code_ai import load_dotenv
        load_dotenv()
        
        # 收集所有需要觸發的檢查點 URL（使用集合去重）
        check_url_set = set()
        
        async with self.session_manager.get_session() as session:
            # 逐一處理每個事件
            for dcop_event in data:
                # 建立事件記錄
                new_data_obj = await DCOPEventModel.create_event_ope_no(
                    tool_id=dcop_event.tool_id,
                    study_uid=dcop_event.study_uid,
                    series_uid=dcop_event.series_uid,
                    study_id=dcop_event.study_id,
                    ope_no=dcop_event.ope_no,
                    result_data=dcop_event.result_data,
                    params_data=dcop_event.params_data,
                    session=session
                )
                
                # 寫入資料庫
                session.add(new_data_obj)
                await session.commit()
                await session.refresh(new_data_obj)

                # 識別是否需要觸發檢查點
                match new_data_obj.ope_no:
                    case DCOPStatus.SERIES_TRANSFER_COMPLETE.value:
                        # Series 傳輸完成 → 檢查 Study 傳輸
                        url = await self.get_check_url_by_ope_no(new_data_obj.ope_no)
                    case DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
                        # Series 轉檔完成 → 檢查 Study 轉檔
                        url = await self.get_check_url_by_ope_no(new_data_obj.ope_no)
                    case _:
                        # 其他事件不觸發檢查點
                        url = None
                
                # 添加到檢查點集合（自動去重）
                if url is not None and url not in check_url_set:
                    check_url_set.add(url)
        
        # 批次執行所有檢查點
        async with httpx.AsyncClient(timeout=180) as client:
            for url in check_url_set:
                try:
                    response = await client.post(url)
                    self.logger.debug(f'Checkpoint API called: {url}, status: {response.status_code}')
                except Exception as e:
                    self.logger.error(f'Error calling checkpoint API {url}: {e}')

    async def check_study_series_transfer_complete(
        self,
        data: Optional[List[DCOPEventRequest]] = None
    ) -> Optional[List[DCOPEventRequest]]:
        """
        檢查 Study/Series 傳輸是否完成，若完成則進入轉檔階段。
        
        此方法是 "檢查點" API，用於推動狀態轉遷。其工作流程為：
        
        1. 獲取所有狀態 ≥ SERIES_TRANSFER_COMPLETE 的 Series
        2. 為每個已完成傳輸的 Study 建立 STUDY_TRANSFER_COMPLETE 事件
        3. 建立 STUDY_CONVERTING 事件，開始轉檔階段
        4. 排程 NIFTI 轉檔工具執行
        
        狀態轉遷圖
        ----------
        SERIES_TRANSFER_COMPLETE (多個)
                    ↓
        [此方法檢查]
                    ↓
        STUDY_TRANSFER_COMPLETE
                    ↓
        STUDY_CONVERTING
        
        Parameters
        ----------
        data : list[DCOPEventRequest], optional
            指定要檢查的事件列表。
            若為 None，則自動掃描資料庫中所有待檢查的 Study。
        
        Returns
        -------
        list[DCOPEventRequest]
            已檢查的 Study 事件清單。
        
        Side Effects
        -----------
        - 在資料庫中建立事件
        - 透過 HTTP 呼叫 CHECK API 進行狀態轉遷
        - 排程 NIFTI 轉檔任務
        
        Examples
        --------
        自動掃描所有待檢查的 Study：
        
        >>> await service.check_study_series_transfer_complete()
        
        檢查指定的 Study：
        
        >>> events = [DCOPEventRequest(study_uid="abc-123", ope_no="100.095")]
        >>> await service.check_study_series_transfer_complete(data=events)
        
        Notes
        -----
        此方法會立即進行以下操作：
        1. 查詢資料庫或使用提供的事件
        2. 建立完成事件
        3. 透過 HTTP 觸發檢查點 API
        4. 等待檢查完成
        
        設計特點：
        - 可推動式（由外部觸發）或自動式（定期掃描）
        - 支援部分 Study 檢查
        - 非同步執行，不阻塞調用方
        """
        from code_ai import load_dotenv
        load_dotenv()
        self.logger.info(f'check_study_series_transfer_complete data {data}')
        # Get configuration from environment
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
        path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")

        # Retrieve study status information if not provided
        if data is None:
            dcop_event_list, dcop_event_dump_list = await self._get_studies_ready_for_transfer()
            for dcop_event_dump in dcop_event_dump_list:
                dcop_event_dump['params_data']
        else:
            dcop_event_list = [DCOPEventRequest.model_validate(event, strict=False) for event in data]
            dcop_event_dump_list = [dcop_event.model_dump() for dcop_event in dcop_event_list]

        # Process eligible studies for conversion
        if dcop_event_list:
            await self._send_events(upload_data_api_url, dcop_event_dump_list)
            await self._initiate_conversion_process(upload_data_api_url, dcop_event_list, path_rename_dicom,
                                                    path_rename_nifti)

        return dcop_event_list

    async def schedule_new_studies(self, study_ids: List[Optional[str]]) -> List[DCOPEventModel]:
        """
        排程新 Study 的同步任務（去重並過濾）。
        
        此方法是 `add_study_new` 的包裝方法，提供以下額外功能：
        1. 自動去重：移除重複的 Study UID
        2. 過濾無效值：移除 None 和空字串
        3. 統一介面：提供更語義化的方法名稱
        
        Parameters
        ----------
        study_ids : list[Optional[str]]
            Study UID 清單（可能包含重複和 None）。
        
        Returns
        -------
        list[DCOPEventModel]
            建立的所有事件模型（STUDY_NEW 和 STUDY_TRANSFERRING）。
        
        Examples
        --------
        >>> study_ids = ["study-uid-123", "study-uid-123", None, "study-uid-456"]
        >>> events = await service.schedule_new_studies(study_ids)
        >>> # 結果：只處理 "study-uid-123" 和 "study-uid-456"（去重並過濾 None）
        
        Notes
        -----
        此方法遵循 Good Taste 設計原則：
        - 消除特殊情況：統一處理重複和無效值
        - 資料結構驅動：使用集合去重而非手動檢查
        """
        # 去重並過濾無效值
        unique_study_ids = list({uid for uid in study_ids if uid})
        
        # 如果沒有有效的 Study ID，返回空列表
        if not unique_study_ids:
            return []
        
        # 調用實際的建立方法
        return await self.add_study_new(unique_study_ids)

    async def add_study_new(self, data_list: List[str]) -> List[DCOPEventModel]:
        """
        建立新 Study 的初始事件。
        
        此方法為每個新 Study 建立兩個初始事件：
        1. STUDY_NEW: Study 剛到達系統
        2. STUDY_TRANSFERRING: Study 開始傳輸，附帶傳輸參數
        
        這是 Study 生命週期的第一步，會立即排程 DICOM 轉檔任務。
        
        Parameters
        ----------
        data_list : list[str]
            Study UID 清單。
        
        Returns
        -------
        list[DCOPEventModel]
            建立的所有事件模型（STUDY_NEW 和 STUDY_TRANSFERRING）。
        
        Raises
        ------
        Exception
            若資料庫操作失敗，將回滾事務並重新拋出異常。
        
        Examples
        --------
        >>> study_ids = ["study-uid-123", "study-uid-456"]
        >>> events = await service.add_study_new(study_ids)
        >>> print(len(events))
        4  # 每個 Study 有 2 個事件
        
        Notes
        -----
        流程：
        1. 為每個 Study 創建 STUDY_NEW 事件
        2. 準備轉檔參數（檔案路徑）
        3. 創建 STUDY_TRANSFERRING 事件並附加參數
        4. 提交事務
        
        檔案路徑配置：
        - PATH_RAW_DICOM: 原始 DICOM 檔案位置
        - PATH_RENAME_DICOM: 重命名後的 DICOM 位置
        - PATH_RENAME_NIFTI: NIFTI 輸出位置
        """
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv
        load_dotenv()
        
        # 載入檔案路徑配置
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))

        result_list = []
        async with self.session_manager.get_session() as session:
            try:
                for ids in data_list:
                    study_uid_raw_dicom_path = raw_dicom_path.joinpath(ids)
                    
                    # 步驟 1: 建立 STUDY_NEW 事件
                    new_data = await DCOPEventModel.create_event(
                        study_uid=ids,
                        series_uid=None,
                        status=DCOPStatus.STUDY_NEW.name,
                        session=session,
                    )
                    self.logger.info(f'Created study event: {DCOPEventRequest.model_validate(new_data).model_dump()}')
                    session.add(new_data)
                    
                    # 步驟 2: 準備轉檔參數
                    task_params = Dicom2NiiParams(
                        sub_dir=study_uid_raw_dicom_path,
                        output_dicom_path=rename_dicom_path,
                        output_nifti_path=rename_nifti_path,
                    )
                    
                    # 步驟 3: 建立 STUDY_TRANSFERRING 事件（包含參數）
                    data_transferring = await DCOPEventModel.create_event(
                        study_uid=ids,
                        series_uid=None,
                        status=DCOPStatus.STUDY_TRANSFERRING.name,
                        session=session,
                    )
                    data_transferring.params_data = task_params.get_str_dict()
                    session.add(data_transferring)
                    
                    # 步驟 4: 提交事務
                    await session.commit()

                    # 收集結果
                    result_list.append(new_data)
                    result_list.append(data_transferring)
                    
            except Exception as e:
                # 發生錯誤時回滾所有更改
                await session.rollback()
                self.logger.error(f"Error in add_study_new: {e}")
                raise

        return result_list

    async def _get_studies_ready_for_transfer(self) -> Tuple[List[DCOPEventRequest], List[dict]]:
        """
        Retrieves studies that are ready for transfer completion marking.

        Returns:
            Tuple containing list of DCOPEventRequest objects and their serialized versions.
        """
        # engine: AsyncEngine = self.repository.session.bind
        dcop_event_list = []
        dcop_event_dump_list = []

        # async with engine.connect() as conn:
        async with self.session_manager.get_session() as session:
            results = await session.execute(text('select * from public.get_all_studies_status()'))
            for result in results.all():
                self.logger.info(f'result {result}')
                study_data = result[0]
                dcop_event = DCOPEventRequest(
                    study_uid=study_data['study_uid'],
                    series_uid=None,
                    ope_no=DCOPStatus.STUDY_TRANSFER_COMPLETE.value,
                    study_id=study_data['study_id'],
                    tool_id='DICOM_TOOL',
                    result_data={'result': json.dumps(study_data['result'])}
                )
                dcop_event_dump_list.append(dcop_event.model_dump())
                dcop_event_list.append(dcop_event)

        return dcop_event_list, dcop_event_dump_list

    async def _send_events(self, api_url: str, event_data) -> None:
        """
        Sends study transfer complete events to the API.

        Args:
            api_url: Base URL for the upload data API.
            event_data: List of serialized DCOPEventRequest objects.
        """
        event_data_list = list(filter(lambda x: x is not None, event_data))
        self.logger.info(f'_send_events {event_data_list}')
        async with httpx.AsyncClient(timeout=180) as client:
            url = f"{api_url}{SYNC_PROT_OPE_NO}"
            # event_data_json = json.dumps(event_data)
            await client.post(url=url, json=event_data_list)

    async def _initiate_conversion_process(
            self,
            api_url: str,
            events: List[DCOPEventRequest],
            dicom_path: str,
            nifti_path: str
    ) -> None:
        """
        Initiates the conversion process for each study.

        Args:
            api_url: Base URL for the upload data API.
            events: List of DCOPEventRequest objects.
            dicom_path: Path for renamed DICOM files.
            nifti_path: Path for NIFTI output.
        """

        url = f"{api_url}{SYNC_PROT_STUDY_NIFTI_TOOL}"
        for event in events:
            study_id = event.study_id
            output_dicom_path = pathlib.Path(os.path.join(dicom_path, study_id))
            output_nifti_path = pathlib.Path(nifti_path)

            # Prepare conversion parameters
            task_params = Dicom2NiiParams(
                sub_dir=None,
                output_dicom_path=output_dicom_path,
                output_nifti_path=output_nifti_path
            )

            # Create and send the conversion request
            async with httpx.AsyncClient(timeout=180) as client:
                nifti_tool_request = DCOPEventNIFTITOOLRequest(
                    ope_no=DCOPStatus.STUDY_CONVERTING.value,
                    study_id=study_id,
                    tool_id='NIFTI_TOOL',
                    params_data=task_params.get_str_dict(),
                    result_data=None
                )

                request_data = json.dumps([nifti_tool_request.model_dump()])
                await client.post(url=url, data=request_data)

    async def study_series_nifti_tool(self, data: List[DCOPEventNIFTITOOLRequest]):
        """
        建立
           DCOPStatus.STUDY_CONVERTING
           DCOPStatus.SERIES_CONVERTING
           DCOPStatus.SERIES_CONVERSION_COMPLETE
           DCOPStatus.STUDY_CONVERSION_COMPLETE
        """

        from code_ai import load_dotenv
        load_dotenv()
        # session: AsyncSession = self.repository.session
        for dcop in data:
            match dcop.ope_no:
                case DCOPStatus.STUDY_CONVERTING.value:
                    async with self.session_manager.get_session() as session:
                        # DICOM_TOOL
                        conf_query = select(DCOPEventModel, ).where(and_(*[DCOPEventModel.study_id == dcop.study_id,
                                                                           DCOPEventModel.study_uid.isnot(None),
                                                                           DCOPEventModel.tool_id == 'DICOM_TOOL',
                                                                           DCOPEventModel.ope_no == DCOPStatus.SERIES_TRANSFER_COMPLETE.value]))
                        execute = await session.execute(conf_query)
                        dcop_event = execute.first()[0]
                        study_transfer_complete_data = await DCOPEventModel.create_event_ope_no(tool_id=dcop.tool_id,
                                                                                                study_uid=dcop_event.study_uid,
                                                                                                series_uid=None,
                                                                                                study_id=dcop.study_id,
                                                                                                ope_no=dcop.ope_no,
                                                                                                result_data=dcop.result_data,
                                                                                                params_data=dcop.params_data,
                                                                                                session=session)

                        session.add(study_transfer_complete_data)
                        await session.commit()
                        await session.refresh(study_transfer_complete_data)
                        await self.nifti_tool_get_series_info(dcop_event.study_uid, session)
                case DCOPStatus.SERIES_CONVERTING.value:
                    pass
                    # new_data_obj = await self.create(new_data, auto_commit=True)

        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        url = f"{upload_data_api_url}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
        async with httpx.AsyncClient(timeout=180) as client:
            await client.post(url=url)

    async def nifti_tool_get_series_info(self, study_uid: str, session: AsyncSession) -> None:
        """
        獲取需要轉換的 Series 資訊並建立 NIFTI 轉換任務。
        
        此方法是整個系統中最複雜的邏輯，處理多輸出序列的特殊情況。
        對於 DWI (Diffusion Weighted Imaging) 等多輸出序列，系統會：
        
        1. 自動檢測序列類型
        2. 掃描文件系統找尋所有輸出（如 DWI0、DWI1000）
        3. 為每個輸出創建獨立的 NIFTI 轉換任務
        4. 等待所有輸出完成後才進入下一階段
        
        多輸出序列配置
        ---------------
        當前支援的多輸出序列：
        - DWI: 需要 DWI0 和 DWI1000 兩個輸出
        - SWAN: 需要 MAG 和 PHASE 兩個輸出（可選）
        - ESWAN: 同 SWAN
        
        Parameters
        ----------
        study_uid : str
            Study UID。
        session : AsyncSession
            活動的非同步資料庫會話。
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            若資料庫寫入或任務隊列操作失敗，將回滾事務。
        
        Side Effects
        -----------
        - 在資料庫中創建多個 SERIES_CONVERTING 事件
        - 將轉檔任務推送到隊列
        - 記錄詳細的處理日誌
        
        Notes
        -----
        此方法使用多級日誌記錄，便於偵錯：
        - 資訊級別: 主要流程步驟
        - 調試級別: 詳細的路徑和參數
        - 警告級別: 潛在的問題（如缺失的輸出）
        
        特殊設計特點：
        - 文件系統掃描容錯：若 result_data 不完整，自動掃描磁盤
        - 重複檢測：避免重複的 rename_dicom_path
        - 任務去重：若快取中已有推論任務，跳過
        
        Examples
        --------
        DWI 序列處理流程：
        
        1. 輸入: STUDY_CONVERTING 狀態的 DWI 序列
        2. 檢測: 識別為多輸出序列 (DWI: [DWI0, DWI1000])
        3. 掃描: 在文件系統中尋找 DWI0 和 DWI1000 目錄
        4. 建立: 為 DWI0 創建任務 + 為 DWI1000 創建任務
        5. 入隊: 兩個任務都推送到執行隊列
        6. 結果: 資料庫中有 2 個 SERIES_CONVERTING 記錄
        
        >>> await service.nifti_tool_get_series_info(
        ...     study_uid="abc-123",
        ...     session=session
        ... )
        # 日誌輸出:
        # [NIFTI_TOOL] 開始處理 study: abc-123
        # [SERIES 1] 🔀 Multi-output series detected!
        # [SERIES 1] [OUTPUT 1/2] ✅ Found DWI0 on disk
        # [SERIES 1] [OUTPUT 2/2] ✅ Found DWI1000 on disk
        # [QUEUE] 所有任務已成功發送到佇列
        # [NIFTI_TOOL] 結束處理 study: abc-123
        """
        from code_ai.task.task_dicom2nii import dicom_2_nii_series
        from code_ai.task.schema.intput_params import Dicom2NiiSeriesParams
        from code_ai import load_dotenv
        
        # 使用類級別的日誌記錄器
        log = self.logger
        
        log.info(f"{'='*80}")
        log.info(f"[NIFTI_TOOL] 開始處理 study: {study_uid}")
        log.info(f"{'='*80}")
        
        load_dotenv()
        path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")
        
        log.info(f"[CONFIG] PATH_RENAME_NIFTI: {path_rename_nifti}")
        
        # 定義 multi-output series 的配置
        # key: series description pattern, value: required output names
        MULTI_OUTPUT_CONFIG = {
            'DWI': ['DWI0', 'DWI1000'],  # DWI 必須有 DWI0 和 DWI1000
            # 'SWAN': ['SWAN_MAG', 'SWAN_PHASE'],  # SWAN 必須有 MAG 和 PHASE
            # 'ESWAN': ['SWAN_MAG', 'SWAN_PHASE'],  # ESWAN 同 SWAN
        }
        
        log.info(f"[CONFIG] MULTI_OUTPUT_CONFIG: {MULTI_OUTPUT_CONFIG}")
        
        def is_multi_output_series(series_desc: str) -> Tuple[bool, Optional[List[str]]]:
            """判斷是否為 multi-output series，返回 (是否, required_outputs)"""
            if not series_desc:
                return False, None
            
            series_desc_upper = series_desc.upper()
            for key, required_outputs in MULTI_OUTPUT_CONFIG.items():
                if key in series_desc_upper:
                    log.debug(f"[DETECTION] Matched '{key}' in '{series_desc}' -> Multi-output: {required_outputs}")
                    return True, required_outputs
            return False, None
        
        # 查詢待轉換的 series
        log.info(f"[QUERY] 查詢狀態: {DCOPStatus.STUDY_CONVERTING.value} for study_uid: {study_uid}")
        sql = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid')
        results = await session.execute(sql,
                                        {'status': DCOPStatus.STUDY_CONVERTING.value,
                                         'study_uid': study_uid})

        dcop_event_list = results.all()
        log.info(f"[QUERY] 找到 {len(dcop_event_list)} 個待處理的 series")
        
        task_params_list = []
        dcop_model_list = []
        
        for idx, dcop_event in enumerate(dcop_event_list, 1):
            series_uid = dcop_event.series_uid
            log.info(f"\n[SERIES {idx}/{len(dcop_event_list)}] {'='*60}")
            log.info(f"[SERIES {idx}] series_uid: {series_uid}")
            log.info(f"[SERIES {idx}] study_id: {dcop_event.study_id}")
            
            # 提取 series description
            series_description = None
            candidate_description_from_path = None
            
            if dcop_event.result_data:
                log.info(f"[SERIES {idx}] result_data 包含 {len(dcop_event.result_data)} 個元素")
                for rd_idx, rd in enumerate(dcop_event.result_data):
                    log.debug(f"[SERIES {idx}] result_data[{rd_idx}]: {rd}")
                    if rd and isinstance(rd, dict):
                        # 優先尋找 'result' 欄位 (通常是 Series Description)
                        if 'result' in rd:
                            series_description = rd['result']
                            log.info(f"[SERIES {idx}] 找到 series_description: '{series_description}'")
                            break
                        
                        # 如果還沒找到 description，暫存第一個有效的 rename_dicom_path 的最後一層目錄名稱作為備案
                        if 'rename_dicom_path' in rd and not candidate_description_from_path:
                            path_str = rd['rename_dicom_path']
                            if path_str:
                                try:
                                    candidate_description_from_path = pathlib.Path(path_str).name
                                    log.debug(f"[SERIES {idx}] 暫存備用 description from path: {candidate_description_from_path}")
                                except Exception as e:
                                    log.warning(f"[SERIES {idx}] 解析路徑失敗: {path_str}, error: {e}")
            else:
                log.warning(f"[SERIES {idx}] result_data 為空")
            
            # 如果沒有找到明確的 series_description，使用備案
            if not series_description and candidate_description_from_path:
                log.info(f"[SERIES {idx}] ⚠️ 未找到明確的 series_description，使用路徑名稱作為替補: '{candidate_description_from_path}'")
                series_description = candidate_description_from_path
            
            # 檢查是否為 multi-output series
            is_multi, required_outputs = is_multi_output_series(series_description)
            log.info(f"[SERIES {idx}] Multi-output 檢測: is_multi={is_multi}, required_outputs={required_outputs}")
            
            # 提取所有 rename_dicom_path（去除重複）
            rename_paths = []
            rename_paths_set = set()  # 用於追蹤唯一的 paths
            for rd_idx, result_data in enumerate(dcop_event.result_data):
                if result_data and isinstance(result_data, dict):
                    if 'rename_dicom_path' in result_data:
                        path = result_data['rename_dicom_path']
                        if path and path not in rename_paths_set:
                            rename_paths.append(path)
                            rename_paths_set.add(path)
                            log.info(f"[SERIES {idx}] 提取唯一的 rename_dicom_path[{len(rename_paths)}]: {path}")
                        elif path and path in rename_paths_set:
                            log.debug(f"[SERIES {idx}] result_data[{rd_idx}] 的 rename_dicom_path 已存在，跳過重複: {path}")
                        else:
                            log.debug(f"[SERIES {idx}] result_data[{rd_idx}] 的 rename_dicom_path 為 None，跳過")
                    else:
                        log.debug(f"[SERIES {idx}] result_data[{rd_idx}] 沒有 rename_dicom_path 欄位")
            
            log.info(f"[SERIES {idx}] 總共提取到 {len(rename_paths)} 個唯一的 rename_dicom_path")
            
            if not rename_paths:
                log.warning(
                    f"[SERIES {idx}] ❌ 沒有找到任何 rename_dicom_path，跳過此 series"
                )
                continue
            
            if is_multi:
                # Multi-output series: 
                # 1. 嘗試從 result_data 中獲取 paths
                # 2. 如果 result_data 不完整（例如只有其中一個），則掃描檔案系統
                log.info(f"[SERIES {idx}] 🔀 Multi-output series detected! Checking file system for outputs...")
                log.info(f"[SERIES {idx}]    - Series Description: {series_description}")
                log.info(f"[SERIES {idx}]    - Required Outputs: {required_outputs}")

                # 使用第一個找到的路徑作為基準進行檔案系統掃描
                base_path = pathlib.Path(rename_paths[0])
                search_dirs = [base_path, base_path.parent]
                
                found_outputs_map = {} # key: output_name (e.g. DWI0), value: full_path

                log.info(f"[SERIES {idx}]    - Base Path for scan: {base_path}")
                log.info(f"[SERIES {idx}]    - Scanning directories: {search_dirs}")

                for search_dir in search_dirs:
                    if not search_dir.exists():
                        log.debug(f"[SERIES {idx}]    - Search dir does not exist: {search_dir}")
                        continue
                    
                    try:
                        subdirs = [d for d in search_dir.iterdir() if d.is_dir()]
                        for req_out in required_outputs:
                            # 如果已經找到了，就跳過
                            if req_out in found_outputs_map:
                                continue
                            
                            req_out_upper = req_out.upper()
                            for subdir in subdirs:
                                subdir_name_upper = subdir.name.upper()
                                # 檢查規則：目錄名稱等於 output 名稱，或者以 _OUTPUT 結尾
                                if subdir_name_upper == req_out_upper or subdir_name_upper.endswith(f"_{req_out_upper}"):
                                    found_outputs_map[req_out] = str(subdir)
                                    log.info(f"[SERIES {idx}]    - ✅ Found {req_out} on disk: {subdir}")
                                    break
                    except Exception as e:
                        log.error(f"[SERIES {idx}]    - Error scanning {search_dir}: {e}")

                # 準備最終要處理的 paths
                final_task_paths = []
                
                if found_outputs_map:
                    log.info(f"[SERIES {idx}] 🔀 File System Scan Results: Found {len(found_outputs_map)}/{len(required_outputs)} outputs")
                    for req_out in required_outputs:
                        if req_out in found_outputs_map:
                             final_task_paths.append(found_outputs_map[req_out])
                        else:
                             log.warning(f"[SERIES {idx}]    - ⚠️ Required output '{req_out}' NOT found on disk.")
                else:
                    log.warning(f"[SERIES {idx}] ⚠️ Is multi-output but found no matching folders on disk via scan. Falling back to DB paths.")
                    final_task_paths = rename_paths

                log.info(f"[SERIES {idx}]    - Creating tasks for {len(final_task_paths)} paths")

                for path_idx, output_dicom_path in enumerate(final_task_paths, 1):
                    output_name = pathlib.Path(output_dicom_path).name
                    
                    log.info(f"[SERIES {idx}] [OUTPUT {path_idx}/{len(final_task_paths)}] 建立 NIFTI 任務")
                    log.info(f"[SERIES {idx}] [OUTPUT {path_idx}]    - Output Name: {output_name}")
                    log.info(f"[SERIES {idx}] [OUTPUT {path_idx}]    - Output Path: {output_dicom_path}")
                    
                    output_nifti_path = pathlib.Path(path_rename_nifti)
                    task_params = Dicom2NiiSeriesParams(
                        sub_dir=None,
                        study_uid=dcop_event.study_uid,
                        series_uid=series_uid,
                        output_dicom_path=output_dicom_path,
                        output_nifti_path=output_nifti_path
                    )
                    
                    log.info(f"[SERIES {idx}] [OUTPUT {path_idx}] 建立 DCOPEventModel 記錄")
                    new_data_obj = await DCOPEventModel.create_event_ope_no(
                        tool_id='NIFTI_TOOL',
                        study_uid=dcop_event.study_uid,
                        series_uid=series_uid,
                        study_id=dcop_event.study_id,
                        ope_no=DCOPStatus.SERIES_CONVERTING.value,
                        result_data=dcop_event.result_data,
                        params_data=task_params.get_str_dict(),
                        session=session
                    )
                    
                    dcop_model_list.append(new_data_obj)
                    task_params_list.append(task_params)
                    log.info(f"[SERIES {idx}] [OUTPUT {path_idx}] ✅ 任務已加入佇列")
                
                log.info(f"[SERIES {idx}] 🔀 Multi-output 處理完成")
            else:
                # 單一 output series: 原有邏輯
                output_dicom_path = rename_paths[0]
                
                log.info(f"[SERIES {idx}] 📄 Single-output series")
                log.info(f"[SERIES {idx}]    - Series Description: {series_description}")
                log.info(f"[SERIES {idx}]    - Output Path: {output_dicom_path}")
                
                if len(rename_paths) > 1:
                    log.warning(
                        f"[SERIES {idx}] ⚠️  Single-output series 但找到 {len(rename_paths)} 個 paths，"
                        f"只使用第一個: {output_dicom_path}"
                    )
                
                output_nifti_path = pathlib.Path(path_rename_nifti)
                task_params = Dicom2NiiSeriesParams(
                    sub_dir=None,
                    study_uid=dcop_event.study_uid,
                    series_uid=series_uid,
                    output_dicom_path=output_dicom_path,
                    output_nifti_path=output_nifti_path
                )
                
                log.info(f"[SERIES {idx}] 建立 DCOPEventModel 記錄")
                new_data_obj = await DCOPEventModel.create_event_ope_no(
                    tool_id='NIFTI_TOOL',
                    study_uid=dcop_event.study_uid,
                    series_uid=series_uid,
                    study_id=dcop_event.study_id,
                    ope_no=DCOPStatus.SERIES_CONVERTING.value,
                    result_data=dcop_event.result_data,
                    params_data=task_params.get_str_dict(),
                    session=session
                )
                
                dcop_model_list.append(new_data_obj)
                task_params_list.append(task_params)
                log.info(f"[SERIES {idx}] 📄 ✅ 任務已加入佇列")

        log.info(f"\n{'='*80}")
        log.info(f"[SUMMARY] 處理完成")
        log.info(f"[SUMMARY] 總共處理了 {len(dcop_event_list)} 個 series")
        log.info(f"[SUMMARY] 建立了 {len(dcop_model_list)} 個 NIFTI 轉換任務")
        log.info(f"{'='*80}")

        try:
            if dcop_model_list:
                log.info(f'[DATABASE] 準備寫入 {len(dcop_model_list)} 個任務到資料庫')
                session.add_all(dcop_model_list)
                await session.commit()
                log.info(f'[DATABASE] ✅ 資料庫寫入成功')
                
                for idx, dcop_model in enumerate(dcop_model_list, 1):
                    await session.refresh(dcop_model)
                    log.debug(f'[DATABASE] 任務 {idx} 已 refresh，VsPrimaryKey: {dcop_model.VsPrimaryKey}')
                
                # 發送任務到 queue
                log.info(f'[QUEUE] 準備發送 {len(task_params_list)} 個任務到執行佇列')
                for idx, task_params in enumerate(task_params_list, 1):
                    log.info(f'[QUEUE] [{idx}/{len(task_params_list)}] 發送任務:')
                    log.info(f'[QUEUE]    - series_uid: {task_params.series_uid}')
                    log.info(f'[QUEUE]    - output_dicom_path: {task_params.output_dicom_path}')
                    dicom_2_nii_series.push(task_params.get_str_dict())
                    log.info(f'[QUEUE] [{idx}/{len(task_params_list)}] ✅ 任務已發送')
                
                log.info(f'[QUEUE] ✅ 所有任務已成功發送到佇列')
            else:
                log.info("[RESULT] 沒有需要建立的 NIFTI 任務")
        except Exception as e:
            await session.rollback()
            log.error(f"[ERROR] ❌ 建立 NIFTI 任務時發生錯誤: {e}")
            log.error(f"[ERROR] 資料庫事務已回滾")
            log.exception("完整錯誤堆疊:")
            raise
        finally:
            log.info(f"{'='*80}")
            log.info(f"[NIFTI_TOOL] 結束處理 study: {study_uid}")
            log.info(f"{'='*80}\n")

    @staticmethod
    def get_orthanc_study_uid_series_uid(instance_path_str: str):
        instance_path = pathlib.Path(instance_path_str)
        UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
        # raw_dicom\ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30\10089413 GUO HSIOU HUA\21002010079 MRI Stroke Wall C C\MR 3D Ax SWAN\*.dcm
        with open(instance_path_str, mode='rb') as f:
            dicom_ds = pydicom.dcmread(f)

        client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300)
        study_uid = instance_path.parent.parent.parent.parent.name
        # (0020,000E)	Series Instance UID	1.2.840.113619.2.44.5554020.7707121.19025.1612063861.703
        series_sop_uid = dicom_ds[0x0020, 0x000E].value
        # series_description = " ".join(instance_path.parent.name.split(" ")[1:]).strip()
        study = Study(study_uid, client=client)
        series_filter = list(filter(lambda series: series.uid == series_sop_uid, study.series))
        if series_filter:
            return str(study_uid), str(series_filter[0].id_)
        else:
            return None

    @staticmethod
    def get_orthanc_series_uid(study_uid: str,
                               series_dir_set: set):
        UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
        client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300)
        series_sop_uid_list = []
        for series_dir in series_dir_set:
            series_path_list = list(series_dir.rglob('*.dcm'))
            instance_path_str = series_path_list[0]
            with open(instance_path_str, mode='rb') as f:
                dicom_ds = pydicom.dcmread(f)
            series_sop_uid = dicom_ds[0x0020, 0x000E].value
            series_sop_uid_list.append(series_sop_uid)
        study = Study(study_uid, client=client)
        series_dict_list = list(map(lambda x: {'series_sop_uid': x.uid,
                                               'uid': x.id_,
                                               'description': x.description, }, study.series))
        df  = pd.DataFrame(series_sop_uid_list, columns=['file_series_sop_uid'])
        df1 = pd.DataFrame(series_dict_list)
        df2 = pd.merge(df, df1, left_on='file_series_sop_uid', right_on='series_sop_uid')
        return df2

    async def dicom_tool_get_series_info(self, data: List[DCOPEventModel]) -> None:
        """
        從 Orthanc DICOM 伺服器擷取 Series 資訊並建立轉檔任務。
        
        此方法在 Study 傳輸完成後調用，用於：
        1. 查詢 Orthanc 取得 Study 下的所有 Series
        2. 建立 SERIES_NEW 和 SERIES_TRANSFERRING 事件
        3. 排程 DICOM 到 NIFTI 的初始轉檔任務
        
        工作流程
        --------
        對於每個 Study：
        1. 掃描原始 DICOM 目錄找尋所有 .dcm 檔案
        2. 根據檔案路徑推斷 Series 目錄結構
        3. 從 DICOM 檔案中讀取 Series Instance UID
        4. 查詢 Orthanc 取得 Series 的標準 UID
        5. 為每個 Series 建立初始事件
        6. 排程轉檔任務
        
        Parameters
        ----------
        data : list[DCOPEventModel]
            STUDY_TRANSFERRING 事件清單。包含了轉檔所需的參數。
        
        Returns
        -------
        None
        
        Side Effects
        -----------
        - 在資料庫中為每個 Series 建立 2 個事件
        - 將轉檔任務推送到任務隊列
        - 記錄詳細的處理日誌
        
        Notes
        -----
        此方法具有容錯能力：
        - 若某個 Series 的事件建立失敗，會回滾該 Series，但繼續處理其他 Series
        - 檔案路徑不存在時會被跳過
        - 與 Orthanc 的連接超時設定為 300 秒
        
        Examples
        --------
        >>> study_events = [DCOPEventModel(...)]  # STUDY_TRANSFERRING 事件
        >>> await service.dicom_tool_get_series_info(study_events)
        # 結果: 為 Study 下的每個 Series 建立事件並入隊
        
        依賴配置
        --------
        - PATH_RAW_DICOM: 原始 DICOM 檔案位置
        - PATH_RENAME_DICOM: 重命名後的 DICOM 位置
        - PATH_RENAME_NIFTI: NIFTI 輸出位置
        - UPLOAD_DATA_DICOM_SEG_URL: Orthanc API 端點
        """
        from code_ai.task.task_dicom2nii import dicom_to_nii
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv
        load_dotenv()
        
        # 載入檔案路徑配置
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))

        for dcop_event in data:
            study_uid = dcop_event.study_uid
            self.logger.info(f'dicom_tool_get_series_info dcop_event {dcop_event}')
            study_uid_raw_dicom_path = raw_dicom_path.joinpath(study_uid)
            if study_uid_raw_dicom_path.exists():
                dcm_path_list = sorted(study_uid_raw_dicom_path.rglob('*.dcm'))
                series_dir_set = set([dcm_path.parent for dcm_path in dcm_path_list])
                df = self.get_orthanc_series_uid(study_uid, series_dir_set)
                series_uid_list = df['uid'].to_list()
                task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
                                              output_dicom_path=rename_dicom_path,
                                              output_nifti_path=rename_nifti_path, )
                flage = True
                for series_uid in series_uid_list:
                    new_data_list = []
                    async with self.session_manager.get_session() as session:
                        try:
                            series_new_data = await DCOPEventModel.create_event(study_uid=study_uid,
                                                                                series_uid=series_uid,
                                                                                status=DCOPStatus.SERIES_NEW.name,
                                                                                session=session, )
                            series_transferring_data = await DCOPEventModel.create_event(study_uid=study_uid,
                                                                                         series_uid=series_uid,
                                                                                         status=DCOPStatus.SERIES_TRANSFERRING.name,
                                                                                         session=session, )
                            series_transferring_data.params_data = task_params.get_str_dict()
                            new_data_list.append(series_new_data)
                            new_data_list.append(series_transferring_data)

                            session.add_all(new_data_list)
                            await session.commit()
                            self.logger.info(f'dicom_tool_get_series_info {new_data_list}')
                        except:
                            flage = False
                            await session.rollback()
                            self.logger.error(f'Error: {traceback.format_exc()}')
                if flage:
                    task = dicom_to_nii.push(task_params.get_str_dict())
        return None

    async def check_study_series_conversion_complete(
        self,
        data: Optional[List[DCOPEventRequest]] = None
    ) -> None:
        """
        檢查 Study/Series NIFTI 轉檔是否完成，若完成則進入推論階段。
        
        此方法是第二個 "檢查點" API，用於推動從轉檔到推論的轉遷。
        其工作流程為：
        
        1. 獲取所有狀態 ≥ SERIES_CONVERSION_COMPLETE 的 Series
        2. 驗證所有必要的 Series 都已完成轉檔
        3. 為已完成 Study 建立 STUDY_CONVERSION_COMPLETE 事件
        4. 建立 STUDY_INFERENCE_READY 事件，開始推論準備
        5. 建立 STUDY_INFERENCE_QUEUED 事件，推論入隊
        6. 排程推論任務到隊列
        
        狀態轉遷圖
        ----------
        SERIES_CONVERSION_COMPLETE (多個)
                    ↓
        [此方法檢查]
                    ↓
        STUDY_CONVERSION_COMPLETE
                    ↓
        STUDY_INFERENCE_READY
                    ↓
        STUDY_INFERENCE_QUEUED
                    ↓
        推論佇列
        
        Parameters
        ----------
        data : list[DCOPEventRequest], optional
            指定要檢查的事件列表。
            若為 None，則自動掃描資料庫中所有待檢查的 Study。
        
        Returns
        -------
        None
        
        Side Effects
        -----------
        - 在資料庫中建立多個事件
        - 在 Redis 中建立推論任務快取
        - 將任務推送到推論隊列
        - 透過 HTTP 報告狀態
        
        Examples
        --------
        自動掃描所有待檢查的 Study：
        
        >>> await service.check_study_series_conversion_complete()
        
        檢查指定的 Study：
        
        >>> events = [DCOPEventRequest(study_uid="abc-123", ope_no="200.195")]
        >>> await service.check_study_series_conversion_complete(data=events)
        
        Notes
        -----
        推論隊列快取：
        - 鍵格式: inference_task:{study_uid},{study_id}
        - 值: "queued"（實際值無關，只是占位符）
        - TTL: 6 小時（21600 秒）
        - 用途: 防止重複入隊
        
        Series 模式驗證：
        - 正常序列: NEW → TRANSFERRING → TRANSFER_COMPLETE → CONVERTING → CONVERSION_COMPLETE
        - 跳過序列: 可以是 CONVERSION_SKIP 而非 CONVERSION_COMPLETE
        - 規則: 所有 Series 必須終止於 COMPLETE 或 SKIP
        """
        from code_ai.task.task_pipeline import task_pipeline_inference

        # Environment variables setup
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
        # post_check_study_series_conversion_complete call check
        if data is None:
            # Query studies not yet at STUDY_CONVERSION_COMPLETE status
            completed_studies = await self.query_studies_pending_completion()
        else:
            completed_studies = set()
            for dcop_enent in data:
                result_set = await self.query_studies_pending_completion(dcop_enent.study_uid)
                self.logger.info(f'result_set {result_set}')
                completed_studies.update(result_set)

        if not completed_studies:
            return None
        self.logger.info(f'completed_studies {completed_studies}')
        # Create and send study completion events
        study_events = await self.create_study_complete_events(
            completed_studies,
            raw_dicom_path,
            rename_dicom_path,
            rename_nifti_path
        )
        # Process events from the provided data list
        completed_study_events = await self.identify_completed_studies(study_events)
        # Process completed studies and queue them for inference
        if completed_study_events:
            study_events_filter = []
            for completed_study in completed_study_events:
                study_event = list(filter(lambda x:x.study_uid == completed_study.study_uid,study_events))
                study_events_filter.extend(study_event)
            study_events_filter = list(map(lambda x:x.model_dump(),study_events_filter))
            await self._send_events(upload_data_api_url, study_events_filter)
            # Queue inference tasks for completed studies
            await self._queue_inference_tasks(
                completed_study_events,
                upload_data_api_url,
                rename_dicom_path,
                rename_nifti_path,
                task_pipeline_inference
            )
        return None

    async def query_studies_pending_completion(self, study_uid: Optional[str] = None):
        """Query for studies that have not yet reached STUDY_CONVERSION_COMPLETE status."""
        # async with self.repository.session as session:
        # --SELECT sos.study_id, debb.ope_no,sos.ope_no
        # --FROM  public.get_stydy_series_ope_no_status_create_time('200.200') as sos ,
        # --       (select deb.study_id, max(deb.ope_no::numeric)as ope_no from dcop_event_bt deb group by study_id  )  as debb
        # --where  sos.study_id = debb.study_id
        # --and debb.ope_no::NUMERIC <= ANY (sos.ope_no::NUMERIC[])
        # --order by sos.create_time desc
        async with self.session_manager.get_session() as session:
            if study_uid is None:
                sql = text('SELECT sos.study_uid , sos.series_uid , sos.study_id , sos.ope_no , sos.result_data , sos.params_data  FROM public.get_stydy_series_ope_no_status(:status) as sos , '
                           '(SELECT deb.study_id, max(deb.ope_no::numeric) as ope_no from dcop_event_bt deb group by study_id)  as debb '
                           'where  sos.study_id = debb.study_id and debb.ope_no::NUMERIC <= ANY (sos.ope_no::NUMERIC[])')
                params = {'status': DCOPStatus.STUDY_CONVERSION_COMPLETE.value}
            else:
                # --
                sql = text('SELECT sos.study_uid , sos.series_uid , sos.study_id , sos.ope_no , sos.result_data , sos.params_data FROM public.get_stydy_series_ope_no_status(:status) as sos , '
                           '(SELECT deb.study_id, max(deb.ope_no::numeric)as ope_no from dcop_event_bt deb where deb.study_uid= :study_uid group by study_id  )  as debb '
                           'where sos.study_uid=:study_uid '
                           'and sos.study_id = debb.study_id '
                           'and debb.ope_no::NUMERIC <= ANY (sos.ope_no::NUMERIC[]) ')
                # sql = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid')
                params = {'status': DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
                          'study_uid': study_uid}
            execute = await session.execute(sql, params)
            results = execute.all()

        can_inference_dict = {}
        wait_inference_dict = {}
        for result in results:
            test_str = ','.join(list(result.ope_no))
            match_result = self.can_inference_pattern.match(test_str)
            self.logger.info('match_result  {} , {}'.format(match_result, test_str))
            self.logger.info("{} {}".format(self.pattern_str, self.can_inference_pattern.findall(test_str)))
            if match_result:
                can_inference_dict.update({result.series_uid: (result.study_uid, result.study_id)})
            else:
                wait_inference_dict.update({result.series_uid: (result.study_uid, result.study_id)})

        wait_inference_set = set(wait_inference_dict.values())
        can_inference_set = set(can_inference_dict.values())
        if wait_inference_dict:
            result_set = can_inference_set - wait_inference_set
        else:
            result_set = can_inference_set

        return result_set

    async def create_study_complete_events(self, study_data_list, raw_dicom_path, rename_dicom_path,
                                           rename_nifti_path):
        """Create STUDY_CONVERSION_COMPLETE events for studies with all series converted."""
        study_events = []

        for data in study_data_list:
            study_uid_raw_dicom_path = raw_dicom_path.joinpath(data[0])

            dcop_event = DCOPEventRequest(
                study_uid=data[0],
                series_uid=None,
                study_id=data[1],
                ope_no=DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
                tool_id='NIFTI_TOOL',
                params_data=dict(
                    sub_dir=str(study_uid_raw_dicom_path),
                    output_dicom_path=str(rename_dicom_path),
                    output_nifti_path=str(rename_nifti_path),
                ),
                result_data=dict(
                    sub_dir=str(study_uid_raw_dicom_path),
                    output_dicom_path=str(rename_dicom_path.joinpath(data[1])),
                    output_nifti_path=str(rename_nifti_path.joinpath(data[1]))
                )
            )
            study_events.append(dcop_event)

        self.logger.info(f'result_set {study_data_list}')
        return study_events

    async def _queue_inference_tasks(self, study_events, upload_data_api_url, rename_dicom_path,
                                     rename_nifti_path, task_pipeline_inference):
        """Queue inference tasks for completed studies and send related events."""
        redis_backend = FastAPICache.get_backend()
        redis_client = redis_backend.redis

        for dcop_event in study_events:
            dicom_study_path = rename_dicom_path.joinpath(dcop_event.study_id)
            nifti_study_path = rename_nifti_path.joinpath(dcop_event.study_id)

            # Create STUDY_INFERENCE_READY event
            dcop_event_inference_ready = DCOPEventRequest(
                study_uid=dcop_event.study_uid,
                series_uid=None,
                study_id=dcop_event.study_id,
                ope_no=DCOPStatus.STUDY_INFERENCE_READY.value,
                tool_id='INFERENCE_TOOL',
                params_data={
                    'nifti_study_path': str(nifti_study_path),
                    'dicom_study_path': str(dicom_study_path),
                    'study_uid': dcop_event.study_uid,
                    'study_id': dcop_event.study_id
                }
            )
            inference_task_key = f"inference_task:{dcop_event.study_uid},{dcop_event.study_id}"
            if await redis_client.get(inference_task_key):
                self.logger.info(
                    f"Skipping duplicate inference task for study_id: {dcop_event.study_id}. Already in cache.")
                continue  # Skip this study_event and move to the next one

            # Push to inference task pipeline
            task_pipeline_result: AsyncResult = task_pipeline_inference.push(
                dcop_event_inference_ready.params_data
            )
            await redis_client.set(inference_task_key, "queued", ex=21600)  # Value can be anything, key is what matters
            self.logger.info(f"Added study_uid: {dcop_event.study_uid} to cache with key: {inference_task_key}")

            # Create STUDY_INFERENCE_QUEUED event
            dcop_event_inference_queued = DCOPEventRequest(
                study_uid=dcop_event.study_uid,
                series_uid=None,
                study_id=dcop_event.study_id,
                ope_no=DCOPStatus.STUDY_INFERENCE_QUEUED.value,
                tool_id='INFERENCE_TOOL',
                params_data={
                    'nifti_study_path': str(nifti_study_path),
                    'dicom_study_path': str(dicom_study_path),
                    'study_uid': dcop_event.study_uid,
                    'study_id': dcop_event.study_id,
                    'task_pipeline_id': task_pipeline_result.task_id
                }
            )

            # Send inference events
            await self._send_events(upload_data_api_url,
                                    [dcop_event_inference_ready.model_dump(),
                                     dcop_event_inference_queued.model_dump()])

    def _group_series_by_study(self, events):
        """Group series completion events by study."""
        series_by_study = {}

        for event in events:
            if event.ope_no == DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
                if event.study_uid not in series_by_study:
                    series_by_study[event.study_uid] = {
                        'completed': set(),
                        'study_id': event.study_id
                    }

                # Add this series to the completed set
                if event.series_uid:
                    series_by_study[event.study_uid]['completed'].add(event.series_uid)

        return series_by_study

    async def identify_completed_studies(self, study_events_list: List[DCOPEventRequest]):
        """Identify studies with all series converted and create completion events."""
        completed_study_events = []
        # Query to get all series for this study
        async with self.session_manager.get_session() as session:
            done_count = 0
            undone = 0
            for study_events in study_events_list:
                sql = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid')
                params = {'status': DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
                          'study_uid': study_events.study_uid}
                execute = await session.execute(sql, params)
                results = execute.all()
                for result in results:
                    if DCOPStatus.SERIES_CONVERSION_COMPLETE.value in result.ope_no:
                        done_count += 1
                    elif DCOPStatus.SERIES_CONVERSION_SKIP.value in result.ope_no:
                        done_count += 1
                    else:
                        undone += 1
                if done_count == len(results):
                    completed_study_events.append(result)
        return completed_study_events


    async def get_stydy_series_ope_no_status(
        self,
        study_uid: OrthancID,
        ope_no: OpeNo,
        limit: int,
        offset: int
    ) -> OffsetPagination[StydySeriesOpeNoStatus]:
        """
        查詢特定 Study 下所有 Series 的特定操作狀態。
        
        此方法用於前端顯示 Study 下所有 Series 在特定操作階段的狀態。
        例如查詢所有 Series 是否都已轉檔完成。
        
        Parameters
        ----------
        study_uid : OrthancID
            Study UID。
        ope_no : OpeNo
            操作編號（e.g. "100.095" 表示傳輸完成）。
        limit : int
            每頁結果數。
        offset : int
            分頁偏移量。
        
        Returns
        -------
        OffsetPagination[StydySeriesOpeNoStatus]
            分頁結果，包含 Series 狀態列表和總計數。
        
        Examples
        --------
        查詢 Study 下傳輸完成的所有 Series：
        
        >>> result = await service.get_stydy_series_ope_no_status(
        ...     study_uid="abc-123",
        ...     ope_no="100.095",
        ...     limit=20,
        ...     offset=0
        ... )
        >>> print(f"找到 {result.total} 個 Series")
        
        Notes
        -----
        此方法使用資料庫函數 get_stydy_series_ope_no_status()，
        該函數返回所有經過特定操作編號的 Series。
        """
        async with self.session_manager.get_session() as session:
            if study_uid is None:
                sql = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status) LIMIT :limit OFFSET :offset')
                params = {'status': ope_no, 'limit': limit, 'offset': offset}
                count_sql = text('SELECT COUNT(*) FROM public.get_stydy_series_ope_no_status(:status)')
                count_params = {'status': ope_no}
            else:
                sql = text(
                    'SELECT * FROM public.get_stydy_series_ope_no_status(:status) WHERE study_uid = :study_uid LIMIT :limit OFFSET :offset')
                params = {'status': ope_no, 'study_uid': study_uid, 'limit': limit, 'offset': offset}
                count_sql = text(
                    'SELECT COUNT(*) FROM public.get_stydy_series_ope_no_status(:status) WHERE study_uid = :study_uid')
                count_params = {'status': ope_no, 'study_uid': study_uid}

            total_count_result = await session.execute(count_sql, count_params)
            total_count = total_count_result.scalar_one()
            execute = await session.execute(sql, params)
            results = execute.all()
            items = [StydySeriesOpeNoStatus.model_validate(row) for row in results]
            return OffsetPagination(
                items=items,
                total=total_count,
                limit=limit,
                offset=offset,
            )

    async def get_stydy_ope_no_status(
        self,
        study_uid: OrthancID,
        ope_no: OpeNo,
        limit: int,
        offset: int
    ) -> OffsetPagination[StydySeriesOpeNoStatus]:
        """
        查詢 Study 維度的特定操作狀態及其所有 Series 的狀態。
        
        此方法返回 Study 級別的聚合視圖，包含 Study 狀態和所有 Series 的狀態。
        用於前端顯示 Study 的完整進度。
        
        Parameters
        ----------
        study_uid : OrthancID
            Study UID。
        ope_no : OpeNo
            操作編號。
        limit : int
            每頁結果數。
        offset : int
            分頁偏移量。
        
        Returns
        -------
        OffsetPagination[StydySeriesOpeNoStatus]
            Study 維度的聚合狀態。
        
        Examples
        --------
        查詢 Study 的轉檔完成狀態：
        
        >>> result = await service.get_stydy_ope_no_status(
        ...     study_uid="abc-123",
        ...     ope_no="200.200",
        ...     limit=20,
        ...     offset=0
        ... )
        
        Notes
        -----
        此方法使用資料庫函數 get_stydy_ope_no_status()，
        返回 Study 級別的聚合視圖。
        """
        async with self.session_manager.get_session() as session:
            if study_uid is None:
                sql = text('SELECT * FROM public.get_stydy_ope_no_status(:status) LIMIT :limit OFFSET :offset')
                params = {'status': ope_no, 'limit': limit, 'offset': offset}
                count_sql = text('SELECT COUNT(*) FROM public.get_stydy_ope_no_status(:status)')
                count_params = {'status': ope_no}
            else:
                sql = text(
                    'SELECT * FROM public.get_stydy_ope_no_status(:status) WHERE study_uid = :study_uid LIMIT :limit OFFSET :offset')
                params = {'status': ope_no, 'study_uid': study_uid, 'limit': limit, 'offset': offset}
                count_sql = text(
                    'SELECT COUNT(*) FROM public.get_stydy_ope_no_status(:status) WHERE study_uid = :study_uid')
                count_params = {'status': ope_no, 'study_uid': study_uid}

            total_count_result = await session.execute(count_sql, count_params)
            total_count = total_count_result.scalar_one()
            execute = await session.execute(sql, params)
            results = execute.all()
            items = [StydySeriesOpeNoStatus.model_validate(row) for row in results]
            return OffsetPagination(
                items=items,
                total=total_count,
                limit=limit,
                offset=offset,
            )


    async def get_check_study_series_conversion_complete(
        self,
        study_uid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        查詢已完成 NIFTI 轉檔的 Study 列表。
        
        此方法是前端用於查詢哪些 Study 已準備好進入推論階段的主要端點。
        返回兩部分資訊：
        1. 待完成的 Study（轉檔未完全完成）
        2. 已完成的 Study（所有 Series 都已轉檔）
        
        Parameters
        ----------
        study_uid : str, optional
            可選的 Study UID 過濾。若提供，只查詢此 Study。
        
        Returns
        -------
        dict
            包含兩個鍵：
            - 'studies_pending_completion': 套組，包含 (study_uid, study_id)
            - 'completed_study_events': 列表，已完成 Study 的詳細狀態
        
        Examples
        --------
        查詢所有已完成轉檔的 Study：
        
        >>> result = await service.get_check_study_series_conversion_complete()
        >>> for event in result['completed_study_events']:
        ...     print(f"Study {event.study_id} is ready for inference")
        
        查詢特定 Study：
        
        >>> result = await service.get_check_study_series_conversion_complete(
        ...     study_uid="abc-123"
        ... )
        
        Notes
        -----
        此方法用於前端查詢進度，不進行狀態轉遷。
        實際的狀態轉遷由 check_study_series_conversion_complete() 執行。
        """
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
        
        # 查詢待完成的 Study
        completed_studies = await self.query_studies_pending_completion(study_uid=study_uid)
        
        # 為每個 Study 建立完成事件
        study_events = await self.create_study_complete_events(
            completed_studies,
            raw_dicom_path,
            rename_dicom_path,
            rename_nifti_path
        )
        
        # 識別已完成的 Study
        completed_study_events = await self.identify_completed_studies(study_events)
        
        return {
            'studies_pending_completion': completed_studies,
            'completed_study_events': [
                StydySeriesOpeNoStatus.model_validate(result)
                for result in completed_study_events
            ]
        }
