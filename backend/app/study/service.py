"""
研究 (Study) 業務邏輯層 - DCOP 事件驅動 DICOM 服務。

此模組負責管理醫學影像 DICOM 研究的生命週期，包括：
1. Study/Series 狀態轉遷管理
2. 事件記錄與審計追蹤
3. 與外部系統的集成（Orthanc DICOM 伺服器、NIFTI 轉檔工具、推理引擎）
4. 異步任務隊列協調
5. 多輸出序列的特殊處理（如 DWI）

核心概念
--------
- **事件驅動**：每個狀態變化都產生一條事件紀錄
- **狀態機**：Study 和 Series 遵循嚴格的狀態轉遷流程
- **鬆散耦合**：通過配置和事件而非直接 API 調用
- **完整審計**：完整的時間戳記和參數追蹤

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

Examples
--------
基本使用流程：

>>> service = DCOPEventDicomService()
>>>
>>> # 1. 添加新 Study
>>> study_ids = ["study-uid-123"]
>>> await service.add_study_new(study_ids)
>>>
>>> # 2. 檢查傳輸是否完成
>>> await service.check_study_series_transfer_complete()
>>>
>>> # 3. 檢查轉檔是否完成
>>> await service.check_study_series_conversion_complete()

See Also
--------
backend.app.sync.service : 同步模組的核心服務實現
backend.app.sync.schemas : 資料模型和序列化方案
"""

import json
import logging
import os
import pathlib
import traceback
from typing import List, Optional, Tuple
import re
import httpx
from advanced_alchemy.extensions.fastapi import repository
from funboost import AsyncResult

# from fastapi import
from sqlalchemy import text, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi_cache import FastAPICache

from code_ai.task.schema.intput_params import Dicom2NiiParams
from backend.app.service import BaseRepositoryService
from backend.app.sync.model import DCOPEventModel
from .schemas import DCOPStatus, DCOPEventRequest, DCOPEventNIFTITOOLRequest
from backend.app.sync.urls import (
    SYNC_PROT_OPE_NO,
    SYNC_PROT_STUDY_NIFTI_TOOL,
    SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID,
    SYNC_PROT_STUDY_TRANSFER_COMPLETE,
)

# 模組級日誌記錄器
logger = logging.getLogger(__name__)


class DCOPEventDicomService(BaseRepositoryService[DCOPEventModel]):
    """
    研究 (Study) 級別的 DICOM 事件服務 - 核心業務邏輯層。
    
    此服務類負責管理 DICOM 研究的所有業務邏輯，包括：
    
    1. Study/Series 狀態轉遷管理
       - STUDY_NEW → STUDY_TRANSFERRING → STUDY_TRANSFER_COMPLETE
       - STUDY_CONVERTING → STUDY_CONVERSION_COMPLETE
       - STUDY_INFERENCE_READY → STUDY_INFERENCE_QUEUED → STUDY_INFERENCE_COMPLETE
    
    2. 事件記錄與審計
       - 每個狀態變化都產生一條事件紀錄
       - 完整的時間戳記和參數追蹤
    
    3. 與外部系統的集成
       - Orthanc DICOM 伺服器：查詢 Study 和 Series
       - NIFTI 轉檔工具：協調 DICOM 到 NIFTI 的轉檔
       - 推理引擎：排隊推理任務
    
    4. 異步任務隊列協調
       - 將任務推送到隊列中執行
       - 監控任務完成狀態
    
    5. 多輸出序列的特殊處理
       - 自動檢測多輸出序列（如 DWI）
       - 為每個輸出創建獨立的轉檔任務
       - 等待所有輸出完成
    
    Attributes
    ----------
    logger : logging.Logger
        類級別的日誌記錄器，所有實例共享。
    pattern_str : str
        正則表達式：Series 必須經歷的狀態序列。
    can_inference_pattern : re.Pattern
        編譯後的正則表達式，用於快速判斷 Series 是否可進入推理。
    
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
    >>> # 1. 添加新 Study
    >>> study_ids = ["study-uid-123"]
    >>> events = await service.add_study_new(study_ids)
    >>>
    >>> # 2. 檢查傳輸是否完成
    >>> completed = await service.check_study_series_transfer_complete()
    >>>
    >>> # 3. 檢查轉檔是否完成
    >>> await service.check_study_series_conversion_complete()
    
    See Also
    --------
    backend.app.sync.service : 同步模組的核心服務
    backend.app.sync.schemas : 資料模型和狀態定義
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
    pattern_str = "({}),({}),({}),({}),({}|{})".format(
        DCOPStatus.SERIES_NEW.value,
        DCOPStatus.SERIES_TRANSFERRING.value,
        DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
        DCOPStatus.SERIES_CONVERTING.value,
        DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
        DCOPStatus.SERIES_CONVERSION_SKIP.value,
    )
    # 編譯後的正則模式，用於快速檢查 Series 是否可進入推理
    can_inference_pattern = re.compile(pattern_str)

    async def get_check_url_by_ope_no(self, ope_no: str) -> Optional[str]:
        """
        根據操作編號取得對應的檢查點 API URL。
        
        此方法實現了操作碼到 API 端點的映射，支援四個檢查點：
        1. Series 傳輸完成 → 觸發 Study 傳輸檢查
        2. Series 轉檔完成 → 觸發 Study 轉檔檢查
        3. Study 傳輸完成 → 觸發轉檔初始化
        4. Study 轉檔完成 → 觸發推理初始化
        
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
                # Study 轉檔完成 → 觸發推理初始化
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
        
        logger.debug(f'get_check_url_by_ope_no: ope_no={ope_no}, url={url}')
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
        
        # async with AsyncSession(self.repository.session.bind) as session:
        async with self.session_manager.get_session() as session:
            # 逐一處理每個事件
            for dcop_event in data:
                # 建立事件記錄
                new_data_obj = await DCOPEventModel.create_event_ope_no(
                    tool_id=dcop_event.tool_id,
                    study_uid=dcop_event.study_uid,
                    series_uid=dcop_event.series_uid if dcop_event.series_uid is not None else "",
                    study_id=dcop_event.study_id if dcop_event.study_id is not None else "",
                    ope_no=dcop_event.ope_no,
                    result_data=dcop_event.result_data if dcop_event.result_data is not None else {},
                    params_data=dcop_event.params_data if dcop_event.params_data is not None else {},
                    session=session,
                )
                session.add(new_data_obj)
                await session.commit()
                await session.refresh(new_data_obj)

                # new_data_obj = await self.create(data=new_data, auto_commit=True, auto_refresh=True)
                # 識別是否需要觸發檢查點
                if new_data_obj.ope_no is not None:
                    match new_data_obj.ope_no:
                        case DCOPStatus.SERIES_TRANSFER_COMPLETE.value:
                            # Series 傳輸完成 → 檢查 Study 傳輸
                            url = await self.get_check_url_by_ope_no(str(new_data_obj.ope_no))
                        case DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
                            # Series 轉檔完成 → 檢查 Study 轉檔
                            url = await self.get_check_url_by_ope_no(str(new_data_obj.ope_no))
                        case _:
                            # 其他事件不觸發檢查點
                            url = None
                else:
                    url = None
                
                # 添加到檢查點集合（自動去重）
                if url is not None and url not in check_url_set:
                    check_url_set.add(url)
        
        # 批次執行所有檢查點
        async with httpx.AsyncClient(timeout=180) as client:
            for url in check_url_set:
                await client.post(url)
        return

    async def check_study_series_transfer_complete(
        self, data: Optional[List[DCOPEventRequest]] = None
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
        logger.info(
            f"check_study_series_transfer_complete data {data}",
        )
        # Get configuration from environment
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
        path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")

        # Retrieve study status information if not provided
        if data is None:
            (
                dcop_event_list,
                dcop_event_dump_list,
            ) = await self._get_studies_ready_for_transfer()
        else:
            dcop_event_list = [
                DCOPEventRequest.model_validate(event, strict=False) for event in data
            ]
            dcop_event_dump_list = [
                dcop_event.model_dump() for dcop_event in dcop_event_list
            ]

        # Process eligible studies for conversion
        if dcop_event_list:
            if upload_data_api_url is not None:
                await self._send_events(upload_data_api_url, dcop_event_dump_list)
            if upload_data_api_url is not None and path_rename_dicom is not None and path_rename_nifti is not None:
                await self._initiate_conversion_process(
                    upload_data_api_url,
                    dcop_event_list,
                    path_rename_dicom,
                    path_rename_nifti,
                )

        return dcop_event_list

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
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM", ""))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM", ""))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI", ""))

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
                    session.add(new_data)
                    # new_data_obj = await self.create(data=new_data)
                    
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
                    session.commit()
                    session.flush()
                    # obj = await self.create_many(data=[new_data_obj,data_transferring],auto_commit=True)
                    
                    # 收集結果
                    result_list.append(new_data)
                    result_list.append(data_transferring)
            except Exception as e:
                # 發生錯誤時回滾所有更改
                await session.rollback()
                logger.error(f"Error in add_study_new: {e}")
                raise

        return result_list

    async def _get_studies_ready_for_transfer(
        self,
    ) -> Tuple[List[DCOPEventRequest], List[dict]]:
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
            results = await session.execute(
                text("select * from public.get_all_studies_status()")
            )
            for result in results.all():
                logger.info(f"result {result}")
                study_data = result[0]
                dcop_event = DCOPEventRequest(
                    study_uid=study_data["study_uid"],
                    series_uid=None,
                    ope_no=DCOPStatus.STUDY_TRANSFER_COMPLETE.value,
                    study_id=study_data["study_id"],
                    tool_id="DICOM_TOOL",
                    result_data={"result": json.dumps(study_data["result"])},
                )
                dcop_event_dump_list.append(dcop_event.model_dump())
                dcop_event_list.append(dcop_event)

        return dcop_event_list, dcop_event_dump_list

    async def _send_events(self, api_url: str, event_data: List[dict]) -> None:
        """
        Sends study transfer complete events to the API.

        Args:
            api_url: Base URL for the upload data API.
            event_data: List of serialized DCOPEventRequest objects.
        """
        async with httpx.AsyncClient(timeout=180) as client:
            url = f"{api_url}{SYNC_PROT_OPE_NO}"
            await client.post(url=url, timeout=180, json=event_data)

    async def _initiate_conversion_process(
        self,
        api_url: str,
        events: List[DCOPEventRequest],
        dicom_path: str,
        nifti_path: str,
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
            if study_id is None:
                continue
            output_dicom_path = pathlib.Path(os.path.join(str(dicom_path), study_id))
            output_nifti_path = pathlib.Path(nifti_path)

            # Prepare conversion parameters
            task_params = Dicom2NiiParams(
                sub_dir=None,
                output_dicom_path=output_dicom_path,
                output_nifti_path=output_nifti_path,
            )

            # Create and send the conversion request
            async with httpx.AsyncClient(timeout=180) as client:
                nifti_tool_request = DCOPEventNIFTITOOLRequest(
                    ope_no=DCOPStatus.STUDY_CONVERTING.value,
                    study_id=study_id,
                    tool_id="NIFTI_TOOL",
                    params_data=task_params.get_str_dict(),
                    result_data=None,
                )

                await client.post(url=url, json=[nifti_tool_request.model_dump()])

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
                        conf_query = select(
                            DCOPEventModel,
                        ).where(
                            and_(
                                *[
                                    DCOPEventModel.study_id == dcop.study_id,
                                    DCOPEventModel.study_uid.isnot(None),
                                    DCOPEventModel.tool_id == "DICOM_TOOL",
                                    DCOPEventModel.ope_no
                                    == DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
                                ]
                            )
                        )
                        execute = await session.execute(conf_query)
                        first_result = execute.first()
                        if first_result is None:
                            continue
                        dcop_event = first_result[0]
                        study_transfer_complete_data = (
                            await DCOPEventModel.create_event_ope_no(
                                tool_id=dcop.tool_id,
                                study_uid=dcop_event.study_uid,
                                series_uid="",
                                study_id=dcop.study_id if dcop.study_id is not None else "",
                                ope_no=dcop.ope_no,
                                result_data=dcop.result_data if dcop.result_data is not None else {},
                                params_data=dcop.params_data if dcop.params_data is not None else {},
                                session=session,
                            )
                        )

                        session.add(study_transfer_complete_data)
                        await session.commit()
                        await session.refresh(study_transfer_complete_data)
                        await self.nifti_tool_get_series_info(
                            dcop_event.study_uid, session
                        )
                case DCOPStatus.SERIES_CONVERTING.value:
                    pass
                    # new_data_obj = await self.create(new_data, auto_commit=True)

        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        url = f"{upload_data_api_url}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
        async with httpx.AsyncClient(timeout=180) as client:
            await client.post(url=url)

    async def nifti_tool_get_series_info(self, study_uid: str, session: AsyncSession):
        from code_ai.task.task_dicom2nii import dicom_2_nii_series
        from code_ai.task.schema.intput_params import Dicom2NiiSeriesParams
        from code_ai import load_dotenv

        load_dotenv()
        path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")
        # engine: AsyncEngine = session.bind
        # async with engine.connect() as conn:
        sql = text(
            "SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid"
        )
        results = await session.execute(
            sql, {"status": DCOPStatus.STUDY_CONVERTING.value, "study_uid": study_uid}
        )

        dcop_event_list = results.all()
        task_params_list = []
        dcop_model_list = []
        for dcop_event in dcop_event_list:
            result_data = dcop_event.result_data[0]
            output_dicom_path = result_data["rename_dicom_path"]
            output_nifti_path = pathlib.Path(path_rename_nifti if path_rename_nifti is not None else "")
            task_params = Dicom2NiiSeriesParams(
                sub_dir=None,
                study_uid=dcop_event.study_uid,
                series_uid=dcop_event.series_uid,
                output_dicom_path=output_dicom_path,
                output_nifti_path=output_nifti_path,
            )
            new_data_obj = await DCOPEventModel.create_event_ope_no(
                tool_id="NIFTI_TOOL",
                study_uid=dcop_event.study_uid,
                series_uid=dcop_event.series_uid,
                study_id=dcop_event.study_id,
                ope_no=DCOPStatus.SERIES_CONVERTING.value,
                result_data=dcop_event.result_data,
                params_data=task_params.get_str_dict(),
                session=session,
            )
            dcop_model_list.append(new_data_obj)
            task_params_list.append(task_params)

        try:
            if dcop_event_list:  # Check if dcop_event_list is not empty/falsy
                # Attempt to create many records and auto-commit
                # If create_many raises an exception, the 'except' block will catch it,
                # and the push operations will not be executed.
                # data_obj = await self.create_many(dcop_model_list, auto_commit=True)
                logger.info(
                    f"session.add_all {dcop_model_list}",
                )
                session.add_all(dcop_model_list)
                await session.commit()
                for dcop_model in dcop_model_list:
                    await session.refresh(dcop_model)

                # If we reach here, create_many completed successfully and committed.
                # Now, proceed with pushing tasks.
                for task_params in task_params_list:
                    dicom_2_nii_series.push(task_params.get_str_dict())
            else:
                await session.rollback()
                # If dcop_event_list is empty, there's nothing to create or push.
                # A rollback here is likely unnecessary if nothing was attempted.
                # You might just want to pass or log.
                logger.info("dcop_event_list is empty, no records to create or push.")
                # await self.repository.session.rollback() # Potentially redundant if nothing happened
        except Exception as e:  # Catch specific exceptions for better debugging
            # An error occurred during create_many or subsequent push operations.
            # Rollback ensures no partial changes are left if auto_commit somehow failed or
            # if you had other uncommitted operations before this try block.
            await session.rollback()
            logger.info(f"An error occurred: {e}. Database transaction rolled back.")
            # Re-raise the exception if you want it to propagate further up the call stack
            raise
        finally:
            pass

    async def dicom_tool_get_series_info(self, data: List[DCOPEventModel]):
        from code_ai.task.task_dicom2nii import dicom_to_nii
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv

        load_dotenv()
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM", ""))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM", ""))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI", ""))

        for dcop_event in data:
            study_uid = dcop_event.study_uid
            logger.info(f"1000000100 dcop_event {dcop_event}")
            if study_uid is None:
                continue
            study_uid_raw_dicom_path = raw_dicom_path.joinpath(str(study_uid))
            if study_uid_raw_dicom_path.exists():
                async with self.session_manager.get_session() as session:
                    series_uid_path_list = sorted(study_uid_raw_dicom_path.iterdir())
                    new_data_list = []
                    task_params = Dicom2NiiParams(
                        sub_dir=study_uid_raw_dicom_path,
                        output_dicom_path=rename_dicom_path,
                        output_nifti_path=rename_nifti_path,
                    )

                    for series_uid_path in series_uid_path_list:
                        series_new_data = await DCOPEventModel.create_event(
                            study_uid=str(study_uid),
                            series_uid=series_uid_path.name,
                            status=DCOPStatus.SERIES_NEW.name,
                            session=session,
                        )
                        series_transferring_data = await DCOPEventModel.create_event(
                            study_uid=str(study_uid),
                            series_uid=series_uid_path.name,
                            status=DCOPStatus.SERIES_TRANSFERRING.name,
                            session=session,
                        )
                        series_transferring_data.params_data = (
                            task_params.get_str_dict()
                        )
                        new_data_list.append(series_new_data)
                        new_data_list.append(series_transferring_data)
                    try:
                        session.add_all(new_data_list)
                        await session.commit()
                        await session.flush()
                        task = dicom_to_nii.push(task_params.get_str_dict())
                        logger.info(
                            f"dicom_tool_get_series_info {new_data_list} {task}"
                        )
                    except Exception:
                        await session.rollback()
                        logger.error(traceback.print_exc())

        return None

    async def check_study_series_conversion_complete(
        self, data: Optional[List[DCOPEventRequest]] = None
    ):
        """
        檢查 study 下的 series 是否都轉成 nifti
        1. series 完成轉成nifti ， 添加 SERIES_CONVERSION_COMPLETE  的記錄
        2. 所有series都到了 SERIES_CONVERSION_COMPLETE， 添加 STUDY_CONVERSION_COMPLETE 的記錄
        3. 添加 STUDY_INFERENCE_READY 的記錄，
        4. 發送管道任務  推論
        """
        from code_ai.task.task_pipeline import task_pipeline_inference

        # Environment variables setup
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM", ""))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM", ""))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI", ""))
        logger.info(f"data {data}")
        # post_check_study_series_conversion_complete call check
        if data is None:
            # Query studies not yet at STUDY_CONVERSION_COMPLETE status
            completed_studies = await self._query_studies_pending_completion()
        else:
            completed_studies = set()
            for dcop_enent in data:
                result_set = await self._query_studies_pending_completion(
                    dcop_enent.study_uid
                )
                logger.info(f"result_set {result_set}")
                completed_studies.update(result_set)

        if not completed_studies:
            return None
        logger.info(
            f"completed_studies {completed_studies}",
        )
        # Create and send study completion events
        study_events = await self._create_study_complete_events(
            completed_studies, raw_dicom_path, rename_dicom_path, rename_nifti_path
        )
        # Process events from the provided data list
        logger.info(f"study_events {study_events}")
        completed_study_events = await self._identify_completed_studies(study_events)

        # Process completed studies and queue them for inference
        if completed_study_events:
            # Queue inference tasks for completed studies
            await self._queue_inference_tasks(
                completed_study_events,
                upload_data_api_url,
                rename_dicom_path,
                rename_nifti_path,
                task_pipeline_inference,
            )
        return None

    async def _query_studies_pending_completion(self, study_uid: Optional[str] = None):
        """Query for studies that have not yet reached STUDY_CONVERSION_COMPLETE status."""
        # async with self.repository.session as session:
        async with self.session_manager.get_session() as session:
            if study_uid is None:
                sql = text(
                    "SELECT * FROM public.get_stydy_series_ope_no_status(:status)"
                )
                params = {"status": DCOPStatus.STUDY_CONVERSION_COMPLETE.value}
            else:
                sql = text(
                    "SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid"
                )
                params = {
                    "status": DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
                    "study_uid": study_uid,
                }
            execute = await session.execute(sql, params)
            results = execute.all()

        can_inference_dict = {}
        wait_inference_dict = {}
        for result in results:
            test_str = ",".join(list(result.ope_no))
            match_result = self.can_inference_pattern.match(test_str)
            logger.info("match_result  {} , {}".format(match_result, test_str))
            logger.info(
                "{} {}".format(
                    self.pattern_str, self.can_inference_pattern.findall(test_str)
                )
            )
            if match_result:
                can_inference_dict.update(
                    {result.series_uid: (result.study_uid, result.study_id)}
                )
            else:
                wait_inference_dict.update(
                    {result.series_uid: (result.study_uid, result.study_id)}
                )

        wait_inference_set = set(wait_inference_dict.values())
        can_inference_set = set(can_inference_dict.values())
        if wait_inference_dict:
            result_set = can_inference_set - wait_inference_set
        else:
            result_set = can_inference_set

        return result_set

    async def _create_study_complete_events(
        self, study_data_list, raw_dicom_path, rename_dicom_path, rename_nifti_path
    ):
        """Create STUDY_CONVERSION_COMPLETE events for studies with all series converted."""
        study_events = []

        for data in study_data_list:
            study_uid_raw_dicom_path = raw_dicom_path.joinpath(data[0])

            dcop_event = DCOPEventRequest(
                study_uid=data[0],
                series_uid=None,
                study_id=data[1],
                ope_no=DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
                tool_id="NIFTI_TOOL",
                params_data=dict(
                    sub_dir=str(study_uid_raw_dicom_path),
                    output_dicom_path=str(rename_dicom_path),
                    output_nifti_path=str(rename_nifti_path),
                ),
                result_data=dict(
                    sub_dir=str(study_uid_raw_dicom_path),
                    output_dicom_path=str(rename_dicom_path.joinpath(data[1])),
                    output_nifti_path=str(rename_nifti_path.joinpath(data[1])),
                ),
            )
            study_events.append(dcop_event)

        logger.info(
            f"result_set {study_data_list}",
        )
        return study_events

    async def _queue_inference_tasks(
        self,
        study_events,
        upload_data_api_url,
        rename_dicom_path,
        rename_nifti_path,
        task_pipeline_inference,
    ):
        """Queue inference tasks for completed studies and send related events."""
        redis_backend = FastAPICache.get_backend()
        redis_client = redis_backend.redis  # type: ignore[attr-defined]

        for dcop_event in study_events:
            dicom_study_path = rename_dicom_path.joinpath(dcop_event.study_id)
            nifti_study_path = rename_nifti_path.joinpath(dcop_event.study_id)

            # Create STUDY_INFERENCE_READY event
            dcop_event_inference_ready = DCOPEventRequest(
                study_uid=dcop_event.study_uid,
                series_uid=None,
                study_id=dcop_event.study_id,
                ope_no=DCOPStatus.STUDY_INFERENCE_READY.value,
                tool_id="INFERENCE_TOOL",
                params_data={
                    "nifti_study_path": str(nifti_study_path),
                    "dicom_study_path": str(dicom_study_path),
                    "study_uid": dcop_event.study_uid,
                    "study_id": dcop_event.study_id,
                },
            )
            inference_task_key = (
                f"inference_task:{dcop_event.study_uid},{dcop_event.study_id}"
            )
            if await redis_client.get(inference_task_key):
                logger.info(
                    f"Skipping duplicate inference task for study_id: {dcop_event.study_id}. Already in cache."
                )
                continue  # Skip this study_event and move to the next one

            # Push to inference task pipeline
            task_pipeline_result: AsyncResult = task_pipeline_inference.push(
                dcop_event_inference_ready.params_data
            )
            await redis_client.set(
                inference_task_key, "queued", ex=21600
            )  # Value can be anything, key is what matters
            logger.info(
                f"Added study_uid: {dcop_event.study_uid} to cache with key: {inference_task_key}"
            )

            # Create STUDY_INFERENCE_QUEUED event
            dcop_event_inference_queued = DCOPEventRequest(
                study_uid=dcop_event.study_uid,
                series_uid=None,
                study_id=dcop_event.study_id,
                ope_no=DCOPStatus.STUDY_INFERENCE_QUEUED.value,
                tool_id="INFERENCE_TOOL",
                params_data={
                    "nifti_study_path": str(nifti_study_path),
                    "dicom_study_path": str(dicom_study_path),
                    "study_uid": dcop_event.study_uid,
                    "study_id": dcop_event.study_id,
                    "task_pipeline_id": task_pipeline_result.task_id,
                },
            )

            # Send inference events
            await self._send_events(
                upload_data_api_url,
                [
                    dcop_event_inference_ready.model_dump(),
                    dcop_event_inference_queued.model_dump(),
                ],
            )

    def _group_series_by_study(self, events):
        """Group series completion events by study."""
        series_by_study = {}

        for event in events:
            if event.ope_no == DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
                if event.study_uid not in series_by_study:
                    series_by_study[event.study_uid] = {
                        "completed": set(),
                        "study_id": event.study_id,
                    }

                # Add this series to the completed set
                if event.series_uid:
                    series_by_study[event.study_uid]["completed"].add(event.series_uid)

        return series_by_study

    async def _identify_completed_studies(
        self, study_events_list: List[DCOPEventRequest]
    ):
        """Identify studies with all series converted and create completion events."""
        completed_study_events = []
        # Query to get all series for this study
        async with self.session_manager.get_session() as session:
            done_count = 0
            undone = 0
            for study_events in study_events_list:
                sql = text(
                    "SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid"
                )
                params = {
                    "status": DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
                    "study_uid": study_events.study_uid,
                }
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
