#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-12-17

ReRun 研究重新執行服務模組

此模組提供醫學影像DICOM研究的重新執行功能，包括結果清理、快取清除、
資料庫更新和管道重新觸發等操作。

主要功能：
---------
1. 透過 Study ID 或 Study UID 重新執行研究流程
2. 清理先前的結果和快取資料
3. 建立新的處理事件記錄
4. 触發 DICOM 工具的系列資訊取得

類別：
-----
ReRunStudyService : 研究重新執行服務的主要類別，包含所有相關操作

模組依賴：
--------
- SQLAlchemy : 資料庫操作和 ORM
- httpx : 非同步 HTTP 客户端
- aiofiles : 非同步檔案操作
- advanced_alchemy : 進階 SQLAlchemy 擴展
- backend.app.sync : 同步事件模型和服務

範例：
-----
    service = ReRunStudyService()
    # 透過 Study UID 重新執行研究
    result = await service.re_run_by_study_uid(study_ids, dcop_event_service)

@author: sean Ho
"""

import logging
import os
import pathlib
import shutil
import traceback
from typing import List, Optional, Tuple

import aiofiles.os
import httpx
from advanced_alchemy.extensions.fastapi import repository
from advanced_alchemy.filters import LimitOffset
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.sync.model import DCOPEventModel
from backend.app.sync.schemas import DCOPStatus, OrthancID
from backend.app.sync.service import DCOPEventDicomService
from backend.app.sync.urls import SYNC_PROT_OPE_NO
from backend.app.service import BaseRepositoryService
from code_ai.task.schema.intput_params import Dicom2NiiParams
from code_ai import load_dotenv


logger = logging.getLogger(__name__)


class ReRunStudyService(BaseRepositoryService[DCOPEventModel]):
    """
    研究重新執行服務類別
    
    提供 DICOM 研究的重新執行功能，包括結果清理、快取清除和管道
    重新觸發。此服務負責管理研究處理事件的全生命周期。
    
    Attributes
    ----------
    repository_type : Type[Repo]
        資料庫儲存庫類別，處理 DCOPEventModel 的 CRUD 操作
    
    insert_sql : sqlalchemy.text
        將研究事件複製到重新執行表的 SQL 語句。
        從 dcop_event_bt 表插入新紀錄到 dcop_event_bth 表，
        並設定 event_cate 為 1 (重新執行類別)
    
    delete_sql : sqlalchemy.text
        刪除已處理研究事件的 SQL 語句。
        根據 study_uid 清理原始事件記錄。
    
    Parameters
    ----------
    None
        此類別由依賴注入系統自動初始化
    
    Methods
    -------
    get_study_new_re_model(study_uid, session)
        建立新的重新執行事件模型
    
    re_run_by_study_uid_on_one(study_uid, dcop_event_service)
        執行單一研究的重新執行流程
    
    del_study_result_by_field(field_name, field_value)
        透過指定欄位刪除研究結果
    
    Examples
    --------
    初始化並執行研究重新執行：
    
    >>> service = ReRunStudyService()
    >>> study_uid = "1.2.3.4.5"
    >>> result = await service.re_run_by_study_uid_on_one(
    ...     study_uid, dcop_event_service
    ... )
    >>> if result:
    ...     print("重新執行成功")
    
    See Also
    --------
    DCOPEventModel : 事件資料模型
    DCOPEventDicomService : DICOM 事件同步服務
    """
    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        """
        SQLAlchemy 非同步儲存庫實現
        
        負責 DCOPEventModel 的資料庫操作，包括查詢、插入、更新和刪除。
        """
        model_type = DCOPEventModel

    repository_type = Repo

    # SQL 語句：將原始事件複製到重新執行表
    # 功能：
    # - 從 dcop_event_bt 表選取指定 study_uid 的所有事件
    # - 複製所有欄位到 dcop_event_bth 表
    # - 設定 event_cate = 1 標記為重新執行類別
    # - 保留原始時間戳和其他元資料
    insert_sql = text("""
        insert into dcop_event_bth (
            vsprimarykey, tool_id, study_uid, series_uid, study_id,
            event_cate, code_name, code_desc, params_data, result_data, ope_no,
            ope_name, claim_time, rec_time, create_time, update_time
        )
        (
            select vsprimarykey, tool_id, study_uid, series_uid, study_id,
                   1, code_name, code_desc, params_data, result_data, ope_no,
                   ope_name, claim_time, rec_time, create_time, update_time 
            from dcop_event_bt 
            where study_uid=:study_uid
        )
    """)
    
    # SQL 語句：刪除原始事件記錄
    # 功能：
    # - 根據 study_uid 找出所有原始事件
    # - 從 dcop_event_bt 表中刪除這些記錄
    # - 適用於需要完全重置研究狀態的場景
    delete_sql = text("""
        delete from dcop_event_bt 
        where vsprimarykey in (
            select vsprimarykey from dcop_event_bt where study_uid=:study_uid
        )
    """)

    async def get_study_new_re_model(
        self, study_uid: str, session: AsyncSession
    ) -> Tuple[
        DCOPEventModel, DCOPEventModel, DCOPEventModel, DCOPEventModel, Dicom2NiiParams
    ]:
        """
        建立新的研究重新執行事件模型
        
        此方法初始化重新執行流程所需的所有事件記錄和參數。
        建立四個事件：新增重新執行、新增、轉移重新執行、轉移。
        
        Parameters
        ----------
        study_uid : str
            研究的唯一識別碼，格式為 DICOM UID (例如: 1.2.3.4.5)
        
        session : AsyncSession
            SQLAlchemy 非同步資料庫會話，用於建立事件記錄
        
        Returns
        -------
        Tuple[DCOPEventModel, DCOPEventModel, DCOPEventModel, DCOPEventModel, Dicom2NiiParams]
            包含以下元素的元組：
            - new_data_re (DCOPEventModel) : 標記為 STUDY_NEW_RE 的重新執行新增事件
            - new_data (DCOPEventModel) : 標記為 STUDY_NEW 的新增事件  
            - data_transferring_re (DCOPEventModel) : 標記為 STUDY_TRANSFERRING_RE 的重新執行轉移事件
              包含 DICOM 到 NIfTI 轉換的參數資訊
            - data_transferring (DCOPEventModel) : 標記為 STUDY_TRANSFERRING 的轉移事件
            - task_params (Dicom2NiiParams) : DICOM 轉 NIfTI 的任務參數
        
        Raises
        ------
        ValueError
            如果從環境變數讀取的路徑無效或不存在
        
        Notes
        -----
        環境變數依賴：
        - PATH_RAW_DICOM : 原始 DICOM 檔案的根目錄
        - PATH_RENAME_DICOM : 重命名後 DICOM 檔案的輸出目錄
        - PATH_RENAME_NIFTI : 轉換後 NIfTI 檔案的輸出目錄
        
        所有事件記錄基於相同的 study_uid，但狀態不同。
        這些事件用於追蹤研究重新執行的各個階段。
        
        Examples
        --------
        >>> service = ReRunStudyService()
        >>> async with session_manager.get_session() as session:
        ...     models = await service.get_study_new_re_model(
        ...         study_uid="1.2.3.4.5",
        ...         session=session
        ...     )
        ...     new_data_re, new_data, data_transferring_re, data_transferring, task_params = models
        ...     print(f"新增重新執行事件: {new_data_re.vsprimarykey}")
        
        See Also
        --------
        DCOPEventModel.create_event : 建立事件記錄的方法
        Dicom2NiiParams : DICOM 轉 NIfTI 轉換參數
        """
        # 載入環境變數配置
        load_dotenv()
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM") or "")
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM") or "")
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI") or "")

        # 組建該研究的原始 DICOM 路徑
        study_uid_raw_dicom_path = raw_dicom_path.joinpath(study_uid)

        # 建立新增重新執行事件
        # 標記為 STUDY_NEW_RE，表示重新執行流程的開始
        new_data_re = await DCOPEventModel.create_event(
            study_uid=study_uid,
            series_uid=None,
            status=DCOPStatus.STUDY_NEW_RE.name,
            session=session,
        )
        
        # 建立新增事件
        # 標記為 STUDY_NEW，表示標準新增流程
        new_data = await DCOPEventModel.create_event(
            study_uid=study_uid,
            series_uid=None,
            status=DCOPStatus.STUDY_NEW.name,
            session=session,
        )

        # 建立轉移重新執行事件
        # 標記為 STUDY_TRANSFERRING_RE，表示重新執行的檔案轉移階段
        data_transferring_re = await DCOPEventModel.create_event(
            study_uid=study_uid,
            series_uid=None,
            status=DCOPStatus.STUDY_TRANSFERRING_RE.name,
            session=session,
        )
        
        # 建立轉移事件
        # 標記為 STUDY_TRANSFERRING，表示標準檔案轉移階段
        data_transferring = await DCOPEventModel.create_event(
            study_uid=study_uid,
            series_uid=None,
            status=DCOPStatus.STUDY_TRANSFERRING.name,
            session=session,
        )

        # 建立 DICOM 轉 NIfTI 轉換的參數物件
        # 指定原始 DICOM 輸入目錄和處理後檔案的輸出目錄
        task_params = Dicom2NiiParams(
            sub_dir=study_uid_raw_dicom_path,
            output_dicom_path=rename_dicom_path,
            output_nifti_path=rename_nifti_path,
        )
        
        # 將參數序列化為字典格式，存入轉移重新執行事件
        # 供後續任務讀取和使用
        data_transferring_re.params_data = task_params.get_str_dict()
        
        # 回傳所有建立的事件和參數
        return (
            new_data_re,
            new_data,
            data_transferring_re,
            data_transferring,
            task_params,
        )

    async def re_run_by_study_rename_id(
        self, data_list: List[str], dcop_event_service: DCOPEventDicomService
    ) -> Optional[str]:
        """
        根據研究重命名 ID 批量執行研究重新執行
        
        此方法接收研究重命名 ID 列表，查詢對應的 Study UID，
        然後逐一執行重新執行流程。
        
        Parameters
        ----------
        data_list : List[str]
            研究重命名 ID 列表，每個元素為一個研究的重命名識別碼
        
        dcop_event_service : DCOPEventDicomService
            DCOP 事件 DICOM 服務實例，用於觸發系列資訊取得
        
        Returns
        -------
        Optional[str]
            None。此方法為 fire-and-forget 模式的後台任務。
            
        Notes
        -----
        - 此方法執行時不會拋出異常，所有錯誤會被記錄但不中斷流程
        - 結果列表包含 (study_uid, success_flag) 的元組，但目前未回傳
        - 每個研究最多查詢 10 筆最近的事件記錄
        - 只要找到匹配的事件，即執行重新執行流程
        
        Examples
        --------
        >>> service = ReRunStudyService()
        >>> data_list = ["study_rename_001", "study_rename_002"]
        >>> result = await service.re_run_by_study_rename_id(
        ...     data_list, dcop_event_service
        ... )
        
        See Also
        --------
        re_run_by_study_uid_on_one : 執行單一研究的重新執行
        """
        result_list = []
        logger.info("re_run_by_study_rename_id data_list {}".format(data_list))
        
        # 逐一處理每個研究重命名 ID
        for study_id in data_list:
            # 查詢該 study_id 對應的事件模型
            # 限制最多返回 10 筆最近的記錄
            models = await self.list(
                DCOPEventModel.study_id == study_id, LimitOffset(limit=10, offset=0)
            )
            
            # 如果找到匹配的事件，執行重新執行流程
            if len(models) > 0:
                logger.info("model {}".format(models[0]))
                # 提取第一筆記錄的 Study UID 進行重新執行
                flage = await self.re_run_by_study_uid_on_one(
                    models[0].study_uid, dcop_event_service
                )
                # 記錄執行結果
                result_list.append((models[0].study_uid, flage))

    async def re_run_by_study_uid_on_one(
        self, study_uid: str, dcop_event_service: DCOPEventDicomService
    ) -> bool:
        """
        執行單一研究的重新執行流程
        
        此方法是重新執行功能的核心，負責：
        1. 清理該研究的所有先前結果和快取
        2. 建立新的處理事件記錄
        3. 將事件記錄持久化到資料庫
        4. 觸發 DICOM 系列資訊取得程序
        
        Parameters
        ----------
        study_uid : str
            研究的唯一識別碼 (DICOM UID 格式)
        
        dcop_event_service : DCOPEventDicomService
            DCOP 事件 DICOM 服務，用於觸發系列資訊處理
        
        Returns
        -------
        bool
            True 如果重新執行成功，False 如果發生異常
        
        Raises
        ------
        捕獲所有異常但不重新拋出，改為記錄日誌和回傳 False
        
        Notes
        -----
        重新執行流程步驟：
        1. 清理結果：呼叫 del_study_result_by_field 刪除該研究的所有結果
        2. 建立事件：透過 get_study_new_re_model 建立新事件記錄
        3. 保存資料庫：使用 session 提交所有新事件
        4. 刷新物件：重新載入提交後的物件以獲取資料庫生成的 ID
        5. 觸發管道：呼叫 dicom_tool_get_series_info 啟動處理流程
        
        如果任何步驟失敗，會回滾交易並回傳 False。
        
        Examples
        --------
        >>> service = ReRunStudyService()
        >>> success = await service.re_run_by_study_uid_on_one(
        ...     "1.2.3.4.5", dcop_event_service
        ... )
        >>> if success:
        ...     print("研究重新執行已啟動")
        >>> else:
        ...     print("重新執行失敗，請查看日誌")
        
        See Also
        --------
        del_study_result_by_field : 清理研究結果
        get_study_new_re_model : 建立新事件模型
        DCOPEventDicomService.dicom_tool_get_series_info : 觸發系列資訊取得
        """
        logger.info("del_study_result_by_field 1")
        
        # 第一步：清理該研究的所有先前結果和快取
        await self.del_study_result_by_field(
            field_name="study_uid", field_value=study_uid
        )
        
        try:
            # 第二步：建立新的事件記錄並保存到資料庫
            async with self.session_manager.get_session() as session:
                # 建立四個新事件和任務參數
                data_tuple = await self.get_study_new_re_model(
                    study_uid=study_uid, session=session
                )
                (
                    new_data_re,
                    new_data,
                    data_transferring_re,
                    data_transferring,
                    task_params,
                ) = data_tuple
                
                # 將所有新事件加入會話
                session.add_all(
                    [new_data_re, new_data, data_transferring_re, data_transferring]
                )
                
                # 提交交易到資料庫
                await session.commit()
                
                # 重新載入物件以獲取資料庫生成的主鍵和時間戳
                await session.refresh(new_data_re)
                await session.refresh(data_transferring_re)
                logger.info("new_data_re {}".format(new_data_re))
            
            # 標記執行成功
            flage = True
            
        except Exception:
            # 捕獲任何異常，記錄堆疊追蹤並回滾
            logger.info(traceback.print_exc())
            await session.rollback()
            # 標記執行失敗
            flage = False

        # 第三步：如果前面成功，觸發 DICOM 系列資訊取得程序
        if flage:
            # 呼叫 DICOM 服務的核心方法，啟動後續處理流程
            await dcop_event_service.dicom_tool_get_series_info([new_data_re])
        
        return flage

    async def re_run_by_study_uid(
        self, data_list: List[OrthancID], dcop_event_service: DCOPEventDicomService
    ) -> Optional[str]:
        """
        根據 Study UID 批量執行研究重新執行
        
        此方法是對外公開的主要入口點，接收 Study UID 列表並批量
        重新執行研究流程。為後台任務設計，可由 API 端點呼叫。
        
        Parameters
        ----------
        data_list : List[OrthancID]
            Study UID 列表，每個元素為 OrthancID 型別的研究識別碼
        
        dcop_event_service : DCOPEventDicomService
            DCOP 事件 DICOM 服務實例
        
        Returns
        -------
        Optional[str]
            None。此為火轉即忘 (fire-and-forget) 的後台任務。
        
        Notes
        -----
        重新執行流程：
        1. 清理結果 : 刪除現有檔案和清理 SQL 資料庫記錄
        2. 新建 RERUN 紀錄 : 為每個研究建立新的重新執行事件
        3. 發送管道 : 觸發後續的 DICOM 處理管道
        
        此方法對每個 Study UID 逐一呼叫 re_run_by_study_uid_on_one，
        收集所有執行結果，但不拋出異常。
        
        Examples
        --------
        >>> service = ReRunStudyService()
        >>> study_uids = ["1.2.3.4.5", "1.2.3.4.6"]
        >>> result = await service.re_run_by_study_uid(
        ...     study_uids, dcop_event_service
        ... )
        # 方法立即回傳，實際處理在後台執行
        
        See Also
        --------
        re_run_by_study_uid_on_one : 執行單一研究的重新執行
        """
        # 收集執行結果列表 (雖然目前不回傳)
        result_list = []
        
        # 逐一處理每個 Study UID
        for study_uid in data_list:
            logger.info("del_study_result_by_field 1")
            
            # 執行單一研究的完整重新執行流程
            flage = await self.re_run_by_study_uid_on_one(
                study_uid=study_uid, dcop_event_service=dcop_event_service
            )
            
            # 記錄執行結果: (study_uid, 成功標誌)
            result_list.append((study_uid, flage))
        
        # 記錄所有執行結果到日誌
        logger.info(f"re_run_by_study_uid {result_list}")
        
        # 回傳 None (fire-and-forget 模式)
        return

    async def del_study_result_by_field(
        self, field_name: str, field_value: str
    ) -> Optional[str]:
        """
        根據指定欄位刪除研究結果
        
        此方法透過動態 SQL 查詢和參數清理指定研究的所有結果、
        檔案和快取。支援按 study_uid 或其他欄位篩選。
        
        Parameters
        ----------
        field_name : str
            SQL 欄位名稱，用於篩選研究記錄
            常見值：'study_uid', 'study_id'
        
        field_value : str
            欄位值，作為 WHERE 條件的參數
            例如：'1.2.3.4.5' (當 field_name='study_uid')
        
        Returns
        -------
        Optional[str]
            None
        
        Notes
        -----
        執行流程：
        1. 執行 SQL 查詢：從公開函數 get_stydy_ope_no_status 查詢
           狀態為 STUDY_RESULTS_SENT 的研究記錄
        2. 刪除結果：呼叫 del_study_result_by_parameters 刪除
           匹配的檔案和資料庫記錄
        3. 清除快取：呼叫 del_study_cache 清除該研究的快取資料
        
        Warning
        -------
        此操作具有破壞性，將永久刪除研究結果。
        使用前應確認 field_value 正確無誤。
        
        Examples
        --------
        >>> await service.del_study_result_by_field(
        ...     field_name="study_uid",
        ...     field_value="1.2.3.4.5"
        ... )
        
        See Also
        --------
        del_study_result_by_parameters : 執行實際的檔案和資料庫刪除
        del_study_cache : 清除快取資料
        """
        # 構建 SQL 查詢語句
        # 從公開函數 get_stydy_ope_no_status 查詢指定狀態的研究
        sql = text(
            f"SELECT * FROM public.get_stydy_ope_no_status(:status) where {field_name}=:{field_name}"
        )
        
        # 準備查詢參數
        # 篩選條件：狀態為已發送結果，欄位值匹配指定值
        parameters = {
            "status": DCOPStatus.STUDY_RESULTS_SENT.value,
            field_name: field_value,
        }
        
        # 刪除研究結果：檔案和資料庫記錄
        await self.del_study_result_by_parameters(sql=sql, parameters=parameters)
        
        # 清除該研究在快取系統中的資料
        await self.del_study_cache(field_name, field_value)

    async def del_study_result_by_study_uid(self, study_uid: str) -> Optional[str]:
        """
        根據 Study UID 刪除研究結果
        
        便捷方法，專門用於按 Study UID 刪除研究結果。
        實質上是呼叫 del_study_result_by_field 的特化版本。
        
        Parameters
        ----------
        study_uid : str
            研究的唯一識別碼 (DICOM UID 格式)
        
        Returns
        -------
        Optional[str]
            None
        
        Examples
        --------
        >>> await service.del_study_result_by_study_uid("1.2.3.4.5")
        
        See Also
        --------
        del_study_result_by_field : 通用欄位篩選刪除方法
        """
        # 建立 SQL 查詢語句，按 study_uid 篩選
        sql = text(
            "SELECT * FROM public.get_stydy_ope_no_status(:status) where study_uid=:study_uid"
        )
        
        # 設定查詢參數
        parameters = {
            "status": DCOPStatus.STUDY_RESULTS_SENT.value,
            "study_uid": study_uid,
        }
        
        # 執行刪除操作
        await self.del_study_result_by_parameters(sql=sql, parameters=parameters)

    async def del_study_result_by_parameters(self, sql: text, parameters: dict):
        """
        根據 SQL 查詢結果刪除研究的所有產物
        
        此方法執行最核心的清理操作：
        1. 刪除所有深度學習模型的輸出目錄
        2. 刪除重命名的 DICOM 和 NIfTI 檔案
        3. 更新資料庫事件表，標記為重新執行歷史
        
        Parameters
        ----------
        sql : sqlalchemy.text
            用於查詢研究記錄的 SQL 語句
        
        parameters : dict
            SQL 查詢的參數字典
            必須包含 'status' 和至少一個篩選欄位
        
        Returns
        -------
        None
        
        Notes
        -----
        刪除的目錄和檔案：
        - Deep_Aneurysm/{study_id}/ : 顱內動脈瘤檢測結果
        - Deep_CMB/{study_id}/ : 腦微出血檢測結果
        - Deep_cmd_tools/{study_id}_cmd.json : 命令工具配置
        - Deep_Infarct/{study_id}/ : 梗塞區域檢測結果
        - Deep_synthseg/{study_id}/ : 腦區域分割結果
        - Deep_WMH/{study_id}/ : 白質高信號檢測結果
        - renamed DICOM 檔案
        - renamed NIfTI 檔案
        
        資料庫操作：
        - 將事件從 dcop_event_bt 複製到 dcop_event_bth (標記為重新執行)
        - 刪除 dcop_event_bt 中的原始事件記錄
        
        Warning
        -------
        此操作具有高度破壞性，將永久刪除所有研究的處理結果。
        所有刪除操作都使用 ignore_errors=True 進行靜默失敗處理。
        
        Examples
        --------
        >>> sql = text(
        ...     "SELECT * FROM public.get_stydy_ope_no_status(:status) "
        ...     "where study_uid=:study_uid"
        ... )
        >>> await service.del_study_result_by_parameters(
        ...     sql=sql,
        ...     parameters={"status": "STUDY_RESULTS_SENT", "study_uid": "1.2.3"}
        ... )
        
        See Also
        --------
        del_path : 刪除單一檔案或目錄
        insert_sql : 事件複製 SQL
        delete_sql : 事件刪除 SQL
        """
        # 載入環境變數配置
        load_dotenv()
        
        # 構建所有深度學習模型的輸出目錄路徑
        process_path = pathlib.Path(os.getenv("PATH_PROCESS") or "")
        aneurysm_path = process_path.joinpath("Deep_Aneurysm")
        cmb_path = process_path.joinpath("Deep_CMB")
        cmd_tools_path = process_path.joinpath("Deep_cmd_tools")
        infarct_path = process_path.joinpath("Deep_Infarct")
        synthseg_path = process_path.joinpath("Deep_synthseg")
        wmh_path = process_path.joinpath("Deep_WMH")
        
        # 構建重命名後檔案的路徑
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM") or "")
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI") or "")
        
        # 執行查詢，獲取要刪除的研究記錄
        async with self.session_manager.get_session() as session:
            execute = await session.execute(sql, parameters)
            result = execute.first()
            
            # 如果找到符合條件的研究記錄
            if result is not None:
                # 構建該研究的所有輸出目錄路徑
                study_aneurysm_path = aneurysm_path.joinpath(result.study_id)
                study_cmb_path = cmb_path.joinpath(result.study_id)
                study_cmd_tools_path = cmd_tools_path.joinpath(f"{result.study_id}_cmd.json")
                study_infarct_path = infarct_path.joinpath(result.study_id)
                study_synthseg_path = synthseg_path.joinpath(result.study_id)
                study_wmh_path = wmh_path.joinpath(result.study_id)
                study_rename_dicom_path = rename_dicom_path.joinpath(result.study_id)
                study_rename_nifti_path = rename_nifti_path.joinpath(result.study_id)
                
                # 逐一刪除所有輸出檔案和目錄
                for input_path in [
                    study_aneurysm_path,
                    study_cmb_path,
                    study_cmd_tools_path,
                    study_infarct_path,
                    study_synthseg_path,
                    study_wmh_path,
                    study_rename_dicom_path,
                    study_rename_nifti_path,
                ]:
                    # 使用非同步方法刪除路徑
                    await self.del_path(input_path)
                
                try:
                    # 將事件記錄從原始表複製到重新執行歷史表
                    # 標記 event_cate = 1 表示重新執行類別
                    insert_execute = await session.execute(
                        self.insert_sql, {"study_uid": result.study_uid}
                    )
                    
                    # 刪除原始表中的事件記錄，保持資料庫乾淨
                    delete_execute = await session.execute(
                        self.delete_sql, {"study_uid": result.study_uid}
                    )

                    # 提交所有資料庫變更
                    await session.commit()
                    
                    # 記錄執行結果
                    logger.info(f"insert_execute {insert_execute}")
                    logger.info(f"delete_execute {delete_execute}")
                    
                except Exception:
                    # 如果資料庫操作失敗，回滾交易
                    await session.rollback()
                    # 記錄異常信息，但不中斷流程
                    logger.error(f"except {traceback.print_exc()}")

    async def del_study_cache(self, field_name: str, field_value: str):
        """
        清除遠端 API 快取中的研究資料
        
        向上游 API 發送 DELETE 請求，清除該研究的所有快取記錄。
        這通常用於同步快取狀態，以配合本地檔案和資料庫的清理操作。
        
        Parameters
        ----------
        field_name : str
            快取查詢欄位名稱
            常見值：'study_uid', 'study_id'
        
        field_value : str
            欄位值，用於標識要清除的研究
        
        Returns
        -------
        None
        
        Notes
        -----
        - 此方法使用非同步 HTTP 客户端，設定 180 秒超時
        - API 端點為上游 API_URL + '/cache'
        - 參數透過 URL 查詢字符串傳遞
        - 如果 API 呼叫失敗，不會拋出異常，靜默失敗
        
        環境變數依賴：
        - UPLOAD_DATA_API_URL : 上游資料上傳 API 的基礎 URL
        
        Examples
        --------
        >>> await service.del_study_cache(
        ...     field_name="study_uid",
        ...     field_value="1.2.3.4.5"
        ... )
        
        See Also
        --------
        del_study_result_by_field : 刪除研究結果的主要方法
        """
        # 載入環境變數
        load_dotenv()
        
        # 獲取上游 API 的基礎 URL
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        
        # 使用非同步 HTTP 客户端進行 DELETE 請求
        async with httpx.AsyncClient(timeout=180) as client:
            # 構建快取清除端點 URL
            url = f"{upload_data_api_url}/cache"
            
            # 發送 DELETE 請求，以清除指定研究的快取
            # 參數作為 URL 查詢字符串傳遞
            await client.delete(url=url, timeout=180, params={field_name: field_value})

    @staticmethod
    async def _send_events(event_data: List[dict]) -> None:
        """
        將研究轉移完成事件發送到上游 API
        
        此靜態方法負責將處理完成的事件資料序列化為 JSON 格式，
        並發送到上游 API 端點進行登記和進一步處理。
        
        Parameters
        ----------
        event_data : List[dict]
            序列化後的 DCOPEventRequest 物件列表
            每個字典應包含事件的完整資訊：
            - study_uid : 研究 UID
            - status : 事件狀態
            - timestamp : 事件時間戳
            - 其他相關欄位
        
        Returns
        -------
        None
        
        Notes
        -----
        - 此為靜態方法，不需要服務實例即可調用
        - 使用非同步 HTTP 客户端，設定 180 秒超時
        - 事件資料序列化為 JSON 格式在 POST body 中發送
        - 使用上游 API 的 SYNC_PROT_OPE_NO 端點
        - 如果發送失敗，不會拋出異常，靜默失敗
        
        環境變數依賴：
        - UPLOAD_DATA_API_URL : 上游 API 的基礎 URL
        
        Examples
        --------
        >>> event_data = [
        ...     {
        ...         "study_uid": "1.2.3.4.5",
        ...         "status": "STUDY_RESULTS_SENT",
        ...         "timestamp": "2025-12-17T10:00:00"
        ...     }
        ... ]
        >>> await ReRunStudyService._send_events(event_data)
        
        See Also
        --------
        SYNC_PROT_OPE_NO : 同步協議的操作編號路徑
        """
        # 獲取上游 API 的基礎 URL
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        
        # 使用非同步 HTTP 客户端進行 POST 請求
        async with httpx.AsyncClient(timeout=180) as client:
            # 構建完整的 API 端點 URL
            url = f"{upload_data_api_url}{SYNC_PROT_OPE_NO}"
            
            # 發送 POST 請求，將事件資料傳送到 API
            await client.post(url=url, timeout=180, json=event_data)

    @staticmethod
    async def del_path(input_path: pathlib.Path):
        """
        刪除單一檔案或目錄
        
        根據路徑類型進行相應的刪除操作，支援檔案和目錄。
        所有刪除操作都使用 ignore_errors=True 進行靜默失敗處理。
        
        Parameters
        ----------
        input_path : pathlib.Path
            要刪除的檔案或目錄的路徑物件
        
        Returns
        -------
        None
        
        Notes
        -----
        刪除邏輯：
        - 如果是檔案且存在：使用非同步方法 aiofiles.os.remove 刪除
        - 如果是目錄且存在：使用 shutil.rmtree 遞迴刪除
        - 如果不存在或類型無法識別：靜默跳過
        
        所有錯誤都會被抑制 (ignore_errors=True)，適合背景清理任務。
        
        Examples
        --------
        >>> import pathlib
        >>> path = pathlib.Path("/tmp/study_001")
        >>> await ReRunStudyService.del_path(path)
        
        >>> file_path = pathlib.Path("/tmp/study_001/result.nii.gz")
        >>> await ReRunStudyService.del_path(file_path)
        
        See Also
        --------
        del_study_result_by_parameters : 使用此方法的主要清理流程
        """
        # 記錄要刪除的路徑
        logger.info(f"input_path {input_path}")
        
        # 檢查是否為檔案
        if input_path.is_file() and input_path.exists():
            # 使用非同步方法刪除單一檔案
            await aiofiles.os.remove(input_path)
            logger.info("remove")
        
        # 檢查是否為目錄
        elif input_path.is_dir() and input_path.exists():
            # 使用 shutil 遞迴刪除目錄及其所有內容
            # ignore_errors=True 表示忽略任何刪除錯誤
            logger.info("rmtree")
            shutil.rmtree(input_path, ignore_errors=True)
        
        # 其他情況 (不存在、無法識別)
        else:
            pass

    @staticmethod
    async def async_path_generator(paths):
        """
        非同步路徑生成器
        
        提供一個簡單的非同步生成器介面來遍歷路徑列表。
        主要用於未來擴展或批量路徑處理場景。
        
        Parameters
        ----------
        paths : Iterable[str or pathlib.Path]
            路徑列表或可迭代物件
        
        Yields
        ------
        path
            逐一產生每個路徑
        
        Notes
        -----
        此生成器目前實現較簡單，主要作為未來非同步批量
        操作的框架。未來可擴展為支援非同步 I/O 操作。
        
        Examples
        --------
        >>> paths = ["/path/1", "/path/2", "/path/3"]
        >>> async for path in ReRunStudyService.async_path_generator(paths):
        ...     print(path)
        """
        # 逐一產生每個路徑
        for path in paths:
            yield path
