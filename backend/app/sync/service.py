import asyncio
import json
import logging
import os
import pathlib
import traceback
from collections import defaultdict
from contextlib import suppress
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
import re
import httpx
import pandas as pd
import pydicom
from advanced_alchemy.extensions.fastapi import repository
from advanced_alchemy.service import OffsetPagination
from funboost import AsyncResult
from pyorthanc import Study, Orthanc

# from fastapi import
from sqlalchemy import text, select, and_, bindparam, delete
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi_cache import FastAPICache

from code_ai.task.schema.intput_params import Dicom2NiiParams
from backend.app.service import BaseRepositoryService
from .model import DCOPEventModel, StudyPrevLinkModel
from .schemas import (
    DCOPStatus,
    DCOPEventRequest,
    DCOPEventNIFTITOOLRequest,
    StydySeriesOpeNoStatus,
    OpeNo,
    OrthancID,
)
from .urls import (
    SYNC_PROT_OPE_NO,
    SYNC_PROT_STUDY_NIFTI_TOOL,
    SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID,
    SYNC_PROT_STUDY_TRANSFER_COMPLETE,
)

logger = logging.getLogger(__name__)


class DCOPEventDicomService(BaseRepositoryService[DCOPEventModel]):
    """封裝 DICOM 同步/轉檔/推論工作流程的主要服務層。"""

    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        """提供給 BaseRepositoryService 使用的 async repository 類別。"""

        model_type = DCOPEventModel

    repository_type = Repo
    pattern_str = "({}),({}),({}),({}),({}|{})".format(
        DCOPStatus.SERIES_NEW.value,
        DCOPStatus.SERIES_TRANSFERRING.value,
        DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
        DCOPStatus.SERIES_CONVERTING.value,
        DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
        DCOPStatus.SERIES_CONVERSION_SKIP.value,
    )
    can_inference_pattern = re.compile(pattern_str)

    async def get_check_url_by_ope_no(self, ope_no: str) -> Optional[str]:
        """依據 ope_no 回傳應該觸發的 callback URL。"""
        from code_ai import load_dotenv

        load_dotenv()
        UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
        match ope_no:
            case DCOPStatus.STUDY_TRANSFER_COMPLETE.value:
                url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_TRANSFER_COMPLETE}"
            case DCOPStatus.STUDY_CONVERSION_COMPLETE.value:
                url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
            case DCOPStatus.SERIES_TRANSFER_COMPLETE.value:
                url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_TRANSFER_COMPLETE}"
            case DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
                url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
            case _:
                url = None
        return url

    async def post_ope_no_task(self, data: List[DCOPEventRequest]):
        """寫入事件並依 ope_no 分組執行回調，確保單一 URL 只收到相關任務。"""
        from code_ai import load_dotenv

        load_dotenv()
        # 按 URL 分組 dcop_event，確保每個 URL 只發送對應的 events
        check_url_events_map = {}  # {url: [dcop_event_list]}
        # async with AsyncSession(self.repository.session.bind) as session:
        async with self.session_manager.get_session() as session:
            for dcop_event in data:
                new_data_obj = await DCOPEventModel.create_event_ope_no(
                    tool_id=dcop_event.tool_id,
                    study_uid=dcop_event.study_uid,
                    series_uid=dcop_event.series_uid,
                    study_id=dcop_event.study_id,
                    ope_no=dcop_event.ope_no,
                    result_data=dcop_event.result_data,
                    params_data=dcop_event.params_data,
                    session=session,
                )
                session.add(new_data_obj)
                await session.commit()
                await session.refresh(new_data_obj)

                # new_data_obj = await self.create(data=new_data, auto_commit=True, auto_refresh=True)
                match new_data_obj.ope_no:
                    case DCOPStatus.SERIES_TRANSFER_COMPLETE.value:
                        url = await self.get_check_url_by_ope_no(new_data_obj.ope_no)
                    case DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
                        url = await self.get_check_url_by_ope_no(new_data_obj.ope_no)
                    case _:
                        url = None
                if url is not None:
                    # 將對應的 dcop_event 加入到該 URL 的事件列表中
                    if url not in check_url_events_map:
                        check_url_events_map[url] = []
                    check_url_events_map[url].append(dcop_event)
        async with httpx.AsyncClient(timeout=180) as client:
            for url, dcop_event_list in check_url_events_map.items():
                # 發送 POST 請求時傳入對應的 dcop_event_list，確保只處理指定的 study
                dcop_event_dump_list = [
                    dcop_event.model_dump() for dcop_event in dcop_event_list
                ]
                rep = await client.post(url, json=dcop_event_dump_list)
        return

    async def check_study_series_transfer_complete(
        self, data: Optional[List[DCOPEventRequest]] = None
    ):
        """
        Checks if all series under a study have completed transfer and initiates the conversion process.
            檢查 study 下的 series 是否都傳輸完成
            1. series 完成傳輸添加 SERIES_TRANSFER_COMPLETE  的記錄
            2. 所有series都到了SERIES_TRANSFER_COMPLETE， 添加 STUDY_TRANSFER_COMPLETE 的記錄
            3. 添加 STUDY_CONVERTING 的記錄，
            4. 發送管道任務  進行轉換
        Process flow:
        1. Mark series completion with SERIES_TRANSFER_COMPLETE record
        2. When all series reach SERIES_TRANSFER_COMPLETE, add STUDY_TRANSFER_COMPLETE record
        3. Add STUDY_CONVERTING record
        4. Send pipeline task for conversion

        Args:
            data: Optional list of DCOPEventRequest objects. If None, retrieves study status from database.

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
            for dcop_event_dump in dcop_event_dump_list:
                dcop_event_dump["params_data"]
        else:
            dcop_event_list = [
                DCOPEventRequest.model_validate(event, strict=False) for event in data
            ]
            dcop_event_dump_list = [
                dcop_event.model_dump() for dcop_event in dcop_event_list
            ]

        # Process eligible studies for conversion
        if dcop_event_list:
            await self._send_events(upload_data_api_url, dcop_event_dump_list)
            await self._initiate_conversion_process(
                upload_data_api_url,
                dcop_event_list,
                path_rename_dicom,
                path_rename_nifti,
            )

        return dcop_event_list

    async def schedule_new_studies(self, study_uids: List[str]) -> List[DCOPEventModel]:
        """將輸入轉換為唯一 study 清單並沿用既有建立流程。"""
        canonical_ids = list(dict.fromkeys(filter(None, study_uids)))
        if not canonical_ids:
            return []
        logger.info("schedule_new_studies count=%s", len(canonical_ids))
        return await self.add_study_new(data_list=canonical_ids)

    async def add_study_new(self, data_list):
        """初始化 study 任務，建立 STUDY_NEW / STUDY_TRANSFERRING 事件並備妥轉檔參數。"""
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv

        load_dotenv()
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))

        result_list = []
        async with self.session_manager.get_session() as session:
            try:
                for ids in data_list:
                    study_uid_raw_dicom_path = raw_dicom_path.joinpath(ids)
                    new_data = await DCOPEventModel.create_event(
                        study_uid=ids,
                        series_uid=None,
                        status=DCOPStatus.STUDY_NEW.name,
                        session=session,
                    )
                    session.add(new_data)
                    task_params = Dicom2NiiParams(
                        sub_dir=study_uid_raw_dicom_path,
                        output_dicom_path=rename_dicom_path,
                        output_nifti_path=rename_nifti_path,
                    )
                    data_transferring = await DCOPEventModel.create_event(
                        study_uid=ids,
                        series_uid=None,
                        status=DCOPStatus.STUDY_TRANSFERRING.name,
                        session=session,
                    )
                    data_transferring.params_data = task_params.get_str_dict()
                    session.add(data_transferring)
                    await session.commit()

                    # obj = await self.create_many(data=[new_data_obj,data_transferring],auto_commit=True)
                    result_list.append(new_data)
                    result_list.append(data_transferring)
            except Exception as e:
                await session.rollback()
                logger.error(f"Error in add_study_new: {e}")
                raise

        return result_list

    async def link_prev_study(
        self, study_uid: Optional[str], prev_study_uid: Optional[str]
    ) -> None:
        """建立 study 與前一個 study 的連結紀錄。"""

        validation_rules = {
            "missing_current": lambda cur, prev: not cur,
            "missing_prev": lambda cur, prev: not prev,
            "duplicated": lambda cur, prev: cur == prev,
        }
        for rule_name, predicate in validation_rules.items():
            if predicate(study_uid, prev_study_uid):
                logger.info(
                    "skip link_prev_study rule=%s study_uid=%s prev_study_uid=%s",
                    rule_name,
                    study_uid,
                    prev_study_uid,
                )
                return

        async with self.session_manager.get_session() as session:
            await session.execute(
                delete(StudyPrevLinkModel).where(
                    StudyPrevLinkModel.study_uid == study_uid
                )
            )
            session.add(
                StudyPrevLinkModel(study_uid=study_uid, prev_study_uid=prev_study_uid)
            )
            await session.commit()
        logger.info(
            "link_prev_study success study_uid=%s prev_study_uid=%s",
            study_uid,
            prev_study_uid,
        )

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

        async with self.session_manager.get_session() as session:
            study_uids = await self._get_recent_study_uids(session)
            if study_uids:
                sql = (
                    text(
                        "SELECT * FROM public.get_all_studies_status() "
                        "WHERE study_uid IN :study_uids"
                    ).bindparams(bindparam("study_uids", expanding=True))
                )
                results = await session.execute(sql, {"study_uids": tuple(study_uids)})
            else:
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

    async def _send_events(self, api_url: str, event_data) -> None:
        """
        Sends study transfer complete events to the API.

        Args:
            api_url: Base URL for the upload data API.
            event_data: List of serialized DCOPEventRequest objects.
        """
        event_data_list = list(filter(lambda x: x is not None, event_data))
        logger.info(f"_send_events {event_data_list}")
        async with httpx.AsyncClient(timeout=180) as client:
            url = f"{api_url}{SYNC_PROT_OPE_NO}"
            # event_data_json = json.dumps(event_data)
            await client.post(url=url, json=event_data_list)

    async def _get_recent_study_uids(
        self,
        session: AsyncSession,
        *,
        limit: Optional[int] = None,
        lookback_hours: Optional[int] = None,
    ) -> List[str]:
        """
        Retrieve recently updated study_uids to constrain heavy status queries.

        Args:
            session: Active AsyncSession.
            limit: Maximum number of study_uids to return.
            lookback_hours: Optional lookback window for update_time filtering.
        """

        default_limit = int(os.getenv("STUDY_STATUS_RECENT_LIMIT", "200"))
        limit = max(limit or default_limit, 1)
        env_lookback = os.getenv("STUDY_STATUS_LOOKBACK_HOURS")
        if lookback_hours is None and env_lookback:
            with suppress(ValueError):
                lookback_hours = int(env_lookback)

        threshold_dt = (
            datetime.utcnow() - timedelta(hours=lookback_hours)
            if lookback_hours
            else None
        )

        base_sql = """
            SELECT
                study_uid,
                MAX(update_time) AS last_update
            FROM dcop_event_bt
            WHERE study_uid IS NOT NULL
            GROUP BY study_uid
        """

        if threshold_dt is not None:
            sql = text(
                f"""
                SELECT sub.study_uid
                FROM ({base_sql}) AS sub
                WHERE sub.last_update >= :threshold
                ORDER BY sub.last_update DESC
                LIMIT :limit
                """
            )
            params = {"threshold": threshold_dt, "limit": limit}
        else:
            sql = text(
                f"""
                SELECT sub.study_uid
                FROM ({base_sql}) AS sub
                ORDER BY sub.last_update DESC
                LIMIT :limit
                """
            )
            params = {"limit": limit}

        results = await session.execute(sql, params)
        return [row.study_uid for row in results.all()]

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

        if not events:
            return

        url = f"{api_url}{SYNC_PROT_STUDY_NIFTI_TOOL}"
        payloads = []
        for event in events:
            study_id = event.study_id
            output_dicom_path = pathlib.Path(os.path.join(dicom_path, study_id))
            output_nifti_path = pathlib.Path(nifti_path)

            task_params = Dicom2NiiParams(
                sub_dir=None,
                output_dicom_path=output_dicom_path,
                output_nifti_path=output_nifti_path,
            )

            nifti_tool_request = DCOPEventNIFTITOOLRequest(
                ope_no=DCOPStatus.STUDY_CONVERTING.value,
                study_id=study_id,
                tool_id="NIFTI_TOOL",
                params_data=task_params.get_str_dict(),
                result_data=None,
            )
            payloads.append(nifti_tool_request.model_dump())

        async with httpx.AsyncClient(timeout=180) as client:
            await asyncio.gather(
                *[client.post(url=url, json=[payload]) for payload in payloads]
            )

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
        study_converting_events = list(
            filter(
                lambda dcop: dcop.ope_no == DCOPStatus.STUDY_CONVERTING.value,
                data,
            )
        )

        if study_converting_events:
            async with self.session_manager.get_session() as session:
                created_records = []
                study_uids_to_process: set[str] = set()
                for dcop in study_converting_events:
                    conf_query = (
                        select(DCOPEventModel)
                        .where(
                            and_(
                                DCOPEventModel.study_id == dcop.study_id,
                                DCOPEventModel.study_uid.isnot(None),
                                DCOPEventModel.tool_id == "DICOM_TOOL",
                                DCOPEventModel.ope_no
                                == DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
                            )
                        )
                        .limit(1)
                    )
                    execute = await session.execute(conf_query)
                    result = execute.first()
                    if not result:
                        logger.warning(
                            "找不到 study_id=%s 的 SERIES_TRANSFER_COMPLETE 記錄，略過 STUDY_CONVERTING",
                            dcop.study_id,
                        )
                        continue
                    dcop_event = result[0]
                    study_transfer_complete_data = await DCOPEventModel.create_event_ope_no(
                        tool_id=dcop.tool_id,
                        study_uid=dcop_event.study_uid,
                        series_uid=None,
                        study_id=dcop.study_id,
                        ope_no=dcop.ope_no,
                        result_data=dcop.result_data,
                        params_data=dcop.params_data,
                        session=session,
                    )
                    session.add(study_transfer_complete_data)
                    created_records.append(study_transfer_complete_data)
                    study_uids_to_process.add(dcop_event.study_uid)

                if created_records:
                    await session.commit()
                    for record in created_records:
                        await session.refresh(record)
                    for study_uid in study_uids_to_process:
                        await self.nifti_tool_get_series_info(study_uid, session)

        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        url = f"{upload_data_api_url}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
        async with httpx.AsyncClient(timeout=180) as client:
            await client.post(url=url)

    async def nifti_tool_get_series_info(self, study_uid: str, session: AsyncSession):
        """查出待轉檔的 series，並推送 funboost 工作到 NIFTI_TOOL。"""
        from code_ai.task.task_dicom2nii import dicom_2_nii_series
        from code_ai.task.schema.intput_params import Dicom2NiiSeriesParams
        from code_ai import load_dotenv

        load_dotenv()
        path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
        path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")
        # 直接查詢原始表，不使用分組函數，避免相同 series_uid 但不同 rename_dicom_path 的記錄被合併
        # 加入狀態檢查邏輯，確保只處理那些所有 ope_no 都小於目標狀態的記錄
        # 支援多種情況：
        # 1. 有 params_data->>'rename_dicom_path' 的新記錄（優先使用）
        # 2. 沒有 params_data->>'rename_dicom_path' 但 result_data 中有 rename_dicom_path 的記錄
        # 3. 都沒有的舊記錄（回退到原始邏輯，只按 series_uid 分組）
        target_ope_no = DCOPStatus.STUDY_CONVERTING.value
        # 使用參數化查詢，避免 SQL 注入並確保參數綁定正確
        sql = text("""
            WITH series_rename_status AS (
                SELECT
                    dcop_event_bt.study_uid,
                    dcop_event_bt.series_uid,
                    COALESCE(
                        dcop_event_bt.params_data->>'rename_dicom_path',
                        dcop_event_bt.result_data->>'rename_dicom_path'
                    ) as rename_dicom_path,
                    MAX(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
                    array_agg(DISTINCT dcop_event_bt.ope_no) as ope_no_array,
                    array_agg(dcop_event_bt.result_data ORDER BY dcop_event_bt.create_time DESC) as result_data_array,
                    array_agg(dcop_event_bt.params_data ORDER BY dcop_event_bt.create_time DESC) as params_data_array,
                    MAX(dcop_event_bt.create_time) as create_time,
                    MAX(dcop_event_bt.update_time) as update_time
                FROM dcop_event_bt
                WHERE dcop_event_bt.study_uid = :study_uid
                  AND dcop_event_bt.series_uid IS NOT NULL
                  AND dcop_event_bt.result_data IS NOT NULL
                GROUP BY 
                    dcop_event_bt.study_uid, 
                    dcop_event_bt.series_uid,
                    COALESCE(
                        dcop_event_bt.params_data->>'rename_dicom_path',
                        dcop_event_bt.result_data->>'rename_dicom_path'
                    )
            )
            SELECT DISTINCT ON (srs.series_uid, COALESCE(srs.rename_dicom_path, ''))
                srs.study_uid,
                srs.series_uid,
                srs.study_id,
                srs.rename_dicom_path,
                srs.ope_no_array as ope_no,
                srs.result_data_array[1] as result_data,
                srs.params_data_array[1] as params_data,
                srs.create_time,
                srs.update_time
            FROM series_rename_status as srs
            WHERE (:target_ope_no)::NUMERIC > ALL (srs.ope_no_array::NUMERIC[])
              AND EXISTS (
                  SELECT 1
                  FROM unnest(srs.result_data_array) AS pd
                  WHERE pd IS NOT NULL
              )
            ORDER BY srs.series_uid, COALESCE(srs.rename_dicom_path, ''), srs.create_time DESC
        """)
        results = await session.execute(
            sql, {"target_ope_no": target_ope_no, "study_uid": study_uid}
        )

        dcop_event_list = results.all()
        task_params_list = []
        dcop_model_list = []
        for dcop_event in dcop_event_list:
            # 使用 Row 物件的屬性訪問或索引訪問（兼容不同 SQLAlchemy 版本）
            try:
                # 嘗試屬性訪問（較新版本）
                if hasattr(dcop_event, "_mapping"):
                    # SQLAlchemy 2.0+ 使用 _mapping
                    rename_dicom_path = dcop_event._mapping.get("rename_dicom_path")
                    result_data = dcop_event._mapping.get("result_data")
                    study_uid = dcop_event._mapping.get("study_uid")
                    series_uid = dcop_event._mapping.get("series_uid")
                    study_id = dcop_event._mapping.get("study_id")
                elif hasattr(dcop_event, "rename_dicom_path"):
                    # 直接屬性訪問
                    rename_dicom_path = dcop_event.rename_dicom_path
                    result_data = dcop_event.result_data
                    study_uid = dcop_event.study_uid
                    series_uid = dcop_event.series_uid
                    study_id = dcop_event.study_id
                else:
                    # 索引訪問（備用方案）：study_uid, series_uid, study_id, rename_dicom_path, ope_no, result_data, params_data, create_time, update_time
                    study_uid = dcop_event[0]
                    series_uid = dcop_event[1]
                    study_id = dcop_event[2]
                    rename_dicom_path = dcop_event[3]
                    result_data = dcop_event[5]
            except (IndexError, AttributeError, KeyError) as e:
                logger.error("無法正確解析查詢結果：%s, row=%s", e, dcop_event)
                continue

            # 驗證必要欄位
            if not study_uid or not series_uid:
                logger.warning(
                    "缺少必要欄位，已略過：study_uid=%s, series_uid=%s",
                    study_uid,
                    series_uid,
                )
                continue

            # 處理 rename_dicom_path：優先使用查詢結果，如果沒有則從 result_data 中提取
            if not rename_dicom_path and result_data:
                if isinstance(result_data, dict):
                    rename_dicom_path = result_data.get("rename_dicom_path")
                elif (
                    isinstance(result_data, list)
                    and len(result_data) > 0
                    and isinstance(result_data[0], dict)
                ):
                    rename_dicom_path = result_data[0].get("rename_dicom_path")

            if not rename_dicom_path:
                logger.warning(
                    "找不到 rename_dicom_path，已略過 series_uid=%s", series_uid
                )
                continue

            # 處理 result_data：確保是 dict 格式
            if not result_data:
                logger.warning(
                    "result_data 為空，已略過 series_uid=%s, rename_dicom_path=%s",
                    series_uid,
                    rename_dicom_path,
                )
                continue

            # 標準化 result_data 為 dict 格式
            if isinstance(result_data, list) and len(result_data) > 0:
                result_data_dict = (
                    result_data[0] if isinstance(result_data[0], dict) else result_data
                )
            elif isinstance(result_data, dict):
                result_data_dict = result_data
            else:
                logger.warning(
                    "result_data 格式不正確，已略過 series_uid=%s, rename_dicom_path=%s",
                    series_uid,
                    rename_dicom_path,
                )
                continue

            # 確保 result_data 中包含 rename_dicom_path（用於後續查詢）
            if "rename_dicom_path" not in result_data_dict:
                result_data_dict["rename_dicom_path"] = rename_dicom_path

            output_nifti_path = pathlib.Path(path_rename_nifti)
            task_params = Dicom2NiiSeriesParams(
                sub_dir=None,
                study_uid=study_uid,
                series_uid=series_uid,
                output_dicom_path=rename_dicom_path,
                output_nifti_path=output_nifti_path,
            )
            # 為每個 rename_dicom_path 建立獨立的 SERIES_CONVERTING 記錄
            new_data_obj = await DCOPEventModel.create_event_ope_no(
                tool_id="NIFTI_TOOL",
                study_uid=study_uid,
                series_uid=series_uid,
                study_id=study_id,
                ope_no=DCOPStatus.SERIES_CONVERTING.value,
                result_data=result_data_dict,
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

    @staticmethod
    def get_orthanc_study_uid_series_uid(instance_path_str: str):
        """由單一 DICOM instance 解析 Orthanc 內的 study/series uid。"""
        instance_path = pathlib.Path(instance_path_str)
        UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
        # raw_dicom\ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30\10089413 GUO HSIOU HUA\21002010079 MRI Stroke Wall C C\MR 3D Ax SWAN\*.dcm
        with open(instance_path_str, mode="rb") as f:
            dicom_ds = pydicom.dcmread(f)

        client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300)
        study_uid = instance_path.parent.parent.parent.parent.name
        # (0020,000E)	Series Instance UID	1.2.840.113619.2.44.5554020.7707121.19025.1612063861.703
        series_sop_uid = dicom_ds[0x0020, 0x000E].value
        # series_description = " ".join(instance_path.parent.name.split(" ")[1:]).strip()
        study = Study(study_uid, client=client)
        series_filter = list(
            filter(lambda series: series.uid == series_sop_uid, study.series)
        )
        if series_filter:
            return str(study_uid), str(series_filter[0].id_)
        else:
            return None

    @staticmethod
    def get_orthanc_series_uid(study_uid: str, series_dir_set: set):
        """批次查詢 series 資訊並回傳 pandas DataFrame 以方便後續 mapping。"""
        UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
        client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300)
        series_sop_uid_list = []
        for series_dir in series_dir_set:
            series_path_list = list(series_dir.rglob("*.dcm"))
            instance_path_str = series_path_list[0]
            with open(instance_path_str, mode="rb") as f:
                dicom_ds = pydicom.dcmread(f)
            series_sop_uid = dicom_ds[0x0020, 0x000E].value
            series_sop_uid_list.append(series_sop_uid)
        study = Study(study_uid, client=client)
        series_dict_list = list(
            map(
                lambda x: {
                    "series_sop_uid": x.uid,
                    "uid": x.id_,
                    "description": x.description,
                },
                study.series,
            )
        )
        df = pd.DataFrame(series_sop_uid_list, columns=["file_series_sop_uid"])
        df1 = pd.DataFrame(series_dict_list)
        df2 = pd.merge(
            df, df1, left_on="file_series_sop_uid", right_on="series_sop_uid"
        )
        return df2

    async def dicom_tool_get_series_info(self, data: List[DCOPEventModel]):
        """在接獲 study 任務後掃描 series 並建立對應事件、推入轉檔任務列。"""
        from code_ai.task.task_dicom2nii import dicom_to_nii
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv

        load_dotenv()
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))

        for dcop_event in data:
            study_uid = dcop_event.study_uid
            logger.info(f"dicom_tool_get_series_info dcop_event {dcop_event}")
            study_uid_raw_dicom_path = raw_dicom_path.joinpath(study_uid)
            if study_uid_raw_dicom_path.exists():
                dcm_path_list = sorted(study_uid_raw_dicom_path.rglob("*.dcm"))
                series_dir_set = set([dcm_path.parent for dcm_path in dcm_path_list])
                df = self.get_orthanc_series_uid(study_uid, series_dir_set)
                series_uid_list = df["uid"].to_list()
                task_params = Dicom2NiiParams(
                    sub_dir=study_uid_raw_dicom_path,
                    output_dicom_path=rename_dicom_path,
                    output_nifti_path=rename_nifti_path,
                )
                flage = True
                async with self.session_manager.get_session() as session:
                    series_event_models = []
                    for series_uid in series_uid_list:
                        try:
                            series_new_data = await DCOPEventModel.create_event(
                                study_uid=study_uid,
                                series_uid=series_uid,
                                status=DCOPStatus.SERIES_NEW.name,
                                session=session,
                            )
                            series_transferring_data = await DCOPEventModel.create_event(
                                study_uid=study_uid,
                                series_uid=series_uid,
                                status=DCOPStatus.SERIES_TRANSFERRING.name,
                                session=session,
                            )
                            series_transferring_data.params_data = (
                                task_params.get_str_dict()
                            )
                            series_event_models.extend(
                                [series_new_data, series_transferring_data]
                            )
                        except Exception:
                            flage = False
                            logger.error(traceback.format_exc())
                            break
                    if flage and series_event_models:
                        session.add_all(series_event_models)
                        await session.commit()
                        for model in series_event_models:
                            await session.refresh(model)
                        logger.info(
                            "dicom_tool_get_series_info 新增 %s 筆 series 事件 (study=%s)",
                            len(series_event_models),
                            study_uid,
                        )
                    else:
                        await session.rollback()
                if flage:
                    dicom_to_nii.push(task_params.get_str_dict())
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
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
        logger.info(f"data {data}")
        # post_check_study_series_conversion_complete call check
        if data is None:
            # Query studies not yet at STUDY_CONVERSION_COMPLETE status
            completed_studies = await self.query_studies_pending_completion()
        else:
            completed_studies = set()
            for dcop_enent in data:
                result_set = await self.query_studies_pending_completion(
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
        study_events = await self.create_study_complete_events(
            completed_studies, raw_dicom_path, rename_dicom_path, rename_nifti_path
        )
        # Process events from the provided data list
        completed_study_events = await self.identify_completed_studies(study_events)
        # Process completed studies and queue them for inference
        if completed_study_events:
            study_events_filter = []
            for completed_study in completed_study_events:
                study_event = list(
                    filter(
                        lambda x: x.study_uid == completed_study.study_uid, study_events
                    )
                )
                study_events_filter.extend(study_event)
            study_events_filter = list(
                map(lambda x: x.model_dump(), study_events_filter)
            )
            await self._send_events(upload_data_api_url, study_events_filter)
            # Queue inference tasks for completed studies
            await self._queue_inference_tasks(
                completed_study_events,
                upload_data_api_url,
                rename_dicom_path,
                rename_nifti_path,
                task_pipeline_inference,
            )
        return None

    async def query_studies_pending_completion(self, study_uid: Optional[str] = None):
        """Query for studies that have not yet reached STUDY_CONVERSION_COMPLETE status.
        
        最小改動：將資料庫函數調用改為 CTE 查詢，在 GROUP BY 中加入 rename_dicom_path。
        保持所有其他邏輯不變，包括防止無限迴圈的條件。
        """
        async with self.session_manager.get_session() as session:
            if study_uid is None:
                completion_limit = int(
                    os.getenv(
                        "STUDY_COMPLETION_RECENT_LIMIT",
                        os.getenv("STUDY_STATUS_RECENT_LIMIT", "200"),
                    )
                )
                study_uids = await self._get_recent_study_uids(
                    session, limit=completion_limit
                )
                # 最小改動：將資料庫函數改為等效的 CTE，加入 rename_dicom_path 分組
                if study_uids:
                    sql = (
                        text("""
                            WITH series_ope_no_status AS (
                                SELECT
                                    dcop_event_bt.study_uid,
                                    dcop_event_bt.series_uid,
                                    MIN(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
                                    COALESCE(
                                        dcop_event_bt.params_data->>'rename_dicom_path',
                                        dcop_event_bt.result_data->>'rename_dicom_path'
                                    ) as rename_dicom_path,
                                    array_agg(DISTINCT dcop_event_bt.ope_no) as ope_no,
                                    array_agg(dcop_event_bt.result_data) as result_data,
                                    array_agg(dcop_event_bt.params_data) as params_data
                                FROM dcop_event_bt
                                WHERE dcop_event_bt.series_uid IS NOT NULL
                                  AND dcop_event_bt.study_uid IN :study_uids
                                GROUP BY 
                                    dcop_event_bt.study_uid,
                                    dcop_event_bt.series_uid,
                                    COALESCE(
                                        dcop_event_bt.params_data->>'rename_dicom_path',
                                        dcop_event_bt.result_data->>'rename_dicom_path'
                                    )
                            ),
                            max_study_ope AS (
                                SELECT deb.study_id, max(deb.ope_no::numeric) as ope_no 
                                FROM dcop_event_bt deb 
                                WHERE deb.study_uid IN :study_uids
                                GROUP BY study_id
                            )
                            SELECT 
                                sos.study_uid,
                                sos.series_uid,
                                sos.study_id,
                                sos.rename_dicom_path,
                                sos.ope_no,
                                sos.result_data,
                                sos.params_data
                            FROM series_ope_no_status as sos,
                                 max_study_ope as debb
                            WHERE sos.study_id = debb.study_id
                              AND debb.ope_no::NUMERIC <= ANY (sos.ope_no::NUMERIC[])
                              AND :status::NUMERIC > ALL (sos.ope_no::NUMERIC[])
                              AND EXISTS (
                                  SELECT 1
                                  FROM unnest(sos.result_data) AS pd
                                  WHERE pd IS NOT NULL
                              )
                        """).bindparams(bindparam("study_uids", expanding=True))
                    )
                    params = {
                        "status": DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
                        "study_uids": tuple(study_uids),
                    }
                else:
                    sql = text("""
                        WITH series_ope_no_status AS (
                            SELECT
                                dcop_event_bt.study_uid,
                                dcop_event_bt.series_uid,
                                MIN(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
                                COALESCE(
                                    dcop_event_bt.params_data->>'rename_dicom_path',
                                    dcop_event_bt.result_data->>'rename_dicom_path'
                                ) as rename_dicom_path,
                                array_agg(DISTINCT dcop_event_bt.ope_no) as ope_no,
                                array_agg(dcop_event_bt.result_data) as result_data,
                                array_agg(dcop_event_bt.params_data) as params_data
                            FROM dcop_event_bt
                            WHERE dcop_event_bt.series_uid IS NOT NULL
                            GROUP BY 
                                dcop_event_bt.study_uid,
                                dcop_event_bt.series_uid,
                                COALESCE(
                                    dcop_event_bt.params_data->>'rename_dicom_path',
                                    dcop_event_bt.result_data->>'rename_dicom_path'
                                )
                        ),
                        max_study_ope AS (
                            SELECT deb.study_id, max(deb.ope_no::numeric) as ope_no 
                            FROM dcop_event_bt deb 
                            GROUP BY study_id
                        )
                        SELECT 
                            sos.study_uid,
                            sos.series_uid,
                            sos.study_id,
                            sos.rename_dicom_path,
                            sos.ope_no,
                            sos.result_data,
                            sos.params_data
                        FROM series_ope_no_status as sos,
                             max_study_ope as debb
                        WHERE sos.study_id = debb.study_id
                          AND debb.ope_no::NUMERIC <= ANY (sos.ope_no::NUMERIC[])
                          AND :status::NUMERIC > ALL (sos.ope_no::NUMERIC[])
                          AND EXISTS (
                              SELECT 1
                              FROM unnest(sos.result_data) AS pd
                              WHERE pd IS NOT NULL
                          )
                    """)
                    params = {"status": DCOPStatus.STUDY_CONVERSION_COMPLETE.value}
            else:
                sql = text("""
                    WITH series_ope_no_status AS (
                        SELECT
                            dcop_event_bt.study_uid,
                            dcop_event_bt.series_uid,
                            MIN(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
                            COALESCE(
                                dcop_event_bt.params_data->>'rename_dicom_path',
                                dcop_event_bt.result_data->>'rename_dicom_path'
                            ) as rename_dicom_path,
                            array_agg(DISTINCT dcop_event_bt.ope_no) as ope_no,
                            array_agg(dcop_event_bt.result_data) as result_data,
                            array_agg(dcop_event_bt.params_data) as params_data
                        FROM dcop_event_bt
                        WHERE dcop_event_bt.study_uid = :study_uid
                          AND dcop_event_bt.series_uid IS NOT NULL
                        GROUP BY 
                            dcop_event_bt.study_uid,
                            dcop_event_bt.series_uid,
                            COALESCE(
                                dcop_event_bt.params_data->>'rename_dicom_path',
                                dcop_event_bt.result_data->>'rename_dicom_path'
                            )
                    ),
                    max_study_ope AS (
                        SELECT deb.study_id, max(deb.ope_no::numeric) as ope_no 
                        FROM dcop_event_bt deb 
                        WHERE deb.study_uid = :study_uid
                        GROUP BY study_id
                    )
                    SELECT 
                        sos.study_uid,
                        sos.series_uid,
                        sos.study_id,
                        sos.rename_dicom_path,
                        sos.ope_no,
                        sos.result_data,
                        sos.params_data
                    FROM series_ope_no_status as sos,
                         max_study_ope as debb
                    WHERE sos.study_uid = :study_uid
                      AND sos.study_id = debb.study_id
                      AND debb.ope_no::NUMERIC <= ANY (sos.ope_no::NUMERIC[])
                      AND :status::NUMERIC > ALL (sos.ope_no::NUMERIC[])
                      AND EXISTS (
                          SELECT 1
                          FROM unnest(sos.result_data) AS pd
                          WHERE pd IS NOT NULL
                      )
                """)
                params = {
                    "status": DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
                    "study_uid": study_uid,
                }
            execute = await session.execute(sql, params)
            results = execute.all()

        # 最小改動：使用複合鍵處理 DWI0/DWI1000 場景
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
            
            # 提取 rename_dicom_path（支援不同 SQLAlchemy 版本）
            if hasattr(result, '_mapping') and result._mapping:
                rename_path = result._mapping.get('rename_dicom_path')
            else:
                rename_path = getattr(result, 'rename_dicom_path', None)
            
            # 使用複合鍵：(series_uid, rename_dicom_path)，解決 DWI0/DWI1000 問題
            composite_key = (result.series_uid, rename_path) if rename_path else result.series_uid
            
            if match_result:
                can_inference_dict.update(
                    {composite_key: (result.study_uid, result.study_id)}
                )
            else:
                wait_inference_dict.update(
                    {composite_key: (result.study_uid, result.study_id)}
                )

        wait_inference_set = set(wait_inference_dict.values())
        can_inference_set = set(can_inference_dict.values())
        if wait_inference_dict:
            result_set = can_inference_set - wait_inference_set
        else:
            result_set = can_inference_set

        return result_set

    async def create_study_complete_events(
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
        redis_client = redis_backend.redis
        events_to_dispatch: List[DCOPEventRequest] = []

        for dcop_event in study_events:
            dicom_study_path = rename_dicom_path.joinpath(dcop_event.study_id)
            nifti_study_path = rename_nifti_path.joinpath(dcop_event.study_id)

            inference_task_key = (
                f"inference_task:{dcop_event.study_uid},{dcop_event.study_id}"
            )
            if await redis_client.get(inference_task_key):
                logger.info(
                    f"Skipping duplicate inference task for study_id: {dcop_event.study_id}. Already in cache."
                )
                continue  # Skip this study_event and move to the next one

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

            events_to_dispatch.extend(
                [dcop_event_inference_ready, dcop_event_inference_queued]
            )

        if events_to_dispatch:
            await self._send_events(
                upload_data_api_url,
                [event.model_dump() for event in events_to_dispatch],
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

    async def identify_completed_studies(
        self, study_events_list: List[DCOPEventRequest]
    ):
        """Identify studies with all series converted and create completion events.
        
        最小改動：將資料庫函數調用改為 CTE 查詢，加入 rename_dicom_path 分組。
        按 (series_uid, rename_dicom_path) 分組檢查完成狀態。
        """
        completed_study_events = []
        async with self.session_manager.get_session() as session:
            done_count = 0
            undone = 0
            for study_events in study_events_list:
                # 最小改動：將資料庫函數改為等效的 CTE，加入 rename_dicom_path 分組
                sql = text("""
                    WITH series_ope_no_status AS (
                        SELECT
                            dcop_event_bt.study_uid,
                            dcop_event_bt.series_uid,
                            MIN(DISTINCT dcop_event_bt.study_id)::varchar as study_id,
                            COALESCE(
                                dcop_event_bt.params_data->>'rename_dicom_path',
                                dcop_event_bt.result_data->>'rename_dicom_path'
                            ) as rename_dicom_path,
                            array_agg(DISTINCT dcop_event_bt.ope_no) as ope_no,
                            array_agg(dcop_event_bt.result_data) as result_data,
                            array_agg(dcop_event_bt.params_data) as params_data
                        FROM dcop_event_bt
                        WHERE dcop_event_bt.study_uid = :study_uid
                          AND dcop_event_bt.series_uid IS NOT NULL
                        GROUP BY 
                            dcop_event_bt.study_uid,
                            dcop_event_bt.series_uid,
                            COALESCE(
                                dcop_event_bt.params_data->>'rename_dicom_path',
                                dcop_event_bt.result_data->>'rename_dicom_path'
                            )
                    )
                    SELECT 
                        sons.study_uid,
                        sons.series_uid,
                        sons.study_id,
                        sons.rename_dicom_path,
                        sons.ope_no,
                        sons.result_data,
                        sons.params_data
                    FROM series_ope_no_status as sons
                    WHERE 
                        :status::NUMERIC > ALL (sons.ope_no::NUMERIC[])
                        AND EXISTS (
                            SELECT 1
                            FROM unnest(sons.result_data) AS pd
                            WHERE pd IS NOT NULL
                        )
                """)
                params = {
                    "status": DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
                    "study_uid": study_events.study_uid,
                }
                execute = await session.execute(sql, params)
                results = execute.all()
                
                # 最小改動：按 (series_uid, rename_dicom_path) 分組檢查
                # 每個 result 現在代表一個 (series_uid, rename_dicom_path) 組合
                for result in results:
                    if DCOPStatus.SERIES_CONVERSION_COMPLETE.value in result.ope_no:
                        done_count += 1
                    elif DCOPStatus.SERIES_CONVERSION_SKIP.value in result.ope_no:
                        done_count += 1
                    else:
                        undone += 1
                
                # 只有當所有 (series_uid, rename_dicom_path) 組合都完成時，才判定為完成
                if done_count == len(results) and len(results) > 0:
                    completed_study_events.append(results[0])
                
                # 重置計數器用於下一個 study
                done_count = 0
                undone = 0
                
        return completed_study_events

    async def get_stydy_series_ope_no_status(
        self, study_uid: OrthancID, ope_no: OpeNo, limit: int, offset: int
    ) -> OffsetPagination[StydySeriesOpeNoStatus]:
        """查詢指定 ope_no 下，每個 series 的狀態分佈。"""
        async with self.session_manager.get_session() as session:
            if study_uid is None:
                sql = text(
                    "SELECT * FROM public.get_stydy_series_ope_no_status(:status) LIMIT :limit OFFSET :offset"
                )
                params = {"status": ope_no, "limit": limit, "offset": offset}
                count_sql = text(
                    "SELECT COUNT(*) FROM public.get_stydy_series_ope_no_status(:status)"
                )
                count_params = {"status": ope_no}
            else:
                sql = text(
                    "SELECT * FROM public.get_stydy_series_ope_no_status(:status) WHERE study_uid = :study_uid LIMIT :limit OFFSET :offset"
                )
                params = {
                    "status": ope_no,
                    "study_uid": study_uid,
                    "limit": limit,
                    "offset": offset,
                }
                count_sql = text(
                    "SELECT COUNT(*) FROM public.get_stydy_series_ope_no_status(:status) WHERE study_uid = :study_uid"
                )
                count_params = {"status": ope_no, "study_uid": study_uid}

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
        self, study_uid: OrthancID, ope_no: OpeNo, limit: int, offset: int
    ) -> OffsetPagination[StydySeriesOpeNoStatus]:
        """查詢 study 維度的 ope_no 狀態，支援分頁。"""
        async with self.session_manager.get_session() as session:
            if study_uid is None:
                sql = text(
                    "SELECT * FROM public.get_stydy_ope_no_status(:status) LIMIT :limit OFFSET :offset"
                )
                params = {"status": ope_no, "limit": limit, "offset": offset}
                count_sql = text(
                    "SELECT COUNT(*) FROM public.get_stydy_ope_no_status(:status)"
                )
                count_params = {"status": ope_no}
            else:
                sql = text(
                    "SELECT * FROM public.get_stydy_ope_no_status(:status) WHERE study_uid = :study_uid LIMIT :limit OFFSET :offset"
                )
                params = {
                    "status": ope_no,
                    "study_uid": study_uid,
                    "limit": limit,
                    "offset": offset,
                }
                count_sql = text(
                    "SELECT COUNT(*) FROM public.get_stydy_ope_no_status(:status) WHERE study_uid = :study_uid"
                )
                count_params = {"status": ope_no, "study_uid": study_uid}

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
        self, study_uid: Optional[str] = None
    ) -> Dict[str, Any]:
        """提供查詢介面，檢視目前未完成/已完成轉檔的 study 清單。"""
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
        completed_studies = await self.query_studies_pending_completion(
            study_uid=study_uid
        )
        study_events = await self.create_study_complete_events(
            completed_studies, raw_dicom_path, rename_dicom_path, rename_nifti_path
        )
        # Process events from the provided data list
        completed_study_events = await self.identify_completed_studies(study_events)
        return {
            "studies_pending_completion": completed_studies,
            "completed_study_events": [
                StydySeriesOpeNoStatus.model_validate(result)
                for result in completed_study_events
            ],
        }
