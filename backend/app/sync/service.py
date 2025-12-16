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

def setup_dcop_event_logger():
    """
    為 DCOPEventDicomService 設置日誌記錄器
    日誌檔案格式: DCOPEventDicomService-YYYY-MM-DD.XXXX.log
    特性：
    - 單一檔案大小限制 100MB
    - 保留 7 天檔案（清理舊檔案）
    """
    # 創建 logger
    dcop_logger = logging.getLogger('DCOPEventDicomService')
    
    # 避免重複添加 handler
    if dcop_logger.handlers:
        return dcop_logger
    
    dcop_logger.setLevel(logging.DEBUG)
    
    # 設置日誌目錄
    log_dir = os.getenv('LOG_PATH', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 清理 7 天前的舊檔案
    _cleanup_old_logs(log_dir, days=7)
    
    # 生成日誌檔案名稱
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_filename = f'{date_str}.0001.DCOPEventDicomService.log'
    log_filepath = os.path.join(log_dir, log_filename)
    
    # 創建檔案 handler (最大 100MB，保留 5 個備份)
    file_handler = RotatingFileHandler(
        log_filepath,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 設置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 添加 handler
    dcop_logger.addHandler(file_handler)
    
    # 同時輸出到控制台（可選）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    dcop_logger.addHandler(console_handler)
    
    # 避免日誌向上傳播到 root logger
    dcop_logger.propagate = False
    
    return dcop_logger


def _cleanup_old_logs(log_dir: str, days: int = 7):
    """
    清理指定日期之前的日誌檔案
    
    Args:
        log_dir: 日誌目錄
        days: 保留天數（預設 7 天）
    """
    from datetime import timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days)
    cutoff_timestamp = cutoff_date.timestamp()
    
    try:
        for filename in os.listdir(log_dir):
            if filename.endswith('.DCOPEventDicomService.log') or \
               filename.endswith('.DCOPEventDicomService.log.1') or \
               filename.endswith('.DCOPEventDicomService.log.2') or \
               filename.endswith('.DCOPEventDicomService.log.3') or \
               filename.endswith('.DCOPEventDicomService.log.4') or \
               filename.endswith('.DCOPEventDicomService.log.5'):
                filepath = os.path.join(log_dir, filename)
                file_mtime = os.path.getmtime(filepath)
                
                if file_mtime < cutoff_timestamp:
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        pass  # 忽略刪除失敗
    except Exception as e:
        pass  # 忽略清理過程中的異常


# 初始化 DCOPEventDicomService logger
_dcop_event_logger = setup_dcop_event_logger()


class DCOPEventDicomService(BaseRepositoryService[DCOPEventModel]):
    """Author repository."""

    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        """Author repository."""

        model_type = DCOPEventModel

    repository_type = Repo
    pattern_str = '({}),({}),({}),({}),({}|{})'.format(DCOPStatus.SERIES_NEW.value,
                                                       DCOPStatus.SERIES_TRANSFERRING.value,
                                                       DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
                                                       DCOPStatus.SERIES_CONVERTING.value,
                                                       DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
                                                       DCOPStatus.SERIES_CONVERSION_SKIP.value
                                                       )
    can_inference_pattern = re.compile(pattern_str)
    logger = _dcop_event_logger  # 設置類級別 logger

    async def get_check_url_by_ope_no(self, ope_no: str) -> Optional[str]:
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
        self.logger.debug(f'get_check_url_by_ope_no: ope_no={ope_no}, url={url}')
        return url

    async def post_ope_no_task(self, data: List[DCOPEventRequest]):
        from code_ai import load_dotenv
        load_dotenv()
        check_url_set = set()
        # async with AsyncSession(self.repository.session.bind) as session:
        async with self.session_manager.get_session() as session:
            for dcop_event in data:
                new_data_obj = await DCOPEventModel.create_event_ope_no(tool_id=dcop_event.tool_id,
                                                                        study_uid=dcop_event.study_uid,
                                                                        series_uid=dcop_event.series_uid,
                                                                        study_id=dcop_event.study_id,
                                                                        ope_no=dcop_event.ope_no,
                                                                        result_data=dcop_event.result_data,
                                                                        params_data=dcop_event.params_data,
                                                                        session=session)
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
                if url is not None and url not in check_url_set:
                    check_url_set.add(url)
        async with httpx.AsyncClient(timeout=180) as client:
            for url in check_url_set:
                rep = await client.post(url)
        return

    async def check_study_series_transfer_complete(self, data: Optional[List[DCOPEventRequest]] = None):
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

    async def add_study_new(self, data_list):
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
                    new_data = await DCOPEventModel.create_event(study_uid=ids,
                                                                 series_uid=None,
                                                                 status=DCOPStatus.STUDY_NEW.name,
                                                                 session=session, )
                    self.logger.info(f'Created study event: {DCOPEventRequest.model_validate(new_data).model_dump()}')
                    session.add(new_data)
                    task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
                                                  output_dicom_path=rename_dicom_path,
                                                  output_nifti_path=rename_nifti_path, )
                    data_transferring = await DCOPEventModel.create_event(study_uid=ids,
                                                                          series_uid=None,
                                                                          status=DCOPStatus.STUDY_TRANSFERRING.name,
                                                                          session=session, )
                    data_transferring.params_data = task_params.get_str_dict()
                    session.add(data_transferring)
                    await session.commit()

                    # obj = await self.create_many(data=[new_data_obj,data_transferring],auto_commit=True)
                    result_list.append(new_data)
                    result_list.append(data_transferring)
            except Exception as e:
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

    async def nifti_tool_get_series_info(self, study_uid: str, session: AsyncSession):
        """
        獲取需要轉換的 series 資訊並建立 NIFTI 轉換任務
        
        重要：對於 multi-output series (如 DWI、SWAN)，為每個 output 建立獨立任務
        只有當所有 required outputs 都完成時，才進入下一階段
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

    async def dicom_tool_get_series_info(self, data: List[DCOPEventModel]):
        from code_ai.task.task_dicom2nii import dicom_to_nii
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv
        load_dotenv()
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

    async def check_study_series_conversion_complete(self, data: Optional[List[DCOPEventRequest]] = None):
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


    async def get_stydy_series_ope_no_status(self,study_uid:OrthancID,
                                             ope_no:OpeNo,
                                             limit: int,
                                             offset: int
                                      ) -> OffsetPagination[StydySeriesOpeNoStatus]:
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

    async def get_stydy_ope_no_status(self,study_uid:OrthancID,
                                      ope_no:OpeNo,
                                      limit: int,
                                      offset: int
                                      ) -> OffsetPagination[StydySeriesOpeNoStatus]:
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


    async def get_check_study_series_conversion_complete(self,study_uid: Optional[str] = None) -> Dict[str,Any] :
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
        completed_studies = await self.query_studies_pending_completion(study_uid=study_uid)
        study_events = await self.create_study_complete_events(
            completed_studies,
            raw_dicom_path,
            rename_dicom_path,
            rename_nifti_path
        )
        # Process events from the provided data list
        completed_study_events = await self.identify_completed_studies(study_events)
        return {'studies_pending_completion':completed_studies,
                "completed_study_events": [StydySeriesOpeNoStatus.model_validate(result) for result in completed_study_events]
        }
