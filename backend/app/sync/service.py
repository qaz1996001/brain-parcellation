import json
import os
import pathlib
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Set
import re
import httpx
from advanced_alchemy.extensions.fastapi import (
    repository,
    service,
)
from funboost import AsyncResult
# from fastapi import
from sqlalchemy import text, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio.engine import AsyncEngine
import logging

from code_ai.task.schema.intput_params import Dicom2NiiParams

from backend.app.service import BaseRepositoryService
from .exceptions import InvalidStateTransitionError, DCOPEventError, DuplicateEventError, ExternalServiceError
from .model import DCOPEventModel
from .schemas import DCOPStatus, DCOPEventRequest, DCOPEventNIFTITOOLRequest
from .urls import SYNC_PROT_OPE_NO, SYNC_PROT_STUDY_NIFTI_TOOL, SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID, \
    SYNC_PROT_STUDY_TRANSFER_COMPLETE

logger = logging.getLogger(__name__)


# class DCOPEventDicomService(service.SQLAlchemyAsyncRepositoryService[DCOPEventModel]):
#     class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
#         model_type = DCOPEventModel
#
#     repository_type = Repo
#     pattern_str = '({}),({}),({}),({}),({}|{})'.format(DCOPStatus.SERIES_NEW.value,
#                                                        DCOPStatus.SERIES_TRANSFERRING.value,
#                                                        DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
#                                                        DCOPStatus.SERIES_CONVERTING.value,
#                                                        DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
#                                                        DCOPStatus.SERIES_CONVERSION_SKIP.value
#                                                        )
#     can_inference_pattern = re.compile(pattern_str)
#
#     async def get_check_url_by_ope_no(self,ope_no:str) -> Optional[str]:
#         from code_ai import load_dotenv
#         load_dotenv()
#         UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
#         match ope_no:
#
#             case DCOPStatus.STUDY_TRANSFER_COMPLETE.value:
#                 url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_TRANSFER_COMPLETE}"
#             case DCOPStatus.STUDY_CONVERSION_COMPLETE.value:
#                 url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
#             case DCOPStatus.SERIES_TRANSFER_COMPLETE.value:
#                 url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_TRANSFER_COMPLETE}"
#             case DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
#                 url = f"{UPLOAD_DATA_API_URL}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
#             case _ :
#                 url = None
#         return url
#
#     async def post_ope_no_task(self,data :List[DCOPEventRequest]):
#         from code_ai import load_dotenv
#         load_dotenv()
#         check_url_set = set()
#         async with AsyncSession(self.repository.session.bind) as session:
#             for dcop_event in data:
#                 new_data_obj = await DCOPEventModel.create_event_ope_no(tool_id=dcop_event.tool_id,
#                                                                         study_uid=dcop_event.study_uid,
#                                                                         series_uid=dcop_event.series_uid,
#                                                                         study_id=dcop_event.study_id,
#                                                                         ope_no=dcop_event.ope_no,
#                                                                         result_data=dcop_event.result_data,
#                                                                         params_data=dcop_event.params_data,
#                                                                         session=session)
#                 session.add(new_data_obj)
#                 await session.commit()
#                 await session.refresh(new_data_obj)
#
#                 # new_data_obj = await self.create(data=new_data, auto_commit=True, auto_refresh=True)
#                 match new_data_obj.ope_no:
#                     case DCOPStatus.SERIES_TRANSFER_COMPLETE.value:
#                         url = await self.get_check_url_by_ope_no(new_data_obj.ope_no)
#                         data_dict = dict(study_uid=dcop_event.study_uid,)
#                     case DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
#                         url = await self.get_check_url_by_ope_no(new_data_obj.ope_no)
#                     case _ :
#                         url = None
#                 if url is not None and url not in check_url_set:
#                     check_url_set.add(url)
#         async with httpx.AsyncClient(timeout=180) as client:
#             for url in check_url_set:
#                 rep = await client.post(url)
#         return
#
    # async def check_study_series_transfer_complete(self, data: Optional[List[DCOPEventRequest]] = None):
    #     """
    #     Checks if all series under a study have completed transfer and initiates the conversion process.
    #         檢查 study 下的 series 是否都傳輸完成
    #         1. series 完成傳輸添加 SERIES_TRANSFER_COMPLETE  的記錄
    #         2. 所有series都到了SERIES_TRANSFER_COMPLETE， 添加 STUDY_TRANSFER_COMPLETE 的記錄
    #         3. 添加 STUDY_CONVERTING 的記錄，
    #         4. 發送管道任務  進行轉換
    #     Process flow:
    #     1. Mark series completion with SERIES_TRANSFER_COMPLETE record
    #     2. When all series reach SERIES_TRANSFER_COMPLETE, add STUDY_TRANSFER_COMPLETE record
    #     3. Add STUDY_CONVERTING record
    #     4. Send pipeline task for conversion
    #
    #     Args:
    #         data: Optional list of DCOPEventRequest objects. If None, retrieves study status from database.
    #
    #     """
    #     from code_ai import load_dotenv
    #     load_dotenv()
    #     print('check_study_series_transfer_complete data',data)
    #     # Get configuration from environment
    #     upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
    #     path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
    #     path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")
    #
    #     # Retrieve study status information if not provided
    #     if data is None:
    #         dcop_event_list, dcop_event_dump_list = await self._get_studies_ready_for_transfer()
    #     else:
    #         dcop_event_list = [DCOPEventRequest.model_validate(event,strict=False) for event in data]
    #         dcop_event_dump_list = [dcop_event.model_dump() for dcop_event in dcop_event_list]
    #
    #     # Process eligible studies for conversion
    #     if dcop_event_list:
    #         await self._send_events(upload_data_api_url, dcop_event_dump_list)
    #         await self._initiate_conversion_process(upload_data_api_url, dcop_event_list, path_rename_dicom,
    #                                                 path_rename_nifti)

#         return dcop_event_list
#
#     async def add_study_new(self,data_list):
#         from code_ai.task.schema.intput_params import Dicom2NiiParams
#         from code_ai import load_dotenv
#         load_dotenv()
#         raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
#         rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
#         rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
#
#         result_list = []
#
#         for ids in data_list:
#             existing = await self.repository.session.execute(
#                 select(DCOPEventModel).where(
#                     DCOPEventModel.study_uid == ids,
#             ))
#             existing_record = existing.scalars().first()
#
#             # 如果记录不存在，则创建新记录
#             if not existing_record:
#                 study_uid_raw_dicom_path = raw_dicom_path.joinpath(ids)
#                 new_data = await DCOPEventModel.create_event(study_uid=ids,
#                                                              series_uid=None,
#                                                              status=DCOPStatus.STUDY_NEW.name,
#                                                              session=self.repository.session, )
#                 new_data_obj = await self.create(data=new_data)
#                 task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
#                                               output_dicom_path=rename_dicom_path,
#                                               output_nifti_path=rename_nifti_path, )
#                 data_transferring = await DCOPEventModel.create_event(study_uid=ids,
#                                                                       series_uid=None,
#                                                                       status=DCOPStatus.STUDY_TRANSFERRING.name,
#                                                                       session=self.repository.session, )
#                 data_transferring.params_data = task_params.get_str_dict()
#
#                 obj = await self.create_many(data=[new_data_obj,data_transferring],auto_commit=True)
#                 result_list.append(new_data_obj)
#
#         return result_list
#
#     async def _get_studies_ready_for_transfer(self) -> Tuple[List[DCOPEventRequest], List[dict]]:
#         """
#         Retrieves studies that are ready for transfer completion marking.
#
#         Returns:
#             Tuple containing list of DCOPEventRequest objects and their serialized versions.
#         """
#         engine: AsyncEngine = self.repository.session.bind
#         dcop_event_list = []
#         dcop_event_dump_list = []
#
#         async with engine.connect() as conn:
#             results = await conn.execute(text('select * from public.get_all_studies_status()'))
#
#             for result in results.all():
#                 print('result',result)
#                 study_data = result[0]
#                 dcop_event = DCOPEventRequest(
#                     study_uid=study_data['study_uid'],
#                     series_uid=None,
#                     ope_no=DCOPStatus.STUDY_TRANSFER_COMPLETE.value,
#                     study_id=study_data['study_id'],
#                     tool_id='DICOM_TOOL',
#                     result_data={'result': json.dumps(study_data['result'])}
#                 )
#                 dcop_event_dump_list.append(dcop_event.model_dump())
#                 dcop_event_list.append(dcop_event)
#
#         return dcop_event_list, dcop_event_dump_list
#
#     async def _send_events(self, api_url: str, event_data: List[dict]) -> None:
#         """
#         Sends study transfer complete events to the API.
#
#         Args:
#             api_url: Base URL for the upload data API.
#             event_data: List of serialized DCOPEventRequest objects.
#         """
#         async with httpx.AsyncClient(timeout=180) as client:
#             url = f"{api_url}{SYNC_PROT_OPE_NO}"
#             event_data_json = json.dumps(event_data)
#             await client.post(url=url, timeout=180, data=event_data_json)
#
    # async def _initiate_conversion_process(
    #         self,
    #         api_url: str,
    #         events: List[DCOPEventRequest],
    #         dicom_path: str,
    #         nifti_path: str
    # ) -> None:
    #     """
    #     Initiates the conversion process for each study.
    #
    #     Args:
    #         api_url: Base URL for the upload data API.
    #         events: List of DCOPEventRequest objects.
    #         dicom_path: Path for renamed DICOM files.
    #         nifti_path: Path for NIFTI output.
    #     """
    #
    #     url = f"{api_url}{SYNC_PROT_STUDY_NIFTI_TOOL}"
    #     for event in events:
    #         study_id = event.study_id
    #         output_dicom_path = pathlib.Path(os.path.join(dicom_path, study_id))
    #         output_nifti_path = pathlib.Path(nifti_path)
    #
    #         # Prepare conversion parameters
    #         task_params = Dicom2NiiParams(
    #             sub_dir=None,
    #             output_dicom_path=output_dicom_path,
    #             output_nifti_path=output_nifti_path
    #         )
    #
    #         # Create and send the conversion request
    #         async with httpx.AsyncClient(timeout=180) as client:
    #             nifti_tool_request = DCOPEventNIFTITOOLRequest(
    #                 ope_no=DCOPStatus.STUDY_CONVERTING.value,
    #                 study_id=study_id,
    #                 tool_id='NIFTI_TOOL',
    #                 params_data=task_params.get_str_dict(),
    #                 result_data=None
    #             )
    #
    #             request_data = json.dumps([nifti_tool_request.model_dump()])
    #             await client.post(url=url, data=request_data)

#     async def study_series_nifti_tool(self,data :List[DCOPEventNIFTITOOLRequest]):
#         """
#         建立
#            DCOPStatus.STUDY_CONVERTING
#            DCOPStatus.SERIES_CONVERTING
#            DCOPStatus.SERIES_CONVERSION_COMPLETE
#            DCOPStatus.STUDY_CONVERSION_COMPLETE
#         """
#
#         from code_ai import load_dotenv
#         load_dotenv()
#         session:AsyncSession = self.repository.session
#         for dcop in data:
#             match dcop.ope_no:
#                 case DCOPStatus.STUDY_CONVERTING.value:
#                     async with session:
#                         # DICOM_TOOL
#                         conf_query = select(DCOPEventModel, ).where(and_(*[DCOPEventModel.study_id == dcop.study_id,
#                                                                            DCOPEventModel.study_uid.isnot(None),
#                                                                            DCOPEventModel.tool_id == 'DICOM_TOOL',
#                                                                            DCOPEventModel.ope_no == DCOPStatus.SERIES_TRANSFER_COMPLETE.value]))
#                         execute = await session.execute(conf_query)
#                         dcop_event = execute.first()[0]
#                     study_transfer_complete_data = await DCOPEventModel.create_event_ope_no(tool_id=dcop.tool_id,
#                                                                                             study_uid=dcop_event.study_uid,
#                                                                                             series_uid=None,
#                                                                                             study_id=dcop.study_id,
#                                                                                             ope_no=dcop.ope_no,
#                                                                                             result_data=dcop.result_data,
#                                                                                             params_data=dcop.params_data,
#                                                                                             session=session)
#                     async with session:
#                         session.add(study_transfer_complete_data)
#                         await session.commit()
#                         await session.refresh(study_transfer_complete_data)
#                         await self.nifti_tool_get_series_info(dcop_event.study_uid, session)
#                 case DCOPStatus.SERIES_CONVERTING.value:
#                     pass
#                     # new_data_obj = await self.create(new_data, auto_commit=True)
#
#
#         upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
#         url = f"{upload_data_api_url}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
#         async with httpx.AsyncClient(timeout=180) as client:
#             await client.post(url=url)
#
#     async def nifti_tool_get_series_info(self, study_uid: str,session:AsyncSession):
#         from code_ai.task.task_dicom2nii import dicom_2_nii_series
#         from code_ai.task.schema.intput_params import Dicom2NiiSeriesParams
#         from code_ai import load_dotenv
#         load_dotenv()
#         path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
#         path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")
#         engine: AsyncEngine = session.bind
#         async with engine.connect() as conn:
#             sql = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid')
#             results = await conn.execute(sql,
#                                          {'status': DCOPStatus.STUDY_CONVERTING.value,
#                                           'study_uid': study_uid})
#
#         dcop_event_list = results.all()
#         task_params_list = []
#         dcop_model_list  = []
#         for dcop_event in dcop_event_list:
#             result_data = dcop_event.result_data[0]
#             output_dicom_path = result_data['rename_dicom_path']
#             output_nifti_path = pathlib.Path(path_rename_nifti)
#             task_params = Dicom2NiiSeriesParams(sub_dir=None,
#                                                 study_uid=dcop_event.study_uid,
#                                                 series_uid =dcop_event.series_uid,
#                                                 output_dicom_path=output_dicom_path,
#                                                 output_nifti_path=output_nifti_path)
#             new_data_obj = await DCOPEventModel.create_event_ope_no(tool_id='NIFTI_TOOL',
#                                                                     study_uid=dcop_event.study_uid,
#                                                                     series_uid=dcop_event.series_uid,
#                                                                     study_id=dcop_event.study_id,
#                                                                     ope_no=DCOPStatus.SERIES_CONVERTING.value,
#                                                                     result_data=dcop_event.result_data,
#                                                                     params_data=task_params.get_str_dict(),
#                                                                     session=session)
#             dcop_model_list.append(new_data_obj)
#             task_params_list.append(task_params)
#
#         try:
#             if dcop_event_list:  # Check if dcop_event_list is not empty/falsy
#                 # Attempt to create many records and auto-commit
#                 # If create_many raises an exception, the 'except' block will catch it,
#                 # and the push operations will not be executed.
#                 # data_obj = await self.create_many(dcop_model_list, auto_commit=True)
#                 print('session.add_all',dcop_model_list)
#                 session.add_all(dcop_model_list)
#                 await session.commit()
#                 for dcop_model in dcop_model_list:
#                     await session.refresh(dcop_model)
#
#                 # If we reach here, create_many completed successfully and committed.
#                 # Now, proceed with pushing tasks.
#                 for task_params in task_params_list:
#                     dicom_2_nii_series.push(task_params.get_str_dict())
#             else:
#                 await session.rollback()
#                 # If dcop_event_list is empty, there's nothing to create or push.
#                 # A rollback here is likely unnecessary if nothing was attempted.
#                 # You might just want to pass or log.
#                 print("dcop_event_list is empty, no records to create or push.")
#                 # await self.repository.session.rollback() # Potentially redundant if nothing happened
#         except Exception as e:  # Catch specific exceptions for better debugging
#             # An error occurred during create_many or subsequent push operations.
#             # Rollback ensures no partial changes are left if auto_commit somehow failed or
#             # if you had other uncommitted operations before this try block.
#             await self.repository.session.rollback()
#             print(f"An error occurred: {e}. Database transaction rolled back.")
#             # Re-raise the exception if you want it to propagate further up the call stack
#             raise
#         finally:
#             pass
#
#
#     async def dicom_tool_get_series_info(self, data :List[DCOPEventModel]):
#         from code_ai.task.task_dicom2nii import dicom_to_nii
#         from code_ai.task.schema.intput_params import Dicom2NiiParams
#         from code_ai import load_dotenv
#         load_dotenv()
#         raw_dicom_path    = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
#         rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
#         rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
#         for dcop_event in data:
#             study_uid = dcop_event.study_uid
#             study_uid_raw_dicom_path = raw_dicom_path.joinpath(study_uid)
#             if study_uid_raw_dicom_path.exists():
#                 series_uid_path_list = sorted(study_uid_raw_dicom_path.iterdir())
#                 new_data_list = []
#                 task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
#                                               output_dicom_path=rename_dicom_path,
#                                               output_nifti_path=rename_nifti_path, )
#
#                 for series_uid_path in series_uid_path_list:
#                     series_new_data = await DCOPEventModel.create_event(study_uid=study_uid,
#                                                                         series_uid=series_uid_path.name,
#                                                                         status=DCOPStatus.SERIES_NEW.name,
#                                                                         session=self.repository.session, )
#                     series_transferring_data = await DCOPEventModel.create_event(study_uid=study_uid,
#                                                                                  series_uid=series_uid_path.name,
#                                                                                  status=DCOPStatus.SERIES_TRANSFERRING.name,
#                                                                                  session=self.repository.session, )
#                     series_transferring_data.params_data = task_params.get_str_dict()
#                     new_data_list.append(series_new_data)
#                     new_data_list.append(series_transferring_data)
#                 try:
#                     data_obj = await self.create_many(new_data_list, auto_commit=True)
#                     task = dicom_to_nii.push(task_params.get_str_dict())
#                 except:
#                     await self.repository.session.rollback()
#
#         return None
#
#
#     async def check_study_series_conversion_complete(self, data: Optional[List[DCOPEventRequest]] = None):
#         """
#             檢查 study 下的 series 是否都轉成 nifti
#             1. series 完成轉成nifti ， 添加 SERIES_CONVERSION_COMPLETE  的記錄
#             2. 所有series都到了 SERIES_CONVERSION_COMPLETE， 添加 STUDY_CONVERSION_COMPLETE 的記錄
#             3. 添加 STUDY_INFERENCE_READY 的記錄，
#             4. 發送管道任務  推論
#         """
#         from code_ai.task.task_pipeline import task_pipeline_inference
#
#         # Environment variables setup
#         upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
#         raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
#         rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
#         rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
#         print('data', data)
#         # post_check_study_series_conversion_complete call check
#         if data is None:
#             # Query studies not yet at STUDY_CONVERSION_COMPLETE status
#             completed_studies = await self._query_studies_pending_completion()
#         else:
#             completed_studies = set()
#             for dcop_enent in data:
#                 result_set = await self._query_studies_pending_completion(dcop_enent.study_uid)
#                 print('result_set',result_set)
#                 completed_studies.update(result_set)
#
#         if not completed_studies:
#             return None
#         print('completed_studies',completed_studies)
#         # Create and send study completion events
#         study_events = await self._create_study_complete_events(
#             completed_studies,
#             raw_dicom_path,
#             rename_dicom_path,
#             rename_nifti_path
#         )
#         # Process events from the provided data list
#         print('study_events', study_events)
#         completed_study_events = await self._identify_completed_studies(study_events)
#
#         # Process completed studies and queue them for inference
#         if completed_study_events:
#             # Queue inference tasks for completed studies
#             await self._queue_inference_tasks(
#                 completed_study_events,
#                 upload_data_api_url,
#                 rename_dicom_path,
#                 rename_nifti_path,
#                 task_pipeline_inference
#             )
#         return None
#
#     async def _query_studies_pending_completion(self,study_uid:Optional[str]=None):
#         """Query for studies that have not yet reached STUDY_CONVERSION_COMPLETE status."""
#         async with self.repository.session as session:
#             if study_uid is None:
#                 sql    = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status)')
#                 params = {'status': DCOPStatus.STUDY_CONVERSION_COMPLETE.value}
#             else:
#                 sql = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid')
#                 params = {'status': DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
#                           'study_uid':study_uid}
#             execute = await session.execute(sql,params)
#             results = execute.all()
#
#         can_inference_dict = {}
#         wait_inference_dict = {}
#         for result in results:
#             test_str = ','.join(list(result.ope_no))
#             match_result = self.can_inference_pattern.match(test_str)
#             print('match_result',match_result,test_str)
#             print(self.pattern_str,self.can_inference_pattern.findall(test_str))
#             if match_result:
#                 can_inference_dict.update({result.series_uid: (result.study_uid, result.study_id)})
#             else:
#                 wait_inference_dict.update({result.series_uid: (result.study_uid, result.study_id)})
#
#         wait_inference_set = set(wait_inference_dict.values())
#         can_inference_set = set(can_inference_dict.values())
#         if wait_inference_dict:
#             result_set = can_inference_set - wait_inference_set
#         else:
#             result_set = can_inference_set
#
#         return result_set
#
#     async def _create_study_complete_events(self, study_data_list, raw_dicom_path, rename_dicom_path,
#                                             rename_nifti_path,upload_data_api_url):
#         """Create STUDY_CONVERSION_COMPLETE events for studies with all series converted."""
#         study_events = []
#
#         for data in study_data_list:
#             study_uid_raw_dicom_path = raw_dicom_path.joinpath(data[0])
#
#             dcop_event = DCOPEventRequest(
#                 study_uid=data[0],
#                 series_uid=None,
#                 study_id=data[1],
#                 ope_no=DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
#                 tool_id='NIFTI_TOOL',
#                 params_data=dict(
#                     sub_dir=str(study_uid_raw_dicom_path),
#                     output_dicom_path=str(rename_dicom_path),
#                     output_nifti_path=str(rename_nifti_path),
#                 ),
#                 result_data=dict(
#                     sub_dir=str(study_uid_raw_dicom_path),
#                     output_dicom_path=str(rename_dicom_path.joinpath(data[1])),
#                     output_nifti_path=str(rename_nifti_path.joinpath(data[1]))
#                 )
#             )
#             study_events.append(dcop_event)
#
#             # Send inference events
#
#         upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
#         study_events_dump_list = list(map(lambda x:x.model_dump(),study_events))
#         await self._send_events(upload_data_api_url,study_events_dump_list)
#
#         return study_events
#
#     async def _queue_inference_tasks(self, study_events, upload_data_api_url, rename_dicom_path,
#                                      rename_nifti_path, task_pipeline_inference):
#         """Queue inference tasks for completed studies and send related events."""
#         async with self.repository.session as session:
#             for dcop_event in study_events:
#                 # Check if this study already has an inference task
#                 query = select(DCOPEventModel).where(
#                     and_(
#                         DCOPEventModel.study_uid == dcop_event.study_uid,
#                         DCOPEventModel.tool_id == 'INFERENCE_TOOL',
#                         DCOPEventModel.ope_no.in_([
#                             DCOPStatus.STUDY_INFERENCE_READY.value,
#                             DCOPStatus.STUDY_INFERENCE_QUEUED.value,
#                             DCOPStatus.STUDY_INFERENCE_RUNNING.value,
#                             DCOPStatus.STUDY_INFERENCE_COMPLETE.value
#                         ])
#                     )
#                 )
#                 result = await session.execute(query)
#                 existing_inference = result.scalars().first()
#
#                 # Skip if this study already has an inference task
#                 if existing_inference:
#                     print(f"Skipping duplicate inference task for study_uid: {dcop_event.study_uid}, "
#                           f"existing status: {existing_inference.ope_no}")
#                     continue
#
#                 # Proceed with creating inference task if not a duplicate
#                 dicom_study_path = rename_dicom_path.joinpath(dcop_event.study_id)
#                 nifti_study_path = rename_nifti_path.joinpath(dcop_event.study_id)
#
#                 # Create STUDY_INFERENCE_READY event
#                 dcop_event_inference_ready = DCOPEventRequest(
#                     study_uid=dcop_event.study_uid,
#                     series_uid=None,
#                     study_id=dcop_event.study_id,
#                     ope_no=DCOPStatus.STUDY_INFERENCE_READY.value,
#                     tool_id='INFERENCE_TOOL',
#                     params_data={
#                         'nifti_study_path': str(nifti_study_path),
#                         'dicom_study_path': str(dicom_study_path),
#                         'study_uid': dcop_event.study_uid,
#                         'study_id': dcop_event.study_id
#                     }
#                 )
#                 # Push to inference task pipeline
#                 task_pipeline_result: AsyncResult = task_pipeline_inference.push(
#                     dcop_event_inference_ready.params_data
#                 )
#
#                 # Create STUDY_INFERENCE_QUEUED event
#                 dcop_event_inference_queued = DCOPEventRequest(
#                     study_uid=dcop_event.study_uid,
#                     series_uid=None,
#                     study_id=dcop_event.study_id,
#                     ope_no=DCOPStatus.STUDY_INFERENCE_QUEUED.value,
#                     tool_id='INFERENCE_TOOL',
#                     params_data={
#                         'nifti_study_path': str(nifti_study_path),
#                         'dicom_study_path': str(dicom_study_path),
#                         'study_uid': dcop_event.study_uid,
#                         'study_id': dcop_event.study_id,
#                         'task_pipeline_id': task_pipeline_result.task_id
#                     }
#                 )
#
#                 # Send inference events
#                 await self._send_events(upload_data_api_url,
#                                         [dcop_event_inference_ready.model_dump(),
#                                          dcop_event_inference_queued.model_dump()])
#
#     def _group_series_by_study(self, events):
#         """Group series completion events by study."""
#         series_by_study = {}
#
#         for event in events:
#             if event.ope_no == DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
#                 if event.study_uid not in series_by_study:
#                     series_by_study[event.study_uid] = {
#                         'completed': set(),
#                         'study_id': event.study_id
#                     }
#
#                 # Add this series to the completed set
#                 if event.series_uid:
#                     series_by_study[event.study_uid]['completed'].add(event.series_uid)
#
#         return series_by_study
#
#     async def _identify_completed_studies(self, study_events_list:List[DCOPEventRequest]):
#         """Identify studies with all series converted and create completion events."""
#         completed_study_events = []
#         # Query to get all series for this study
#         async with self.repository.session as session:
#             done_count = 0
#             undone     = 0
#             for study_events in study_events_list:
#                 sql    = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid')
#                 params = {'status': DCOPStatus.STUDY_CONVERSION_COMPLETE.value,
#                           'study_uid': study_events.study_uid}
#                 execute = await session.execute(sql,params)
#                 results = execute.all()
#                 for result in results:
#                     if DCOPStatus.SERIES_CONVERSION_COMPLETE.value in result.ope_no:
#                         done_count+=1
#                     elif DCOPStatus.SERIES_CONVERSION_SKIP.value in result.ope_no:
#                         done_count+=1
#                     else:
#                         undone+=1
#                 if done_count== len(results):
#                     completed_study_events.append(result)
#         return completed_study_events


class DCOPEventDicomService(BaseRepositoryService[DCOPEventModel]):

    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
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

    async def post_ope_no_task(self, data: List[DCOPEventRequest]):

        """
        處理操作編號（ope_no）更新任務

        Args:
            data: DCOPEventRequest 列表，包含要更新的事件狀態

        Returns:
            Dict 包含處理結果：
            - processed: 成功處理的記錄數
            - failed: 失敗的記錄數
            - errors: 錯誤詳情列表
            - triggered_checks: 觸發的檢查 URL 列表
        """
        from code_ai import load_dotenv
        load_dotenv()

        # 初始化結果統計
        result = {
            "processed": 0,
            "failed": 0,
            "errors": [],
            "triggered_checks": []
        }

        # 用於追蹤需要觸發的檢查
        check_urls_with_data: Dict[str, Set[Tuple[str, str]]] = {}  # {url: {(study_uid, study_id)}}

        # 批量處理準備
        events_to_create = []
        events_metadata = []

        # 使用統一的 session 管理
        async with self.session_manager.get_session() as session:
            try:
                # 1. 驗證並準備所有事件
                for idx, dcop_event in enumerate(data):
                    try:
                        # 驗證狀態轉換的合法性
                        await self._validate_state_transition(
                            dcop_event, session
                        )

                        # 檢查是否為重複事件
                        is_duplicate = await self._check_duplicate_event(
                            dcop_event, session
                        )

                        if is_duplicate:
                            logger.warning(
                                f"Duplicate event detected for study_uid: {dcop_event.study_uid}, "
                                f"series_uid: {dcop_event.series_uid}, ope_no: {dcop_event.ope_no}"
                            )
                            result["errors"].append({
                                "index": idx,
                                "error": "Duplicate event",
                                "study_uid": dcop_event.study_uid,
                                "ope_no": dcop_event.ope_no
                            })
                            result["failed"] += 1
                            continue

                        # 創建新的事件對象
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

                        events_to_create.append(new_data_obj)
                        events_metadata.append({
                            "event": dcop_event,
                            "index": idx
                        })

                        # 收集需要觸發的檢查
                        await self._collect_check_urls(
                            dcop_event, check_urls_with_data
                        )

                    except (InvalidStateTransitionError, ValueError) as e:
                        logger.error(f"Validation error for event {idx}: {e}")
                        result["errors"].append({
                            "index": idx,
                            "error": str(e),
                            "study_uid": dcop_event.study_uid,
                            "ope_no": dcop_event.ope_no
                        })
                        result["failed"] += 1
                    except Exception as e:
                        logger.error(f"Unexpected error for event {idx}: {e}")
                        result["errors"].append({
                            "index": idx,
                            "error": f"Unexpected error: {str(e)}",
                            "study_uid": dcop_event.study_uid
                        })
                        result["failed"] += 1

                # 2. 批量保存事件
                if events_to_create:
                    try:
                        # 批量添加到 session
                        session.add_all(events_to_create)
                        await session.flush()

                        # 為每個成功創建的事件更新相關狀態
                        for event_obj, metadata in zip(events_to_create, events_metadata):
                            await self._update_related_status(
                                event_obj, metadata["event"], session
                            )

                        # 提交事務
                        await session.commit()
                        result["processed"] = len(events_to_create)

                        logger.info(f"Successfully processed {result['processed']} events")

                    except IntegrityError as e:
                        await session.rollback()
                        logger.error(f"Database integrity error: {e}")
                        result["errors"].append({
                            "error": "Database integrity violation",
                            "details": str(e)
                        })
                        result["failed"] += len(events_to_create)
                        result["processed"] = 0
                        # 清空檢查 URLs，因為事務失敗
                        check_urls_with_data.clear()

                    except SQLAlchemyError as e:
                        await session.rollback()
                        logger.error(f"Database error: {e}")
                        result["errors"].append({
                            "error": "Database error",
                            "details": str(e)
                        })
                        result["failed"] += len(events_to_create)
                        result["processed"] = 0
                        check_urls_with_data.clear()

            except Exception as e:
                logger.error(f"Critical error in post_ope_no_task: {e}")
                result["errors"].append({
                    "error": "Critical error",
                    "details": str(e)
                })
                raise

        # 3. 在事務成功提交後，觸發外部檢查
        if check_urls_with_data and result["processed"] > 0:
            triggered_urls = await self._trigger_external_checks(
                check_urls_with_data
            )
            result["triggered_checks"] = triggered_urls

        return result


    async def _check_duplicate_event(
            self,
            dcop_event: DCOPEventRequest,
            session: AsyncSession
    ) -> bool:
        """
        檢查是否為重複事件

        Args:
            dcop_event: 要檢查的事件
            session: 資料庫 session

        Returns:
            bool: True 如果是重複事件
        """
        # 定義時間窗口（例如：5分鐘內的相同事件視為重複）
        time_window = datetime.utcnow() - timedelta(minutes=5)

        stmt = select(DCOPEventModel).where(
            and_(
                DCOPEventModel.study_uid == dcop_event.study_uid,
                DCOPEventModel.series_uid == dcop_event.series_uid
                if dcop_event.series_uid else DCOPEventModel.series_uid.is_(None),
                DCOPEventModel.tool_id == dcop_event.tool_id,
                DCOPEventModel.ope_no == dcop_event.ope_no,
                DCOPEventModel.create_time >= time_window
            )
        )

        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def _collect_check_urls(
            self,
            dcop_event: DCOPEventRequest,
            check_urls_with_data: Dict[str, Set[Tuple[str, str]]]
    ) -> None:
        """
        收集需要觸發的檢查 URL

        Args:
            dcop_event: 當前事件
            check_urls_with_data: 收集檢查 URL 的字典
        """
        url = await self.get_check_url_by_ope_no(dcop_event.ope_no)

        if url:
            if url not in check_urls_with_data:
                check_urls_with_data[url] = set()

            # 收集相關的 study 資訊
            check_urls_with_data[url].add(
                (dcop_event.study_uid, dcop_event.study_id or "")
            )

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
        return url

    async def _update_related_status(
            self,
            event_obj: DCOPEventModel,
            dcop_event: DCOPEventRequest,
            session: AsyncSession
    ) -> None:
        """
        更新相關的狀態記錄

        Args:
            event_obj: 新創建的事件對象
            dcop_event: 原始請求
            session: 資料庫 session
        """
        # 如果是 series 完成事件，檢查是否需要更新 study 狀態
        if dcop_event.ope_no == DCOPStatus.SERIES_TRANSFER_COMPLETE.value:
            await self._check_and_update_study_transfer_status(
                dcop_event.study_uid, session
            )
        elif dcop_event.ope_no == DCOPStatus.SERIES_CONVERSION_COMPLETE.value:
            await self._check_and_update_study_conversion_status(
                dcop_event.study_uid, session
            )

    async def _check_and_update_study_transfer_status(self,
                                                      study_uid: str,
                                                      session: AsyncSession
                                                      ) -> None:
        """
        檢查並更新 study 的傳輸狀態

        當所有 series 都達到 SERIES_TRANSFER_COMPLETE (100.095) 狀態時，
        創建 STUDY_TRANSFER_COMPLETE 記錄

        Args:
            study_uid: Study 的 UUID
            session: 資料庫 session
        """
        try:
            # 查詢該 study 下所有 series 的狀態
            sql = text(
                'SELECT * FROM public.get_stydy_series_ope_no_status(:status) '
                'WHERE study_uid = :study_uid'
            )

            result = await session.execute(
                sql,
                {
                    'status': DCOPStatus.STUDY_TRANSFER_COMPLETE.value,
                    'study_uid': study_uid
                }
            )

            series_status_list = result.all()

            if not series_status_list:
                logger.warning(f"No series found for study_uid: {study_uid}")
                return

            # 分析每個 series 的狀態
            all_series_complete = True
            incomplete_series = []
            complete_series = []
            study_id = None

            for series_info in series_status_list:
                # 提取基本信息
                series_uid = series_info.series_uid
                study_id = series_info.study_id  # 所有 series 應該有相同的 study_id
                ope_no_list = series_info.ope_no  # 這是一個列表

                # 檢查該 series 是否已達到 SERIES_TRANSFER_COMPLETE
                if DCOPStatus.SERIES_TRANSFER_COMPLETE.value in ope_no_list:
                    complete_series.append(series_uid)
                else:
                    # 檢查是否有其他阻止狀態
                    if (DCOPStatus.SERIES_NEW.value in ope_no_list or
                            DCOPStatus.SERIES_TRANSFERRING.value in ope_no_list):
                        all_series_complete = False
                        incomplete_series.append(series_uid)

            logger.info(
                f"Study {study_uid} transfer status check: "
                f"total_series={len(series_status_list)}, "
                f"complete={len(complete_series)}, "
                f"incomplete={len(incomplete_series)}"
            )

            # 如果所有 series 都已完成傳輸
            if all_series_complete and study_id:
                # 檢查是否已存在 STUDY_TRANSFER_COMPLETE 記錄
                existing_study_complete = await session.execute(
                    select(DCOPEventModel).where(
                        and_(
                            DCOPEventModel.study_uid == study_uid,
                            DCOPEventModel.series_uid.is_(None),
                            DCOPEventModel.ope_no == DCOPStatus.STUDY_TRANSFER_COMPLETE.value
                        )
                    )
                )

                if not existing_study_complete.scalar_one_or_none():
                    # 收集所有 series 的結果數據
                    study_result_data = {
                        "study_uid": study_uid,
                        "study_id": study_id,
                        "series_count": len(series_status_list),
                        "complete_series": complete_series,
                        "transfer_complete_time": datetime.utcnow().isoformat()
                    }

                    # 從第一個 series 提取 study 級別的參數
                    first_series = series_status_list[0]
                    params_data_array = first_series.params_data

                    # 查找包含 study 級別路徑的參數
                    study_params = None
                    if params_data_array:
                        for param_str in params_data_array:
                            if param_str and param_str != "null":
                                try:
                                    param_dict = json.loads(param_str)
                                    if (param_dict.get("sub_dir") and
                                            "output_dicom_path" in param_dict and
                                            "series_uid" not in param_dict):
                                        study_params = param_dict
                                        break
                                except json.JSONDecodeError:
                                    continue

                    # 創建 STUDY_TRANSFER_COMPLETE 事件
                    study_complete_event = await DCOPEventModel.create_event_ope_no(
                        tool_id='DICOM_TOOL',
                        study_uid=study_uid,
                        series_uid=None,
                        study_id=study_id,
                        ope_no=DCOPStatus.STUDY_TRANSFER_COMPLETE.value,
                        result_data=study_result_data,
                        params_data=study_params or {},
                        session=session
                    )

                    session.add(study_complete_event)
                    await session.flush()

                    logger.info(
                        f"Created STUDY_TRANSFER_COMPLETE event for study_uid: {study_uid}, "
                        f"study_id: {study_id}"
                    )

                    # 觸發後續的檢查流程
                    await self._trigger_study_transfer_complete_check(
                        study_uid, study_id, session
                    )
                else:
                    logger.info(
                        f"STUDY_TRANSFER_COMPLETE already exists for study_uid: {study_uid}"
                    )
            else:
                logger.debug(
                    f"Study {study_uid} not ready for STUDY_TRANSFER_COMPLETE. "
                    f"Incomplete series: {incomplete_series}"
                )

        except Exception as e:
            logger.error(
                f"Error in _check_and_update_study_transfer_status for "
                f"study_uid {study_uid}: {e}"
            )
            raise


    async def _trigger_study_transfer_complete_check(
            self,
            study_uid: str,
            study_id: str,
            session: AsyncSession
    ) -> None:
        """
        觸發 STUDY_TRANSFER_COMPLETE 後的檢查流程

        Args:
            study_uid: Study UUID
            study_id: Study ID
            session: 資料庫 session
        """
        # 這裡可以添加額外的邏輯，例如：
        # 1. 觸發下一步的處理流程
        from code_ai import load_dotenv
        load_dotenv()
        print('check_study_series_transfer_complete data', data)
        # Get configuration from environment
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
        path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")

        # 記錄審計日誌

        audit_log = {
            "event_type": "STUDY_TRANSFER_COMPLETE",
            "study_uid": study_uid,
            "study_id": study_id,
            "timestamp": datetime.utcnow().isoformat(),
            "triggered_by": "auto_check"
        }
        await self._initiate_conversion_process(upload_data_api_url, dcop_event_list, path_rename_dicom,
                                                path_rename_nifti)

        logger.info(f"Study transfer complete audit: {audit_log}")


    async def _initiate_conversion_process(
            self,
            api_url: str,
            events: DCOPEventRequest,
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

    def _parse_series_data(self, series_info) -> dict:
        """
        解析 series 數據結構

        Args:
            series_info: 從數據庫查詢返回的 series 信息

        Returns:
            解析後的字典
        """
        parsed_data = {
            "study_uid": series_info.study_uid,
            "series_uid": series_info.series_uid,
            "study_id": series_info.study_id,
            "ope_no_list": series_info.ope_no,
            "result_data": [],
            "params_data": []
        }

        # 解析 result_data
        if series_info.result_data:
            result_array = series_info.result_data
            for result_str in result_array:
                if result_str and result_str != "null":
                    try:
                        result_dict = json.loads(result_str)
                        parsed_data["result_data"].append(result_dict)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse result_data: {result_str}")

        # 解析 params_data
        if series_info.params_data:
            params_array = series_info.params_data
            for param_str in params_array:
                if param_str and param_str != "null":
                    try:
                        param_dict = json.loads(param_str)
                        parsed_data["params_data"].append(param_dict)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse params_data: {param_str}")

        return parsed_data

    def _check_series_transfer_status(self, ope_no_list: list) -> str:
        """
        檢查 series 的傳輸狀態

        Args:
            ope_no_list: ope_no 列表

        Returns:
            狀態字符串: 'complete', 'incomplete', 'skip'
        """
        # 檢查是否已完成
        if DCOPStatus.SERIES_TRANSFER_COMPLETE.value in ope_no_list:
            return 'complete'

        # 檢查是否被跳過
        if DCOPStatus.SERIES_CONVERSION_SKIP.value in ope_no_list:
            return 'skip'

        # 檢查是否還在進行中
        if (DCOPStatus.SERIES_NEW.value in ope_no_list or
                DCOPStatus.SERIES_TRANSFERRING.value in ope_no_list):
            return 'incomplete'

        return 'unknown'

    async def _check_and_update_study_conversion_status(
            self,
            study_uid: str,
            session: AsyncSession
    ) -> None:
        """
        檢查並更新 study 的轉換狀態
        """
        # 實現檢查邏輯...
        pass

    async def _validate_state_transition(
            self,
            dcop_event: DCOPEventRequest,
            session: AsyncSession
    ) -> None:
        """
        驗證狀態轉換的合法性

        Args:
            dcop_event: 要驗證的事件
            session: 資料庫 session

        Raises:
            InvalidStateTransitionError: 如果狀態轉換不合法
        """
        # 定義合法的狀態轉換
        VALID_TRANSITIONS = {
            DCOPStatus.SERIES_TRANSFERRING.value: [
                DCOPStatus.SERIES_TRANSFER_COMPLETE.value
            ],
            DCOPStatus.SERIES_CONVERTING.value: [
                DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
                DCOPStatus.SERIES_CONVERSION_SKIP.value
            ],
            DCOPStatus.STUDY_TRANSFERRING.value: [
                DCOPStatus.STUDY_TRANSFER_COMPLETE.value
            ],
            DCOPStatus.STUDY_CONVERTING.value: [
                DCOPStatus.STUDY_CONVERSION_COMPLETE.value
            ],
            # 添加更多合法轉換...
        }

        # 查詢當前狀態
        stmt = select(DCOPEventModel).where(
            and_(
                DCOPEventModel.study_uid == dcop_event.study_uid,
                DCOPEventModel.series_uid == dcop_event.series_uid
                if dcop_event.series_uid else DCOPEventModel.series_uid.is_(None),
                DCOPEventModel.tool_id == dcop_event.tool_id
            )
        ).order_by(DCOPEventModel.create_time.desc()).limit(1)

        result = await session.execute(stmt)
        current_event = result.scalar_one_or_none()

        if current_event:
            current_status = current_event.ope_no
            new_status = dcop_event.ope_no

            # 檢查是否為合法轉換
            valid_next_states = VALID_TRANSITIONS.get(current_status, [])
            if new_status not in valid_next_states and current_status != new_status:
                raise InvalidStateTransitionError(
                    f"Invalid state transition from {current_status} to {new_status} "
                    f"for study_uid: {dcop_event.study_uid}"
                )

    async def _trigger_external_checks(
            self,
            check_urls_with_data: Dict[str, Set[Tuple[str, str]]]
    ) -> List[str]:
        """
        觸發外部檢查

        Args:
            check_urls_with_data: 包含 URL 和相關數據的字典

        Returns:
            List[str]: 成功觸發的 URL 列表
        """
        triggered_urls = []

        # 使用並發請求提高效率
        async with httpx.AsyncClient(timeout=30) as client:
            tasks = []

            for url, study_data in check_urls_with_data.items():
                # 準備請求數據
                if "transfer_complete" in url:
                    # 只發送 study_uid 列表
                    request_data = {
                        "study_uid_list": [uid for uid, _ in study_data]
                    }
                else:
                    # 發送完整數據
                    request_data = {
                        "study_data": [
                            {"study_uid": uid, "study_id": sid}
                            for uid, sid in study_data
                        ]
                    }

                task = self._send_check_request(
                    client, url, request_data
                )
                tasks.append((url, task))

            # 並發執行所有請求
            for url, task in tasks:
                try:
                    await task
                    triggered_urls.append(url)
                    logger.info(f"Successfully triggered check at {url}")
                except Exception as e:
                    logger.error(f"Failed to trigger check at {url}: {e}")

        return triggered_urls

    async def _send_check_request(
            self,
            client: httpx.AsyncClient,
            url: str,
            data: dict
    ) -> None:
        """
        發送檢查請求

        Args:
            client: HTTP 客戶端
            url: 目標 URL
            data: 請求數據
        """
        try:
            response = await client.post(
                url,
                json=data,
                timeout=30
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            logger.error(f"Timeout when sending request to {url}")
            raise ExternalServiceError(f"Request timeout for {url}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} from {url}")
            raise ExternalServiceError(
                f"HTTP {e.response.status_code} from {url}"
            )
        except Exception as e:
            logger.error(f"Unexpected error when calling {url}: {e}")
            raise ExternalServiceError(f"Failed to call {url}: {str(e)}")

    async def _create_event_ope_no_in_session(
            self,
            dcop_event: DCOPEventRequest,
            session: AsyncSession
    ) -> DCOPEventModel:
        """在指定的 session 中創建事件"""
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

        session.add(new_data_obj)
        # 注意：不在這裡提交，讓 session_manager 統一管理
        await session.flush()  # 確保獲得生成的 ID
        return new_data_obj

    async def study_series_nifti_tool(self, data: List[DCOPEventNIFTITOOLRequest]):
        """
        使用統一的 session 管理處理 NIFTI 工具操作
        """
        from code_ai import load_dotenv
        load_dotenv()

        # 使用單一 session 處理所有操作
        async with self.session_manager.get_session() as session:
            for dcop in data:
                await self._process_nifti_tool_request(dcop, session)

        # 交易成功後，發送完成檢查請求
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        url = f"{upload_data_api_url}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"

        async with httpx.AsyncClient(timeout=180) as client:
            await client.post(url=url)

    async def _process_nifti_tool_request(
            self,
            dcop: DCOPEventNIFTITOOLRequest,
            session: AsyncSession
    ):
        """處理單個 NIFTI 工具請求"""
        match dcop.ope_no:
            case DCOPStatus.STUDY_CONVERTING.value:
                # 查詢相關數據
                conf_query = select(DCOPEventModel).where(
                    and_(
                        DCOPEventModel.study_id == dcop.study_id,
                        DCOPEventModel.study_uid.isnot(None),
                        DCOPEventModel.tool_id == 'DICOM_TOOL',
                        DCOPEventModel.ope_no == DCOPStatus.SERIES_TRANSFER_COMPLETE.value
                    )
                )

                result = await session.execute(conf_query)
                dcop_event = result.scalar_one_or_none()

                if not dcop_event:
                    raise ValueError(f"No matching DICOM_TOOL event found for study_id: {dcop.study_id}")

                # 創建新的事件記錄
                study_transfer_complete_data = await DCOPEventModel.create_event_ope_no(
                    tool_id=dcop.tool_id,
                    study_uid=dcop_event.study_uid,
                    series_uid=None,
                    study_id=dcop.study_id,
                    ope_no=dcop.ope_no,
                    result_data=dcop.result_data,
                    params_data=dcop.params_data,
                    session=session
                )

                session.add(study_transfer_complete_data)
                await session.flush()

                # 在同一個交易中處理 series 資訊
                await self.nifti_tool_get_series_info(dcop_event.study_uid, session)

            case DCOPStatus.SERIES_CONVERTING.value:
                # 處理 series 轉換邏輯
                pass

    async def nifti_tool_get_series_info(self, study_uid: str, session: AsyncSession):
        """
        在現有 session 中獲取並處理 series 資訊
        """
        from code_ai.task.task_dicom2nii import dicom_2_nii_series
        from code_ai.task.schema.intput_params import Dicom2NiiSeriesParams
        from code_ai import load_dotenv
        load_dotenv()

        path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
        path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")

        # 使用提供的 session 進行查詢
        sql = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid')
        results = await session.execute(
            sql,
            {
                'status': DCOPStatus.STUDY_CONVERTING.value,
                'study_uid': study_uid
            }
        )

        dcop_event_list = results.all()
        task_params_list = []
        dcop_model_list = []

        for dcop_event in dcop_event_list:
            result_data = dcop_event.result_data[0]
            output_dicom_path = result_data['rename_dicom_path']
            output_nifti_path = pathlib.Path(path_rename_nifti)

            task_params = Dicom2NiiSeriesParams(
                sub_dir=None,
                study_uid=dcop_event.study_uid,
                series_uid=dcop_event.series_uid,
                output_dicom_path=output_dicom_path,
                output_nifti_path=output_nifti_path
            )

            new_data_obj = await DCOPEventModel.create_event_ope_no(
                tool_id='NIFTI_TOOL',
                study_uid=dcop_event.study_uid,
                series_uid=dcop_event.series_uid,
                study_id=dcop_event.study_id,
                ope_no=DCOPStatus.SERIES_CONVERTING.value,
                result_data=dcop_event.result_data,
                params_data=task_params.get_str_dict(),
                session=session
            )

            dcop_model_list.append(new_data_obj)
            task_params_list.append(task_params)

        if dcop_model_list:
            # 批量添加到 session
            session.add_all(dcop_model_list)
            await session.flush()  # 確保所有對象都有 ID

            # 注意：只有在交易成功提交後才推送任務
            # 這裡我們先收集任務，稍後在交易成功後推送
            self._pending_tasks = task_params_list

    async def add_study_new(self, data_list: List[str]) -> List[DCOPEventModel]:
        """
        使用統一的 session 管理添加新的 study
        """
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv
        load_dotenv()

        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))

        result_list = []

        async with self.session_manager.get_session() as session:
            for study_uid in data_list:
                # 檢查是否已存在
                existing = await session.execute(
                    select(DCOPEventModel).where(
                        DCOPEventModel.study_uid == study_uid
                    )
                )

                if not existing.scalar_one_or_none():
                    # 創建新記錄
                    study_uid_raw_dicom_path = raw_dicom_path.joinpath(study_uid)

                    # 使用嵌套交易確保原子性
                    async with self.session_manager.transaction(session) as transaction:
                        new_data = await DCOPEventModel.create_event(
                            study_uid=study_uid,
                            series_uid=None,
                            status=DCOPStatus.STUDY_NEW.name,
                            session=session
                        )

                        session.add(new_data)

                        task_params = Dicom2NiiParams(
                            sub_dir=study_uid_raw_dicom_path,
                            output_dicom_path=rename_dicom_path,
                            output_nifti_path=rename_nifti_path
                        )

                        data_transferring = await DCOPEventModel.create_event(
                            study_uid=study_uid,
                            series_uid=None,
                            status=DCOPStatus.STUDY_TRANSFERRING.name,
                            session=session
                        )
                        data_transferring.params_data = task_params.get_str_dict()

                        session.add(data_transferring)
                        await session.flush()

                        result_list.append(new_data)

        return result_list
