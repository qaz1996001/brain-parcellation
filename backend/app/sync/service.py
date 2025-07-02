import json
import logging
import os
import pathlib
import traceback
from typing import List, Optional, Tuple, Dict, Any
import re
import httpx
from advanced_alchemy.extensions.fastapi import repository
from advanced_alchemy.service import OffsetPagination
from funboost import AsyncResult
# from fastapi import
from sqlalchemy import text, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi_cache import FastAPICache

from code_ai.task.schema.intput_params import Dicom2NiiParams
from backend.app.service import BaseRepositoryService
from .model import DCOPEventModel
from .schemas import DCOPStatus, DCOPEventRequest, DCOPEventNIFTITOOLRequest, StydySeriesOpeNoStatus,OpeNo,OrthancID
from .urls import SYNC_PROT_OPE_NO, SYNC_PROT_STUDY_NIFTI_TOOL, SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID, \
    SYNC_PROT_STUDY_TRANSFER_COMPLETE

logger = logging.getLogger(__name__)


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
        logger.info(f'check_study_series_transfer_complete data {data}', )
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
                    session.add(new_data)
                    # new_data_obj = await self.create(data=new_data)
                    task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
                                                  output_dicom_path=rename_dicom_path,
                                                  output_nifti_path=rename_nifti_path, )
                    data_transferring = await DCOPEventModel.create_event(study_uid=ids,
                                                                          series_uid=None,
                                                                          status=DCOPStatus.STUDY_TRANSFERRING.name,
                                                                          session=session, )
                    data_transferring.params_data = task_params.get_str_dict()
                    session.add(data_transferring)
                    session.commit()
                    session.flush()
                    # obj = await self.create_many(data=[new_data_obj,data_transferring],auto_commit=True)
                    result_list.append(new_data)
                    result_list.append(data_transferring)
            except Exception as e:
                await session.rollback()
                logger.error(f"Error in add_study_new: {e}")
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
                logger.info(f'result {result}')
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
        logger.info(f'_send_events {event_data_list}')
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
        from code_ai.task.task_dicom2nii import dicom_2_nii_series
        from code_ai.task.schema.intput_params import Dicom2NiiSeriesParams
        from code_ai import load_dotenv
        load_dotenv()
        path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
        path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")
        # engine: AsyncEngine = session.bind
        # async with engine.connect() as conn:
        sql = text('SELECT * FROM public.get_stydy_series_ope_no_status(:status) where study_uid=:study_uid')
        results = await session.execute(sql,
                                        {'status': DCOPStatus.STUDY_CONVERTING.value,
                                         'study_uid': study_uid})

        dcop_event_list = results.all()
        task_params_list = []
        dcop_model_list = []
        for dcop_event in dcop_event_list:
            result_data = dcop_event.result_data[0]
            output_dicom_path = result_data['rename_dicom_path']
            output_nifti_path = pathlib.Path(path_rename_nifti)
            task_params = Dicom2NiiSeriesParams(sub_dir=None,
                                                study_uid=dcop_event.study_uid,
                                                series_uid=dcop_event.series_uid,
                                                output_dicom_path=output_dicom_path,
                                                output_nifti_path=output_nifti_path)
            new_data_obj = await DCOPEventModel.create_event_ope_no(tool_id='NIFTI_TOOL',
                                                                    study_uid=dcop_event.study_uid,
                                                                    series_uid=dcop_event.series_uid,
                                                                    study_id=dcop_event.study_id,
                                                                    ope_no=DCOPStatus.SERIES_CONVERTING.value,
                                                                    result_data=dcop_event.result_data,
                                                                    params_data=task_params.get_str_dict(),
                                                                    session=session)
            dcop_model_list.append(new_data_obj)
            task_params_list.append(task_params)

        try:
            if dcop_event_list:  # Check if dcop_event_list is not empty/falsy
                # Attempt to create many records and auto-commit
                # If create_many raises an exception, the 'except' block will catch it,
                # and the push operations will not be executed.
                # data_obj = await self.create_many(dcop_model_list, auto_commit=True)
                logger.info(f'session.add_all {dcop_model_list}', )
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
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))

        for dcop_event in data:
            study_uid = dcop_event.study_uid
            logger.info(f'dicom_tool_get_series_info dcop_event {dcop_event}')
            study_uid_raw_dicom_path = raw_dicom_path.joinpath(study_uid)
            if study_uid_raw_dicom_path.exists():
                async with self.session_manager.get_session() as session:
                    series_uid_path_list = sorted(study_uid_raw_dicom_path.iterdir())
                    new_data_list = []
                    task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
                                                  output_dicom_path=rename_dicom_path,
                                                  output_nifti_path=rename_nifti_path, )

                    for series_uid_path in series_uid_path_list:
                        series_new_data = await DCOPEventModel.create_event(study_uid=study_uid,
                                                                            series_uid=series_uid_path.name,
                                                                            status=DCOPStatus.SERIES_NEW.name,
                                                                            session=session, )
                        series_transferring_data = await DCOPEventModel.create_event(study_uid=study_uid,
                                                                                     series_uid=series_uid_path.name,
                                                                                     status=DCOPStatus.SERIES_TRANSFERRING.name,
                                                                                     session=session, )
                        series_transferring_data.params_data = task_params.get_str_dict()
                        new_data_list.append(series_new_data)
                        new_data_list.append(series_transferring_data)
                    try:
                        session.add_all(new_data_list)
                        await session.commit()
                        await session.flush()
                        task = dicom_to_nii.push(task_params.get_str_dict())
                        logger.info(f'dicom_tool_get_series_info {new_data_list} {task}')
                    except:
                        await session.rollback()
                        logger.error(traceback.print_exc())
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
        logger.info(f'data {data}')
        # post_check_study_series_conversion_complete call check
        if data is None:
            # Query studies not yet at STUDY_CONVERSION_COMPLETE status
            completed_studies = await self.query_studies_pending_completion()
        else:
            completed_studies = set()
            for dcop_enent in data:
                result_set = await self.query_studies_pending_completion(dcop_enent.study_uid)
                logger.info(f'result_set {result_set}')
                completed_studies.update(result_set)

        if not completed_studies:
            return None
        logger.info(f'completed_studies {completed_studies}', )
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
            logger.info('match_result  {} , {}'.format(match_result, test_str))
            logger.info("{} {}".format(self.pattern_str, self.can_inference_pattern.findall(test_str)))
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

        logger.info(f'result_set {study_data_list}', )
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
                logger.info(
                    f"Skipping duplicate inference task for study_id: {dcop_event.study_id}. Already in cache.")
                continue  # Skip this study_event and move to the next one

            # Push to inference task pipeline
            task_pipeline_result: AsyncResult = task_pipeline_inference.push(
                dcop_event_inference_ready.params_data
            )
            await redis_client.set(inference_task_key, "queued", ex=21600)  # Value can be anything, key is what matters
            logger.info(f"Added study_uid: {dcop_event.study_uid} to cache with key: {inference_task_key}")

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