import json
import os
import pathlib
import shutil
import traceback
from typing import List, Optional, Tuple
import re

import aiofiles.os
import httpx
from advanced_alchemy.extensions.fastapi import (
    repository,
    service,
)
from advanced_alchemy.filters import LimitOffset
from funboost import AsyncResult
from sqlalchemy import text
from sqlalchemy.ext.asyncio.engine import AsyncEngine
from backend.app.sync.model import DCOPEventModel
from backend.app.sync.schemas import DCOPStatus, OrthancID
from backend.app.sync.service import DCOPEventDicomService
from backend.app.sync.urls import SYNC_PROT_OPE_NO
from code_ai import load_dotenv


class ReRunStudyService(service.SQLAlchemyAsyncRepositoryService[DCOPEventModel]):
    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        model_type = DCOPEventModel

    repository_type = Repo

    insert_sql = text("""insert into dcop_event_bth (vsprimarykey, tool_id, study_uid, series_uid, study_id,
                                                     event_cate, code_name, code_desc,params_data, result_data, ope_no,
                                                     ope_name, claim_time, rec_time, create_time, update_time)
                        (select vsprimarykey, tool_id, study_uid, series_uid, study_id,
                                         1, code_name, code_desc,params_data, result_data, ope_no,
                                         ope_name, claim_time, rec_time, create_time, update_time from dcop_event_bt where study_uid=:study_uid)
                      """)
    delete_sql = text("""delete from dcop_event_bt where vsprimarykey in (select vsprimarykey from dcop_event_bt where study_uid=:study_uid)
                      """)

    # pattern_str = '({}),({}),({}),({}),({}|{})'.format(DCOPStatus.SERIES_NEW.value,
    #                                                    DCOPStatus.SERIES_TRANSFERRING.value,
    #                                                    DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
    #                                                    DCOPStatus.SERIES_CONVERTING.value,
    #                                                    DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
    #                                                    DCOPStatus.SERIES_CONVERSION_SKIP.value
    #                                                    )
    # can_inference_pattern = re.compile(pattern_str)


    async def get_study_new_re_model(self, study_uid:str) :
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv
        load_dotenv()
        raw_dicom_path = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))


        study_uid_raw_dicom_path = raw_dicom_path.joinpath(study_uid)
        new_data_re = await DCOPEventModel.create_event(study_uid=study_uid,
                                                     series_uid=None,
                                                     status=DCOPStatus.STUDY_NEW_RE.name,
                                                     session=self.repository.session, )
        task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
                                      output_dicom_path=rename_dicom_path,
                                      output_nifti_path=rename_nifti_path, )
        data_transferring_re = await DCOPEventModel.create_event(study_uid=study_uid,
                                                              series_uid=None,
                                                              status=DCOPStatus.STUDY_TRANSFERRING_RE.name,
                                                              session=self.repository.session, )
        data_transferring_re.params_data = task_params.get_str_dict()
        return new_data_re, data_transferring_re,task_params



    async def re_run_by_study_rename_id(self, data_list:List[str],
                                        dcop_event_service:DCOPEventDicomService) -> Optional[str]:
        for study_id in data_list:
            models = await self.list(DCOPEventModel.study_id==study_id,LimitOffset(limit=10,offset=0))
            print('models',models)
            if len(models) > 0:
                print('model',models[0])
                await self.del_study_result_by_field(field_name='study_uid',
                                                     field_value=models[0].study_uid)

                new_data_re, data_transferring_re, task_params = await self.get_study_new_re_model(models[0].study_uid)
                result  = await self.create_many([new_data_re,data_transferring_re],auto_commit=True)
                result1 = await dcop_event_service.dicom_tool_get_series_info([new_data_re])
                print('result',result)
                print('result1', result1)
        return

    async def re_run_by_study_uid(self, data_list=List[OrthancID]) -> Optional[str]:
        for study_uid in data_list:
            await self.del_study_result_by_field(field_name='study_uid',
                                                 field_value=study_uid)
            # await self._send_events(study_uid=study_uid)
            # await self.del_study_result_by_study_uid(study_uid=study_uid)
            # await self._send_events(study_uid=study_uid)
        return ''

    async def del_study_result_by_field(self, field_name:str,field_value:str) -> Optional[str]:
        sql = text(f'SELECT * FROM public.get_stydy_ope_no_status(:status) where {field_name}=:{field_name}')
        parameters = {'status': DCOPStatus.STUDY_RESULTS_SENT.value,
                      field_name: field_value}
        await self.del_study_result_by_parameters(sql=sql, parameters=parameters)
        return ''

    async def del_study_result_by_study_uid(self, study_uid:str) -> Optional[str]:

        sql = text('SELECT * FROM public.get_stydy_ope_no_status(:status) where study_uid=:study_uid')
        parameters = {'status': DCOPStatus.STUDY_RESULTS_SENT.value,
                      'study_uid': study_uid}
        await self.del_study_result_by_parameters(sql=sql,parameters=parameters)

        return


    async def del_study_result_by_parameters(self, sql: text, parameters: dict):
        load_dotenv()
        process_path = pathlib.Path(os.getenv("PATH_PROCESS"))
        aneurysm_path = process_path.joinpath('Deep_Aneurysm')
        cmb_path = process_path.joinpath('Deep_CMB')
        cmd_tools_path = process_path.joinpath('Deep_cmd_tools')
        infarct_path = process_path.joinpath('Deep_Infarct')
        synthseg_path = process_path.joinpath('Deep_synthseg')
        wmh_path = process_path.joinpath('Deep_WMH')
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
        engine: AsyncEngine = self.repository.session.bind
        async with engine.connect() as conn:
            execute = await conn.execute(sql, parameters)
            result = execute.first()
            if result is not None:
                study_aneurysm_path = aneurysm_path.joinpath(result.study_id)
                study_cmb_path = cmb_path.joinpath(result.study_id)
                study_cmd_tools_path = cmd_tools_path.joinpath(result.study_id)
                study_infarct_path = infarct_path.joinpath(result.study_id)
                study_synthseg_path = synthseg_path.joinpath(result.study_id)
                study_wmh_path = wmh_path.joinpath(result.study_id)
                study_rename_dicom_path = rename_dicom_path.joinpath(result.study_id)
                study_rename_nifti_path = rename_nifti_path.joinpath(result.study_id)
                async for input_path in self.async_path_generator(
                        [study_aneurysm_path, study_cmb_path, study_cmd_tools_path,
                         study_infarct_path, study_synthseg_path, study_wmh_path,
                         study_rename_dicom_path, study_rename_nifti_path]):
                    await self.del_path(input_path)
                try:
                    if conn.in_transaction():
                        insert_execute = await conn.execute(self.insert_sql, {'study_uid': result.study_uid})
                        delete_execute = await conn.execute(self.delete_sql, {'study_uid': result.study_uid})

                    await conn.commit()
                    print('insert_execute', insert_execute)
                    print('delete_execute', delete_execute)
                except:
                    await conn.rollback()
                    traceback.print_exc()
        return

    @staticmethod
    async def _send_events(event_data: List[dict]) -> None:
        """
        Sends study transfer complete events to the API.

        Args:
            api_url: Base URL for the upload data API.
            event_data: List of serialized DCOPEventRequest objects.
        """
        upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
        async with httpx.AsyncClient(timeout=180) as client:
            url = f"{upload_data_api_url}{SYNC_PROT_OPE_NO}"
            event_data_json = json.dumps(event_data)
            await client.post(url=url, timeout=180, data=event_data_json)

    @staticmethod
    async def del_path(input_path :pathlib.Path):
        print('input_path',input_path)
        if input_path.is_file() and input_path.exists():
            await aiofiles.os.remove(input_path)
            print('remove')
        elif input_path.is_dir() and input_path.exists():
            print('rmtree')
            shutil.rmtree(input_path,ignore_errors=True)
        else:
            pass

    @staticmethod
    async def async_path_generator(paths):
        """異步路徑生成器"""
        for path in paths:
            yield path