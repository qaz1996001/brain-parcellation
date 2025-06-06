import json
import os
import pathlib
from typing import List, Optional

import httpx
from advanced_alchemy.extensions.fastapi import (
    repository,
    service,
)
# from fastapi import
from sqlalchemy import text, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio.engine import AsyncConnection, AsyncEngine

from code_ai.task.task_dicom2nii import dicom_to_nii
from code_ai.task.schema.intput_params import Dicom2NiiParams
from .model import DCOPEventModel
from .schemas import DCOPStatus, DCOPEventRequest,DCOPEventNIFTITOOLRequest
from .urls import SYNC_PROT_OPE_NO, SYNC_PROT_STUDY_NIFTI_TOOL


class DCOPEventDicomService(service.SQLAlchemyAsyncRepositoryService[DCOPEventModel]):
    """Author repository."""

    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        """Author repository."""

        model_type = DCOPEventModel

    repository_type = Repo


    async def post_ope_no_task(self,data :List[DCOPEventRequest]):
        from code_ai import load_dotenv
        load_dotenv()
        result_list = []
        for dcop_event in data:
            new_data = await DCOPEventModel.create_event_ope_no(tool_id    = dcop_event.tool_id,
                                                                study_uid  = dcop_event.study_uid,
                                                                series_uid = dcop_event.series_uid,
                                                                study_id   = dcop_event.study_id,
                                                                ope_no     = dcop_event.ope_no,
                                                                result_data= dcop_event.result_data,
                                                                params_data= dcop_event.params_data,
                                                                session    = self.repository.session)
            new_data_obj = await self.create(data=new_data)
            result_list.append(new_data_obj)
        try:
            data_obj = await self.create_many(result_list,auto_commit= True)
        except:
            await self.repository.session.rollback()
        return data_obj


    async def check_study_series_conversion_complete(self,data :Optional[List[DCOPEventRequest]] = None):
        """
            檢查 study 下的 series 是否都轉成 nii
            1. series 完成轉成nii ， 添加 SERIES_CONVERSION_COMPLETE  的記錄
            2. 所有series都到了 SERIES_CONVERSION_COMPLETE， 添加 STUDY_CONVERSION_COMPLETE 的記錄
            3. 添加 STUDY_INFERENCE_READY 的記錄，
            4. 發送管道任務  推論
        """
        pass

    async def check_study_series_transfer_complete(self, data: Optional[List[DCOPEventRequest]] = None):
        """
            檢查 study 下的 series 是否都傳輸完成
            1. series 完成傳輸添加 SERIES_TRANSFER_COMPLETE  的記錄
            2. 所有series都到了SERIES_TRANSFER_COMPLETE， 添加 STUDY_TRANSFER_COMPLETE 的記錄
            3. 添加 STUDY_CONVERTING 的記錄，
            4. 發送管道任務  進行轉換
        """
        from code_ai import load_dotenv
        load_dotenv()
        UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")

        engine :AsyncEngine = self.repository.session.bind
        print(engine)
        dcop_event_list = []
        dcop_event_dump_list = []
        if data is None:
            async with engine.connect() as conn:
                results = await conn.execute(text('select * from public.get_all_studies_status()'))
                for result in results.all():
                    print('result',result)
                    dcop_event = DCOPEventRequest(study_uid=result[0]['study_uid'],
                                                  series_uid=None,
                                                  ope_no=DCOPStatus.STUDY_TRANSFER_COMPLETE.value,
                                                  study_id=result[0]['study_id'],
                                                  tool_id='DICOM_TOOL',
                                                  result_data={'result':json.dumps(result[0]['result'])})
                    dcop_event_dump_list.append(dcop_event.model_dump())
                    dcop_event_list.append(dcop_event)
            async with httpx.AsyncClient(timeout=180) as client:
                url = "{}{}".format(UPLOAD_DATA_API_URL, SYNC_PROT_OPE_NO)
                dcop_event_list_json = json.dumps(dcop_event_dump_list)
                response = await client.post(url=url,timeout=180,data=dcop_event_list_json)
        if len(dcop_event_list) > 0:
            path_raw_dicom = os.getenv("PATH_RAW_DICOM")
            path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
            path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")
            url = "{}{}".format(UPLOAD_DATA_API_URL, SYNC_PROT_STUDY_NIFTI_TOOL)
            # DCOPStatus.STUDY_INFERENCE_COMPLETE
            for dcop_event in dcop_event_list:
                study_id = dcop_event.study_id
                output_dicom_path = pathlib.Path(os.path.join(path_rename_dicom,study_id))
                output_nifti_path = pathlib.Path(path_rename_nifti)
                task_params = Dicom2NiiParams(sub_dir=None,
                                              output_dicom_path= output_dicom_path,
                                              output_nifti_path= output_nifti_path )
                async with httpx.AsyncClient(timeout=180) as client:

                    dcop_event_nifti_tool = DCOPEventNIFTITOOLRequest(ope_no=DCOPStatus.STUDY_CONVERTING,
                                                                      study_id=study_id,
                                                                      tool_id='NIFTI_TOOL',
                                                                      params_data=task_params.get_str_dict(),
                                                                      result_data=None)
                    dcop_event_nifti_tool_json = json.dumps([dcop_event_nifti_tool.model_dump(),])
                    response = await client.post(url=url,  data=dcop_event_nifti_tool_json)

                # dicom_to_nii.push(task_params.get_str_dict())

    async def study_series_nifti_tool(self,data :List[DCOPEventNIFTITOOLRequest]):
        """
        建立
           DCOPStatus.STUDY_CONVERTING
           DCOPStatus.SERIES_CONVERTING
           DCOPStatus.SERIES_CONVERSION_COMPLETE
           DCOPStatus.STUDY_CONVERSION_COMPLETE
        """

        from code_ai import load_dotenv
        load_dotenv()
        session:AsyncSession = self.repository.session
        result_list = []
        for dcop in data:
            match dcop.ope_no:
                case DCOPStatus.STUDY_CONVERTING.value:
                    async with session:
                        conf_query = select(DCOPEventModel, ).where(and_(*[DCOPEventModel.study_id == dcop.study_id,
                                                                           DCOPEventModel.study_uid.isnot(None),
                                                                           DCOPEventModel.tool_id == 'DICOM_TOOL',
                                                                           DCOPEventModel.ope_no == DCOPStatus.STUDY_TRANSFER_COMPLETE.value]))
                        execute = await session.execute(conf_query)
                        dcop_event = execute.first()[0]
                    new_data = await DCOPEventModel.create_event_ope_no(tool_id=dcop.tool_id,
                                                                        study_uid=dcop_event.study_uid,
                                                                        series_uid=dcop_event.series_uid,
                                                                        study_id=dcop.study_id,
                                                                        ope_no=dcop.ope_no,
                                                                        result_data=dcop.result_data,
                                                                        params_data=dcop.params_data,
                                                                        session=self.repository.session)
                    new_data_obj = await self.create(new_data, auto_commit=True)
                    await self.nifti_tool_get_series_info(dcop_event.study_uid)

        # try:
        #     data_obj = await self.create_many(result_list, auto_commit=True)
        # except:
        #     await self.repository.session.rollback()

        return

    async def nifti_tool_get_series_info(self, study_uid: str):
        from code_ai.task.task_dicom2nii import dicom_to_nii
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv
        load_dotenv()
        path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
        path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")

        async with session:
            conf_query = select(DCOPEventModel, ).where(and_(*[DCOPEventModel.study_id == dcop.study_id,
                                                               DCOPEventModel.study_uid.isnot(None),
                                                               DCOPEventModel.tool_id == 'DICOM_TOOL',
                                                               DCOPEventModel.ope_no == DCOPStatus.STUDY_TRANSFER_COMPLETE.value]))
        dcop_event_list =

        for dcop_event in dcop_event_list:
            study_id = dcop_event.study_id
            output_dicom_path = pathlib.Path(os.path.join(path_rename_dicom, study_id))
            output_nifti_path = pathlib.Path(path_rename_nifti)
            task_params = Dicom2NiiParams(sub_dir=None,
                                          output_dicom_path=output_dicom_path,
                                          output_nifti_path=output_nifti_path)
            dicom_to_nii.push(task_params.get_str_dict())



    async def dicom_tool_get_series_info(self, study_uid: str):
        from code_ai.task.task_dicom2nii import dicom_to_nii
        from code_ai.task.schema.intput_params import Dicom2NiiParams
        from code_ai import load_dotenv
        load_dotenv()
        raw_dicom_path    = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
        study_uid_raw_dicom_path = raw_dicom_path.joinpath(study_uid)
        if study_uid_raw_dicom_path.exists():
            series_uid_path_list = sorted(study_uid_raw_dicom_path.iterdir())
            new_data_list = []
            for series_uid_path in series_uid_path_list:
                series_new_data = await DCOPEventModel.create_event(study_uid=study_uid,
                                                                    series_uid=series_uid_path.name,
                                                                    status=DCOPStatus.SERIES_NEW.name,
                                                                    session=self.repository.session, )
                series_transferring_data = await DCOPEventModel.create_event(study_uid=study_uid,
                                                                             series_uid=series_uid_path.name,
                                                                             status=DCOPStatus.SERIES_TRANSFERRING.name,
                                                                             session=self.repository.session, )
                new_data_list.append(series_new_data)
                new_data_list.append(series_transferring_data)
            try:
                data_obj = await self.create_many(new_data_list, auto_commit=True)
            except:
                await self.repository.session.rollback()
            task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
                                          output_dicom_path=rename_dicom_path,
                                          output_nifti_path=rename_nifti_path, )

            task = dicom_to_nii.push(task_params.get_str_dict())
        return task