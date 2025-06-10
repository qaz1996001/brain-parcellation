import json
import os
import pathlib
import shutil
from typing import List, Optional, Tuple
import re

import aiofiles.os
import httpx
from advanced_alchemy.extensions.fastapi import (
    repository,
    service,
)
from funboost import AsyncResult
from sqlalchemy import text, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio.engine import AsyncConnection, AsyncEngine
from backend.app.sync.model import DCOPEventModel
from backend.app.sync.schemas import DCOPStatus, OrthancID


class ReRunStudyService(service.SQLAlchemyAsyncRepositoryService[DCOPEventModel]):
    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        model_type = DCOPEventModel

    repository_type = Repo

    # pattern_str = '({}),({}),({}),({}),({}|{})'.format(DCOPStatus.SERIES_NEW.value,
    #                                                    DCOPStatus.SERIES_TRANSFERRING.value,
    #                                                    DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
    #                                                    DCOPStatus.SERIES_CONVERTING.value,
    #                                                    DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
    #                                                    DCOPStatus.SERIES_CONVERSION_SKIP.value
    #                                                    )
    # can_inference_pattern = re.compile(pattern_str)
    async def del_study_result_by_rename_id(self, data_list=List[str]) -> Optional[str]:
        return ''

    async def re_run_by_study_uid(self, data_list=List[OrthancID]) -> Optional[str]:

        await self.del_study_result_by_study_uid(data_list=data_list)
        return ''

    async def del_study_result_by_study_uid(self, data_list=List[OrthancID]) -> Optional[str]:
        process_path   = pathlib.Path(os.getenv("PATH_PROCESS"))
        aneurysm_path  = process_path.joinpath('Deep_Aneurysm')
        cmb_path       = process_path.joinpath('Deep_CMB')
        cmd_tools_path = process_path.joinpath('Deep_cmd_tools')
        infarct_path   = process_path.joinpath('Deep_Infarct')
        synthseg_path  = process_path.joinpath('Deep_synthseg')
        wmh_path       = process_path.joinpath('Deep_WMH')
        json_path      = pathlib.Path(os.getenv("PATH_JSON"))
        log_path       = pathlib.Path(os.getenv("PATH_LOG"))
        raw_dicom_path      = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
        rename_dicom_path   = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
        rename_nifti_path   = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))

        engine: AsyncEngine = self.repository.session.bind
        study_uid = data_list[0]
        async with engine.connect() as conn:
            sql = text('SELECT * FROM public.get_stydy_ope_no_status(:status) where study_uid=:study_uid')
            execute = await conn.execute(sql,
                                         {'status': DCOPStatus.STUDY_RESULTS_SENT.value,
                                          'study_uid': study_uid})
            result = execute.first()
        print('results',result)

        study_aneurysm_path     = aneurysm_path.joinpath(result.study_id)
        study_cmb_path          = cmb_path.joinpath(result.study_id)
        study_cmd_tools_path    = cmd_tools_path.joinpath(result.study_id)
        study_infarct_path      = infarct_path.joinpath(result.study_id)
        study_synthseg_path     = synthseg_path.joinpath(result.study_id)
        study_wmh_path          = wmh_path.joinpath(result.study_id)
        study_rename_dicom_path = rename_dicom_path.joinpath(result.study_id)
        study_rename_nifti_path = rename_nifti_path.joinpath(result.study_id)

        async for input_path in self.async_path_generator((study_aneurysm_path,study_cmb_path,study_cmd_tools_path,
                                                           study_infarct_path,study_synthseg_path,study_wmh_path,
                                                           study_rename_dicom_path,study_rename_nifti_path)):
            await self.del_path(input_path)

        # for params_data in result.params_data:
        #     print('params_data',params_data)
        return ''
    @staticmethod
    async def del_path(input_path :pathlib.Path):
        if input_path.is_file() and input_path.exists():
            # await aiofiles.os.remove(input_path)
            print('remove')
        elif input_path.is_dir() and input_path.exists():
            print('rmtree')
            # shutil.rmtree(input_path,ignore_errors=True)
        else:
            pass
    @staticmethod
    async def async_path_generator(paths):
        """異步路徑生成器"""
        for path in paths:
            yield path