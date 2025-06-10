# app/sync/routers.py
import os
import pathlib
from typing import Annotated, Tuple, List, Optional, Any, Coroutine
from advanced_alchemy.extensions.fastapi.providers import FieldNameType
from advanced_alchemy.service import OffsetPagination
from fastapi import APIRouter, Depends, Response, BackgroundTasks, Body, Query
from advanced_alchemy.extensions.fastapi import (service, filters,)

from backend.app.sync.service import DCOPEventDicomService
from backend.app.sync.schemas import DCOPEventRequest,OrthancID,OrthancIDRequest
from backend.app.sync.model import DCOPEventModel

from . import urls
from .service import ReRunStudyService
from ..database import alchemy

router = APIRouter()



@router.post(urls.RERUN_PROT_STUDY_RENAME_ID, status_code=200,
            summary="根據 STUDY Rename ID 對Study 重跑整個流程  ",
            description="",
            response_description="",
            response_model=service.OffsetPagination[DCOPEventRequest])
async def post_re_run_study_by_study_uid(request:OrthancIDRequest,
                                         re_event_service: Annotated[ReRunStudyService,
                                                                      Depends(alchemy.provide_service(ReRunStudyService))],
                                         background_tasks: BackgroundTasks) -> Response:
    path_process_path   = pathlib.Path(os.getenv("PATH_PROCESS"))
    path_cmd_tools_path = path_process_path.joinpath('Deep_cmd_tools')
    path_json_path      = pathlib.Path(os.getenv("PATH_JSON"))
    path_log_path       = pathlib.Path(os.getenv("PATH_LOG"))
    raw_dicom_path      = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
    rename_dicom_path   = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
    rename_nifti_path   = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))

    return Response('')


@router.post(urls.RERUN_PROT_STUDY_UID, status_code=200,
            summary="根據 STUDY UID 對Study 重跑整個流程  ",
            description="",
            response_description="",
            response_model=service.OffsetPagination[DCOPEventRequest])
async def post_re_run_study_by_study_uid(request:OrthancIDRequest,
                                         re_event_service: Annotated[ReRunStudyService,
                                                                      Depends(alchemy.provide_service(ReRunStudyService))],
                                         background_tasks: BackgroundTasks) -> Response:
    background_tasks.add_task(re_event_service.re_run_by_study_uid, request.ids)
    return Response('')