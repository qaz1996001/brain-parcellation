# app/sync/routers.py
from typing import Annotated, List
from fastapi import APIRouter, Depends, Response, BackgroundTasks
from advanced_alchemy.extensions.fastapi import (
    service,
)

from backend.app.sync.service import DCOPEventDicomService
from backend.app.sync.schemas import DCOPEventRequest, PostStudyRequest

from . import urls
from .service import ReRunStudyService
from ..database import alchemy

router = APIRouter()


@router.post(
    urls.RERUN_PROT_STUDY_RENAME_ID,
    status_code=200,
    summary="根據 STUDY Rename ID 對Study 重跑整個流程  ",
    description="",
    response_description="",
    response_model=service.OffsetPagination[DCOPEventRequest],
)
async def post_re_run_study_by_study_rename_id(
    data_list: List[str],
    re_event_service: Annotated[
        ReRunStudyService, Depends(alchemy.provide_service(ReRunStudyService))
    ],
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
) -> Response:
    background_tasks.add_task(
        re_event_service.re_run_by_study_rename_id, data_list, dcop_event_service
    )
    return Response("")


@router.post(
    urls.RERUN_PROT_STUDY_UID,
    status_code=200,
    summary="根據 STUDY UID 對Study 重跑整個流程  ",
    description="",
    response_description="",
    response_model=service.OffsetPagination[DCOPEventRequest],
)
async def post_re_run_study_by_study_uid(
    request: PostStudyRequest,
    re_event_service: Annotated[
        ReRunStudyService, Depends(alchemy.provide_service(ReRunStudyService))
    ],
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
) -> Response:
    background_tasks.add_task(
        re_event_service.re_run_by_study_uid, request.ids, dcop_event_service
    )

    return Response("")
