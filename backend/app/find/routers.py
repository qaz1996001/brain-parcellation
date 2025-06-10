# app/sync/routers.py
import os
import pathlib
from typing import Annotated, Tuple, List, Optional, Any, Coroutine
from advanced_alchemy.extensions.fastapi.providers import FieldNameType
from advanced_alchemy.service import OffsetPagination
from fastapi import APIRouter, Depends, Response, BackgroundTasks, Body, Query
from advanced_alchemy.extensions.fastapi import (service, filters,)

from backend.app.sync.service import DCOPEventDicomService
from backend.app.sync.schemas import DCOPEventRequest,OrthancID
from backend.app.sync.model import DCOPEventModel

from . import urls
from ..database import alchemy

router = APIRouter()


@router.get(urls.FIND_GET_STUDY_FINISH, status_code=200,
            summary="查詢已完成的 Study",
            description="",
            response_description="",
            response_model=service.OffsetPagination[DCOPEventRequest])
async def get_study_finsh(dcop_event_service: Annotated[DCOPEventDicomService,
                                                        Depends(alchemy.provide_service(DCOPEventDicomService))],
                          filters_list: Annotated[list[filters.FilterTypes],
                                                  Depends(alchemy.provide_filters({
                                                   # 日期範圍過濾器
                                                   "created_at": "before_after",

                                                  # 集合過濾器
                                                   "in_fields":[FieldNameType(name='tool_id',type_hint=str),
                                                                FieldNameType(name='ope_no',type_hint=str),
                                                                FieldNameType(name='study_uid',type_hint=str),
                                                                FieldNameType(name='study_id',type_hint=str),
                                                                ],
                                                   # 排序配置
                                                   "order_by": ["study_uid", "ope_no","create_time", ],
                                                   # 分頁配置
                                                   "pagination_type": "limit_offset",
                                                   "limit": 50,
                                                   # ID 過濾器
                                                      "id_filter": OrthancID,}))]
                          ) -> service.OffsetPagination[DCOPEventModel]:
    print('filters_list', filters_list)
    results, total = await dcop_event_service.list_and_count(*filters_list)
    print('results', results)
    print('total', total)
    return dcop_event_service.to_schema(results, total, filters=filters_list)
