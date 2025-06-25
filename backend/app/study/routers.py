# app/study/routers.py
import logging
from datetime import datetime
from typing import Annotated, Tuple, List, Optional, Any, Coroutine

from advanced_alchemy.extensions.fastapi.providers import FieldNameType
from advanced_alchemy.service import OffsetPagination
from fastapi import APIRouter, Depends, Response, BackgroundTasks, Body, Query
from fastapi_cache import FastAPICache
from advanced_alchemy.extensions.fastapi import service, filters
from sqlalchemy import Select

from backend.app.study import urls
from backend.app.sync.service import DCOPEventDicomService
from backend.app.sync.model import DCOPEventModel
from backend.app.sync.schemas import DCOPStatus, DCOPEventRequest, OrthancID
from .deps import provide_filters
from ..database import alchemy

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(urls.STUDY_GET_LIST,
            status_code=200,
            summary="複雜多條件搜索",
            response_model=service.OffsetPagination[DCOPEventRequest])
async def get_events_complex(
        dcop_event_service: Annotated[DCOPEventDicomService,
        Depends(alchemy.provide_service(DCOPEventDicomService))],
        filters_list: Annotated[list[filters.FilterTypes],
        Depends(provide_filters({
            # 多欄位搜索
            "search": "params_data,result_data",
            "search_ignore_case": True,

            # 集合過濾器
            "in_fields":[FieldNameType(name='tool_id',type_hint=str),
                         FieldNameType(name='ope_no',type_hint=str),
                         FieldNameType(name='study_uid',type_hint=str),
                         FieldNameType(name='study_id',type_hint=str),
                         FieldNameType(name='series_uid',type_hint=str),
                         ],
            # BeforeAfter 日期時間過濾器
            "before_after_fields": [
                FieldNameType(name='create_time', type_hint=datetime),
                FieldNameType(name='update_time', type_hint=datetime),
            ],

            # 排序配置
            "order_by": ["study_uid", "ope_no","create_time", ],

            # 分頁配置
            "pagination_type": "limit_offset",
            "limit": 50,
            "offset": 0,
            # ID 過濾器
            "id_filter": OrthancID,
        }))],
) -> service.OffsetPagination[DCOPEventModel]:
    """
    複雜多條件搜索範例：
    - 支援多種過濾器類型
    - 可以組合使用不同的過濾條件

    使用範例：
    ?search=keyword&status=active,pending&created_at_after=2024-01-01&order_by=created_at,-study_uid&limit=10
    """
    results, total = await dcop_event_service.list_and_count(*filters_list)
    return dcop_event_service.to_schema(results, total, filters=filters_list)

