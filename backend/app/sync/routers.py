# app/sync/routers.py
import logging
import os
import pathlib
from typing import Annotated, Tuple, List, Optional, Any
from advanced_alchemy.extensions.fastapi.providers import FieldNameType
from advanced_alchemy.service import OffsetPagination
from fastapi import APIRouter, Depends, Response, BackgroundTasks, Body, Query
from fastapi_cache import FastAPICache
from advanced_alchemy.extensions.fastapi import (service, filters,)
from sqlalchemy import Select, text
from sqlalchemy.engine.row import Row

from backend.app.sync import urls
from .service import DCOPEventDicomService
from .model import DCOPEventModel
from .schemas import DCOPStatus, DCOPEventRequest, OrthancID, DCOPEventNIFTITOOLRequest, PostStudyRequest, OpeNo, \
    StydySeriesOpeNoStatus

from ..database import alchemy

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(urls.SYNC_PROT_STUDY, status_code=200,
            summary="查詢已同步的 Study UUID ",
            description="",
            response_description="",)
async def get_study_uuid() -> Response:
    return Response("DICOM Service is running")


@router.post(urls.SYNC_PROT_STUDY, status_code=200,
             summary="更新同步的 Study UUID",
             description="傳入 STUDY_TRANSFER_COMPLETE Study UUID",
             response_description="",)
async def post_study_uuid(request:PostStudyRequest,
                          dcop_event_service: Annotated[DCOPEventDicomService,
                          Depends(alchemy.provide_service(DCOPEventDicomService))],
                          background_tasks: BackgroundTasks
                          ) -> Response:
    # logger.info(f'1000000000 request {request}')
    # return request

    result_list = await dcop_event_service.add_study_new(data_list=request.ids)
    logger.info(f'1000000000 result_list {result_list}')
    background_tasks.add_task(dcop_event_service.dicom_tool_get_series_info,result_list)
    return result_list



@router.get(urls.SYNC_PROT_OPE_NO, status_code=200,
             summary="根據 study series UUID ope_no 查詢資料",
             description="根據 study series UUID ope_no 查詢資料",
             response_description="",
             response_model=service.OffsetPagination[DCOPEventRequest])
async def get_ope_no(dcop_event_service: Annotated[DCOPEventDicomService,
                                                   Depends(alchemy.provide_service(DCOPEventDicomService))],
                     filters: Annotated[list[filters.FilterTypes],
                                        Depends(alchemy.provide_filters(
                                            {
                                                "id_filter": OrthancID,
                                                "pagination_type": "limit_offset",
                                                "search": "study_uid,study_id,ope_no,tool_id",
                                                "search_ignore_case": True,
                                            }
                                        )),]) -> OffsetPagination[DCOPEventModel]:
    logger.info(f'filters {filters}',)
    results, total = await dcop_event_service.list_and_count(*filters)
    return dcop_event_service.to_schema(results, total, filters=filters)
    # return service.OffsetPagination[DCOPEventRequest]



@router.post(urls.SYNC_PROT_OPE_NO, status_code=200,
             summary="更新同步的 study series UUID ope_no",
             description="傳入  study series UUID ope_no  ",
             response_description="",)
async def post_ope_no(data :List[DCOPEventRequest],
                      dcop_event_service: Annotated[DCOPEventDicomService,
                      Depends(alchemy.provide_service(DCOPEventDicomService))],
                      background_tasks: BackgroundTasks
                      ) -> Response:

    background_tasks.add_task(dcop_event_service.post_ope_no_task,data)
    return Response("post_ope_no")


@router.post(urls.SYNC_PROT_STUDY_TRANSFER_COMPLETE,
             status_code=200,
             summary="檢查 study series dicom transfer complete",
             description="",
             response_description="",)
async def post_check_study_series_transfer_complete(background_tasks   : BackgroundTasks,
                                                    dcop_event_service : Annotated[DCOPEventDicomService,
                                                                         Depends(alchemy.provide_service(DCOPEventDicomService))],
                                                    dcop_event_list    : Optional[List[DCOPEventRequest]] = Body(default=None),
                                                    ) -> Response:

    if dcop_event_list is None:
        background_tasks.add_task(dcop_event_service.check_study_series_transfer_complete)
    else:
        background_tasks.add_task(dcop_event_service.check_study_series_transfer_complete,dcop_event_list)
    return Response("post_check_study_series_transfer_complete")



@router.post(urls.SYNC_PROT_STUDY_NIFTI_TOOL,
             status_code=200,
             summary="study series NIFTI TOOL STUDY_CONVERSION -> SERIES_CONVERTING -> SERIES_CONVERSION_COMPLETE -> STUDY_CONVERSION_COMPLETE",
             description="",
             response_description="",)
async def post_study_series_nifti_tool(data_list :List[DCOPEventNIFTITOOLRequest],
                                       dcop_event_service: Annotated[DCOPEventDicomService,
                                       Depends(alchemy.provide_service(DCOPEventDicomService))],
                                       background_tasks: BackgroundTasks) -> Response:
    background_tasks.add_task(dcop_event_service.study_series_nifti_tool,data_list)
    return Response("post_study_nifti_tool")




@router.post(urls.SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID,
             status_code=200,
             summary="檢查 study series nifti conversion complete",
             description="",
             response_description="",)
async def post_check_study_series_conversion_complete(dcop_event_service: Annotated[DCOPEventDicomService,
                                                                          Depends(alchemy.provide_service(DCOPEventDicomService))],
                                                      background_tasks: BackgroundTasks,
                                                      dcop_event_list    : Optional[List[DCOPEventRequest]] = Body(default=None),) -> Response:
    if dcop_event_list is None:
        background_tasks.add_task(dcop_event_service.check_study_series_conversion_complete)
    else:
        background_tasks.add_task(dcop_event_service.check_study_series_conversion_complete,dcop_event_list)
    return Response("post_check_study_series_conversion_complete")


@router.post(urls.SYNC_PROT_STUDY_CONVERSION_COMPLETE_RENAME_ID,
             status_code=200,
             summary="檢查 study series nifti conversion complete",
             description="study rename id list",
             response_description="",)
async def post_check_study_series_conversion_complete(dcop_event_service: Annotated[DCOPEventDicomService,
                                                                          Depends(alchemy.provide_service(DCOPEventDicomService))],
                                                      background_tasks: BackgroundTasks,
                                                      study_id_list    : Optional[List[str]] = Body(default=None),) -> Response:
    if study_id_list is None:
        return Response("post_check_study_series_conversion_complete")
    else:
        session = dcop_event_service.repository.session
        statement = Select(DCOPEventModel.study_uid.distinct(),DCOPEventModel.study_id).where(DCOPEventModel.study_id.in_(study_id_list))
        async with session:
            execute = await session.execute(statement)
            results:List[Row] = list(execute.all())
            logger.info(f'results {results}',)
        dcop_event_list = [DCOPEventRequest(study_uid=result[0],
                                            study_id=result[1],
                                            ope_no=DCOPStatus.SERIES_CONVERSION_COMPLETE.value) for result in results]
        background_tasks.add_task(dcop_event_service.check_study_series_conversion_complete,dcop_event_list)
    return Response("post_check_study_series_conversion_complete")


# Advanced Alchemy 多條件搜索優化範例

# 方法3: 使用複雜的過濾器組合
@router.get("/events/complex",
            status_code=200,
            summary="複雜多條件搜索",
            response_model=service.OffsetPagination[DCOPEventRequest])
async def get_events_complex(
        dcop_event_service: Annotated[DCOPEventDicomService,
        Depends(alchemy.provide_service(DCOPEventDicomService))],
        filters_list: Annotated[list[filters.FilterTypes],
        Depends(alchemy.provide_filters({
            # 多欄位搜索
            "search": "params_data,result_data",
            "search_ignore_case": True,

            # 日期範圍過濾器
            "created_at": "before_after",

            # 集合過濾器
            "in_fields":[FieldNameType(name='tool_id',type_hint=str),
                         FieldNameType(name='ope_no',type_hint=str),
                         FieldNameType(name='study_uid',type_hint=str),
                         FieldNameType(name='study_id',type_hint=str),
                         FieldNameType(name='series_uid',type_hint=str),
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



@router.get("/cache",
            status_code=200,
            summary="cache")
async def get_events_complex():
    redis_backend = FastAPICache.get_backend()
    redis_client = redis_backend.redis
    cached_keys = await redis_client.keys("inference_task:*")
    logger.warning(f"cached_keys {cached_keys}")
    study_uids = []
    for key in cached_keys:
        # 從鍵中提取 study_uid。鍵的格式為 "fastapi-cache:inference_task:<study_uid>"
        # Extract study_uid from the key. Key format is "fastapi-cache:inference_task:<study_uid>"
        try:
            # Assuming key is bytes, decode it first
            if isinstance(key,str):
                key_str = key
            else:
                key_str = key.decode('utf-8')
            parts = key_str.split(':')
            if parts[0] == "inference_task":
                study_uids.append(parts[1].split(','))
        except Exception as e:
            logger.info(f"Error parsing cache key {key}: {e}")

    return study_uids


@router.delete("/cache",
               status_code=200,
               summary="cache")
async def delete_events_complex(study_id:Optional[str]        = Query(None),
                                study_uid:Optional[OrthancID] = Query(None)):
    redis_backend = FastAPICache.get_backend()
    redis_client = redis_backend.redis

    if not study_id and not study_uid:
        return {"error": "Either study_id or study_uid must be provided"}

    # 取得所有快取鍵
    cached_keys = await redis_client.keys("inference_task:*")

    deleted_keys = []

    for key in cached_keys:
        try:
            # 確保鍵是字串格式
            if isinstance(key, str):
                key_str = key
            else:
                key_str = key.decode('utf-8')

            # 解析鍵格式: "inference_task:<study_uid>,<study_id>"
            parts = key_str.split(':')
            logger.info('parts {}'.format(parts))
            if len(parts) >= 2 and parts[0] == "inference_task":
                # 取得 study_uid 和 study_id 部分
                uid_id_part = parts[1]

                # 分割 study_uid 和 study_id
                if ',' in uid_id_part:
                    cached_study_uid, cached_study_id = uid_id_part.split(',', 1)

                    # 檢查是否符合刪除條件
                    should_delete = False
                    if study_uid and cached_study_uid == study_uid:
                        should_delete = True
                    elif study_id and cached_study_id == study_id:
                        should_delete = True

                    if should_delete:
                        # 刪除快取
                        await redis_client.delete(key)
                        deleted_keys.append(key_str)
                        logger.info(f"Deleted cache key: {key_str}")

        except Exception as e:
            logger.error(f"Error processing cache key {key}: {e}")

    return {
        "message": f"Cache deletion completed",
        "deleted_keys": deleted_keys,
        "deleted_count": len(deleted_keys),
        "search_criteria": {
            "study_id": study_id,
            "study_uid": study_uid
        }
    }





@router.get("/query/study_series_ope_no_status",
            status_code=200,
            summary="study_series_ope_no_status")
async def get_study_series_ope_no_status(dcop_event_service: Annotated[DCOPEventDicomService,
                                                           Depends(alchemy.provide_service(DCOPEventDicomService))],
                                         limit: int = Query(20, ge=1),
                                         offset: int = Query(0, ge=0),
                    study_uid:Optional[OrthancID] = Query(None),
                    ope_no:OpeNo = Query(...)):
    result = await dcop_event_service.get_stydy_series_ope_no_status(study_uid=study_uid,
                                                                     ope_no=ope_no,
                                                                     limit=limit,
                                                                     offset=offset, )
    return result


@router.get("/query/stydy_ope_no_status",
            status_code=200,
            summary="stydy_ope_no_status")
async def get_stydy_ope_no_status(dcop_event_service: Annotated[DCOPEventDicomService,
                                                           Depends(alchemy.provide_service(DCOPEventDicomService))],
                                  study_uid:Optional[OrthancID] = Query(None),
                                  ope_no:OpeNo = Query(...),
                                  limit: int = Query(20, ge=1),
                                  offset: int = Query(0, ge=0),
                                  ) -> OffsetPagination[StydySeriesOpeNoStatus]:
    result = await dcop_event_service.get_stydy_ope_no_status(study_uid=study_uid,
                                                              ope_no=ope_no,
                                                              limit=limit,
                                                              offset=offset,)
    return result



@router.get("/query/check_study_series_conversion_complete",
            status_code=200,
            summary="check_study_series_conversion_complete")
async def get_check_study_series_conversion_complete(dcop_event_service: Annotated[DCOPEventDicomService,
                                                     Depends(alchemy.provide_service(DCOPEventDicomService))],
                                                     study_uid: Optional[str] = Query(None)):
    result = await dcop_event_service.get_check_study_series_conversion_complete(study_uid=study_uid)
    return result

