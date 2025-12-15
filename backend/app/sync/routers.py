"""FastAPI 路由：處理 Study/Series 同步、NIFTI 轉檔、快取工具等 API。"""

# app/sync/routers.py
import logging
from typing import Annotated, List, Optional
from advanced_alchemy.extensions.fastapi.providers import FieldNameType
from advanced_alchemy.service import OffsetPagination
from fastapi import (
    APIRouter,
    Depends,
    Response,
    BackgroundTasks,
    Body,
    Query,
    HTTPException,
)
from fastapi_cache import FastAPICache
from advanced_alchemy.extensions.fastapi import (
    service,
    filters,
)
from sqlalchemy import Select
from sqlalchemy.engine.row import Row

from backend.app.sync import urls
from .service import DCOPEventDicomService
from .model import DCOPEventModel
from .schemas import (
    DCOPStatus,
    DCOPEventRequest,
    OrthancID,
    DCOPEventNIFTITOOLRequest,
    PostStudyRequest,
    OpeNo,
    StydySeriesOpeNoStatus,
)

from ..database import alchemy

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    urls.SYNC_PROT_STUDY,
    status_code=200,
    summary="查詢已同步的 Study UUID ",
    description="",
    response_description="",
)
async def get_study_uuid() -> Response:
    """健康檢查端點，確認 DICOM sync service 仍活著。"""
    return Response("DICOM Service is running")


@router.post(
    urls.SYNC_PROT_STUDY,
    status_code=200,
    summary="更新同步的 Study UUID",
    description="傳入 STUDY_TRANSFER_COMPLETE Study UUID",
    response_description="",
)
async def post_study_uuid(
    request: PostStudyRequest,
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
) -> Response:
    """輸入 study uid 後即刻排程後台任務並回傳已建立的事件列表。"""
    study_ids = request.resolved_ids()
    if not study_ids:
        raise HTTPException(status_code=422, detail="缺少 study ids")

    result_list = await dcop_event_service.schedule_new_studies(study_ids)
    logger.info("post_study_uuid scheduled=%s", len(result_list))

    background_tasks.add_task(
        dcop_event_service.dicom_tool_get_series_info, result_list
    )

    link_target = request.study_uid or (study_ids[0] if len(study_ids) == 1 else None)
    if request.prev_study_uid and link_target:
        background_tasks.add_task(
            dcop_event_service.link_prev_study,
            link_target,
            request.prev_study_uid,
        )
    return result_list


@router.get(
    urls.SYNC_PROT_OPE_NO,
    status_code=200,
    summary="根據 study series UUID ope_no 查詢資料",
    description="根據 study series UUID ope_no 查詢資料",
    response_description="",
    response_model=service.OffsetPagination[DCOPEventRequest],
)
async def get_ope_no(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    filters: Annotated[
        list[filters.FilterTypes],
        Depends(
            alchemy.provide_filters(
                {
                    "id_filter": OrthancID,
                    "pagination_type": "limit_offset",
                    "search": "study_uid,study_id,ope_no,tool_id",
                    "search_ignore_case": True,
                }
            )
        ),
    ],
) -> OffsetPagination[DCOPEventModel]:
    """支援搜尋/分頁的 ope_no 查詢，便於人工追蹤任務狀態。"""
    logger.info(
        f"filters {filters}",
    )
    results, total = await dcop_event_service.list_and_count(*filters)
    return dcop_event_service.to_schema(results, total, filters=filters)
    # return service.OffsetPagination[DCOPEventRequest]


@router.post(
    urls.SYNC_PROT_OPE_NO,
    status_code=200,
    summary="更新同步的 study series UUID ope_no",
    description="傳入  study series UUID ope_no  ",
    response_description="",
)
async def post_ope_no(
    data: List[DCOPEventRequest],
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
) -> Response:
    """批次寫入 ope_no 事件並交給背景任務進行後續轉檔/檢查。"""
    background_tasks.add_task(dcop_event_service.post_ope_no_task, data)
    return Response("post_ope_no")


@router.post(
    urls.SYNC_PROT_STUDY_TRANSFER_COMPLETE,
    status_code=200,
    summary="檢查 study series dicom transfer complete",
    description="",
    response_description="",
)
async def post_check_study_series_transfer_complete(
    background_tasks: BackgroundTasks,
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    dcop_event_list: Optional[List[DCOPEventRequest]] = Body(default=None),
) -> Response:
    """觸發系列轉檔前檢查；可指定 payload 或讓後台自行掃描。"""
    if dcop_event_list is None:
        background_tasks.add_task(
            dcop_event_service.check_study_series_transfer_complete
        )
    else:
        background_tasks.add_task(
            dcop_event_service.check_study_series_transfer_complete, dcop_event_list
        )
    return Response("post_check_study_series_transfer_complete")


@router.post(
    urls.SYNC_PROT_STUDY_NIFTI_TOOL,
    status_code=200,
    summary="study series NIFTI TOOL STUDY_CONVERSION -> SERIES_CONVERTING -> SERIES_CONVERSION_COMPLETE -> STUDY_CONVERSION_COMPLETE",
    description="",
    response_description="",
)
async def post_study_series_nifti_tool(
    data_list: List[DCOPEventNIFTITOOLRequest],
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
) -> Response:
    """接受 NIFTI_TOOL 回報並更新 study/series 相關狀態。"""
    background_tasks.add_task(dcop_event_service.study_series_nifti_tool, data_list)
    return Response("post_study_nifti_tool")


@router.post(
    urls.SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID,
    status_code=200,
    summary="檢查 study series nifti conversion complete",
    description="",
    response_description="",
)
async def post_check_study_series_conversion_complete(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
    dcop_event_list: Optional[List[DCOPEventRequest]] = Body(default=None),
) -> Response:
    """確認整體轉檔是否完成，若已完成則排程推論。"""
    if dcop_event_list is None:
        background_tasks.add_task(
            dcop_event_service.check_study_series_conversion_complete
        )
    else:
        background_tasks.add_task(
            dcop_event_service.check_study_series_conversion_complete, dcop_event_list
        )
    return Response("post_check_study_series_conversion_complete")


@router.post(
    urls.SYNC_PROT_STUDY_CONVERSION_COMPLETE_RENAME_ID,
    status_code=200,
    summary="檢查 study series nifti conversion complete",
    description="study rename id list",
    response_description="",
)
async def post_check_study_series_conversion_complete(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
    study_id_list: Optional[List[str]] = Body(default=None),
) -> Response:
    """專供 rename study id 使用者查核轉檔狀態。"""
    if study_id_list is None:
        return Response("post_check_study_series_conversion_complete")
    else:
        session = dcop_event_service.repository.session
        statement = Select(
            DCOPEventModel.study_uid.distinct(), DCOPEventModel.study_id
        ).where(DCOPEventModel.study_id.in_(study_id_list))
        async with session:
            execute = await session.execute(statement)
            results: List[Row] = list(execute.all())
            logger.info(
                f"results {results}",
            )
        dcop_event_list = [
            DCOPEventRequest(
                study_uid=result[0],
                study_id=result[1],
                ope_no=DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
            )
            for result in results
        ]
        background_tasks.add_task(
            dcop_event_service.check_study_series_conversion_complete, dcop_event_list
        )
    return Response("post_check_study_series_conversion_complete")


# Advanced Alchemy 多條件搜索優化範例


# 方法3: 使用複雜的過濾器組合
@router.get(
    "/events/complex",
    status_code=200,
    summary="複雜多條件搜索",
    response_model=service.OffsetPagination[DCOPEventRequest],
)
async def get_events_complex(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    filters_list: Annotated[
        list[filters.FilterTypes],
        Depends(
            alchemy.provide_filters(
                {
                    # 多欄位搜索
                    "search": "params_data,result_data",
                    "search_ignore_case": True,
                    # 日期範圍過濾器
                    "created_at": "before_after",
                    # 集合過濾器
                    "in_fields": [
                        FieldNameType(name="tool_id", type_hint=str),
                        FieldNameType(name="ope_no", type_hint=str),
                        FieldNameType(name="study_uid", type_hint=str),
                        FieldNameType(name="study_id", type_hint=str),
                        FieldNameType(name="series_uid", type_hint=str),
                    ],
                    # 排序配置
                    "order_by": [
                        "study_uid",
                        "ope_no",
                        "create_time",
                    ],
                    # 分頁配置
                    "pagination_type": "limit_offset",
                    "limit": 50,
                    "offset": 0,
                    # ID 過濾器
                    "id_filter": OrthancID,
                }
            )
        ),
    ],
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


@router.get("/cache", status_code=200, summary="cache")
async def get_events_complex():
    """列出推論任務快取鍵，協助偵錯/觀察 queue 狀態。"""
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
            if isinstance(key, str):
                key_str = key
            else:
                key_str = key.decode("utf-8")
            parts = key_str.split(":")
            if parts[0] == "inference_task":
                study_uids.append(parts[1].split(","))
        except Exception as e:
            logger.info(f"Error parsing cache key {key}: {e}")

    return study_uids


@router.delete("/cache", status_code=200, summary="cache")
async def delete_events_complex(
    study_id: Optional[str] = Query(None), study_uid: Optional[OrthancID] = Query(None)
) -> dict:
    """從 redis 中刪除指定 study_id 或 study_uid 的快取。"""
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
                key_str = key.decode("utf-8")

            # 解析鍵格式: "inference_task:<study_uid>,<study_id>"
            parts = key_str.split(":")
            logger.info("parts {}".format(parts))
            if len(parts) >= 2 and parts[0] == "inference_task":
                # 取得 study_uid 和 study_id 部分
                uid_id_part = parts[1]

                # 分割 study_uid 和 study_id
                if "," in uid_id_part:
                    cached_study_uid, cached_study_id = uid_id_part.split(",", 1)

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
        "message": "Cache deletion completed",
        "deleted_keys": deleted_keys,
        "deleted_count": len(deleted_keys),
        "search_criteria": {"study_id": study_id, "study_uid": study_uid},
    }


@router.get(
    "/query/study_series_ope_no_status",
    status_code=200,
    summary="study_series_ope_no_status",
)
async def get_study_series_ope_no_status(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    limit: int = Query(20, ge=1),
    offset: int = Query(0, ge=0),
    study_uid: Optional[OrthancID] = Query(None),
    ope_no: OpeNo = Query(...),
    ):
    """查詢單一 study 下 series 的 ope_no 狀態分頁資料。"""
    result = await dcop_event_service.get_stydy_series_ope_no_status(
        study_uid=study_uid,
        ope_no=ope_no,
        limit=limit,
        offset=offset,
    )
    return result


@router.get(
    "/query/stydy_ope_no_status", status_code=200, summary="stydy_ope_no_status"
)
async def get_stydy_ope_no_status(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    study_uid: Optional[OrthancID] = Query(None),
    ope_no: OpeNo = Query(...),
    limit: int = Query(20, ge=1),
    offset: int = Query(0, ge=0),
) -> OffsetPagination[StydySeriesOpeNoStatus]:
    """查詢 study 維度的 ope_no 狀態並支援 limit/offset。"""
    result = await dcop_event_service.get_stydy_ope_no_status(
        study_uid=study_uid,
        ope_no=ope_no,
        limit=limit,
        offset=offset,
    )
    return result


@router.get(
    "/query/check_study_series_conversion_complete",
    status_code=200,
    summary="check_study_series_conversion_complete",
)
async def get_check_study_series_conversion_complete(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    study_uid: Optional[str] = Query(None),
):
    """查詢已完成轉檔的 study 列表，方便整合 UI 顯示。"""
    result = await dcop_event_service.get_check_study_series_conversion_complete(
        study_uid=study_uid
    )
    return result
