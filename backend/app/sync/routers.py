"""
FastAPI 路由：DICOM 同步系統的 REST API 端點。

此模組定義了 DICOM 同步系統的所有 REST API 端點，包括：
1. Study/Series 同步管理（POST/GET）
2. 狀態轉遷檢查和觸發（POST）
3. NIFTI 轉檔工具集成（POST）
4. 推論快取管理（GET/DELETE）
5. 查詢和報表 API（GET）

API 設計遵循以下原則：
- 無狀態：每個端點獨立處理
- 異步優先：使用 BackgroundTasks 處理長流程
- 分頁支持：複雜查詢支援 limit/offset
- 早期驗證：Pydantic 和 Annotated 提供類型檢查
- 清晰的狀態轉遷：使用 DCOPStatus 定義所有狀態

Endpoints Overview
==================

Study Management (傳輸階段):
    GET    /sync/study            - 健康檢查
    POST   /sync/study            - 排程新 Study 同步

Event Log Queries (查詢):
    GET    /sync/ope_no           - 查詢事件日誌（支援搜尋/分頁）
    POST   /sync/ope_no           - 批次寫入事件

State Transitions (狀態轉遷):
    POST   /sync/study/transfer   - 檢查 Study 傳輸是否完成
    POST   /sync/study/convert    - 檢查轉檔是否完成
    POST   /sync/nifti_tool       - 接收 NIFTI_TOOL 回報

Cache Management (快取管理):
    GET    /sync/cache            - 列出推論任務快取
    DELETE /sync/cache            - 清除指定 Study 快取

Status Queries (狀態查詢):
    GET    /sync/query/...        - 各種聚合查詢端點

Good Taste Design Principles Applied
====================================

1. 消除特殊情況：
   - 所有 Study 查詢使用統一的 PostStudyRequest
   - 無論新舊版本欄位，resolved_ids() 統一返回列表

2. 資料結構驅動：
   - 使用 DCOPStatus 列舉而非魔術字符串
   - 後台任務通過狀態碼驅動流程轉遷

3. 扁平 API 設計：
   - 無深度嵌套的端點路徑
   - 清晰的職責分離

4. Early Return 原則：
   - 立即驗證輸入，拋出異常
   - 避免深層條件判斷

Notes
-----
此模組依賴 DCOPEventDicomService 來實現業務邏輯。
路由只負責：
  1. 輸入驗證和轉換
  2. 依賴注入和背景任務排程
  3. 輸出序列化
"""

import logging
from typing import Annotated, List, Optional, cast
from advanced_alchemy.extensions.fastapi.providers import FieldNameType, FilterConfig
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
from .settings import get_sync_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    urls.SYNC_PROT_STUDY,
    status_code=200,
    summary="健康檢查：確認 DICOM sync service 正常運作",
    description="簡單的健康檢查端點，用於監控和負載平衡器探測",
    response_description="服務狀態消息",
    tags=["Health Check"],
)
async def get_study_uuid() -> Response:
    """
    健康檢查端點。
    
    此端點用於確認 DICOM sync 服務是否正常運作。
    通常被監控系統或負載平衡器定期調用。
    
    Returns
    -------
    Response
        固定回傳 "DICOM Service is running"
    
    Examples
    --------
    定期監控調用：
    
    >>> requests.get("http://api.server/sync/study")
    Response(..., text="DICOM Service is running")
    """
    return Response("DICOM Service is running")


@router.post(
    urls.SYNC_PROT_STUDY,
    status_code=200,
    summary="排程新 Study 同步",
    description="接收 Study UID，建立同步事件，排程後台任務",
    response_description="已建立的事件列表",
    response_model=List[DCOPEventRequest],
    tags=["Study Management"],
)
async def post_study_uuid(
    request: PostStudyRequest,
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
) -> List[DCOPEventRequest]:
    """
    排程新 Study 同步，進入 Study 傳輸階段。
    
    此端點接收 Study UID，立即建立初始同步事件（STUDY_NEW），
    然後排程後台任務進行後續處理：
    1. 從 Orthanc 擷取 Series 資訊
    2. 建立 Study 時序鏈結（若有指定前驅）
    
    Parameters
    ----------
    request : PostStudyRequest
        包含 Study UID（新版或舊版格式）和可選的前驅 Study UID。
    dcop_event_service : DCOPEventDicomService
        由依賴注入提供的服務實例。
    background_tasks : BackgroundTasks
        FastAPI 提供的後台任務管理器。
    
    Returns
    -------
    Response
        已建立的事件列表。
    
    Raises
    ------
    HTTPException
        422 Unprocessable Entity：未提供 Study UID。
    
    Examples
    --------
    基本用法（新版本）：
    
    >>> req = {
    ...     "study_uid": "abc-123"
    ... }
    >>> response = await post_study_uuid(
    ...     PostStudyRequest(**req),
    ...     service,
    ...     tasks
    ... )
    
    支援時序鏈結：
    
    >>> req = {
    ...     "study_uid": "followup-001",
    ...     "prev_study_uid": "baseline-001"
    ... }
    >>> response = await post_study_uuid(
    ...     PostStudyRequest(**req),
    ...     service,
    ...     tasks
    ... )
    
    Notes
    -----
    此端點不阻塞等待後台任務完成，立即返回結果。
    後續狀態轉遷通過查詢 event log 追蹤。
    
    Flow:
        1. 驗證輸入（自動通過 Pydantic）
        2. 正規化 Study ID 列表
        3. 建立 STUDY_NEW 事件
        4. 排程後台任務
        5. 立即返回
    """
    # 正規化輸入，消除舊新版本差異
    study_ids = request.resolved_ids()
    
    # Early validation：缺少必要資訊立即拋出
    if not study_ids:
        raise HTTPException(
            status_code=422,
            detail="缺少 Study UID（需提供 ids 或 study_uid 之一）"
        )

    # 建立初始事件並返回
    # Convert OrthancID list to Optional[str] list
    study_ids_str: List[Optional[str]] = [str(uid) if uid is not None else None for uid in study_ids]
    result_list = await dcop_event_service.schedule_new_studies(study_ids_str)
    logger.info("post_study_uuid scheduled=%s", len(result_list))

    # 排程後台任務：擷取 Series 資訊
    background_tasks.add_task(
        dcop_event_service.dicom_tool_get_series_info, result_list
    )

    # 若指定了前驅 Study，建立時#序鏈結
    #link_target = request.study_uid or (study_ids[0] if len(study_ids) == 1 else None)
    #if request.prev_study_uid and link_target:
     #   background_tasks.add_task(
         #   dcop_event_service.link_prev_study,
       #     link_target,
    #        request.prev_study_uid,
     #   )
    # 將資料庫模型轉換為 Pydantic schema（from_attributes=True）
    return [DCOPEventRequest.model_validate(event) for event in result_list]


@router.get(
    urls.SYNC_PROT_OPE_NO,
    status_code=200,
    summary="查詢事件日誌（支援搜尋和分頁）",
    description="查詢 DICOM 同步事件，支援多條件搜尋、排序和分頁",
    response_description="分頁的事件列表",
    response_model=service.OffsetPagination[DCOPEventRequest],
    tags=["Event Queries"],
)
async def get_ope_no(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    filters: Annotated[
        list[filters.FilterTypes],
        Depends(
            alchemy.provide_filters(
                cast(
                    FilterConfig,
                    {
                        "id_filter": str,  # type: ignore[dict-item]
                        "pagination_type": "limit_offset",
                        "search": "study_uid,study_id,ope_no,tool_id",
                        "search_ignore_case": True,
                    },
                )
            )
        ),
    ],
) -> OffsetPagination[DCOPEventModel]:
    """
    查詢事件日誌，支援多條件搜尋和分頁。
    
    此端點用於查詢 DICOM 同步系統的事件日誌，是追蹤 Study/Series
    狀態轉遷的主要方式。支援複雜的搜尋條件、排序和分頁。
    
    Query Parameters
    ----------------
    search : str, optional
        關鍵詞搜尋，作用於多個欄位：
        - study_uid：DICOM Study UID
        - study_id：醫院 HIS Study ID
        - ope_no：操作編號
        - tool_id：工具代碼
        
        例如：?search=abc-123
    
    limit : int, default=20
        每頁結果數量（1-100）。
        例如：?limit=50
    
    offset : int, default=0
        分頁偏移量。
        例如：?offset=100
    
    order_by : str, optional
        排序欄位，支援多個欄位和反向排序。
        例如：?order_by=create_time,-study_uid
    
    Returns
    -------
    OffsetPagination[DCOPEventRequest]
        分頁結果，包含事件列表和總計數。
    
    Examples
    --------
    查詢特定 Study 的所有事件：
    
    >>> GET /sync/ope_no?search=abc-123&limit=20
    
    查詢特定工具的事件（降序）：
    
    >>> GET /sync/ope_no?search=NIFTI_TOOL&order_by=-create_time
    
    分頁查詢：
    
    >>> GET /sync/ope_no?limit=50&offset=100&order_by=study_uid
    
    Notes
    -----
    此端點由 Advanced Alchemy 的 provide_filters 系統提供
    通用的搜尋和分頁能力。所有查詢都是大小寫不敏感的。
    """
    logger.info(f"filters {filters}")
    
    # 執行查詢和計數
    results, total = await dcop_event_service.list_and_count(*filters)
    
    # 轉換為 Pydantic schema 並返回分頁結果
    return dcop_event_service.to_schema(results, total, filters=filters)


@router.post(
    urls.SYNC_PROT_OPE_NO,
    status_code=200,
    summary="批次寫入事件紀錄",
    description="接收事件列表，寫入資料庫，排程後續處理",
    response_description="已排程任務的確認消息",
    tags=["Event Management"],
)
async def post_ope_no(
    data: List[DCOPEventRequest],
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
) -> Response:
    """
    批次寫入事件紀錄並排程後續檢查。
    
    此端點接收來自外部系統（如 Orthanc、NIFTI_TOOL 等）的事件列表，
    立即排程後台任務處理：
    1. 寫入事件到資料庫
    2. 檢查狀態轉遷條件
    3. 觸發下一階段流程（例如轉檔檢查）
    
    Parameters
    ----------
    data : list[DCOPEventRequest]
        待寫入的事件列表。支援批次大小無上限。
    dcop_event_service : DCOPEventDicomService
        由依賴注入提供的服務實例。
    background_tasks : BackgroundTasks
        FastAPI 後台任務管理器。
    
    Returns
    -------
    Response
        簡單的確認消息。
    
    Examples
    --------
    批次寫入 Series 傳輸完成事件：
    
    >>> events = [
    ...     {
    ...         "study_uid": "abc-123",
    ...         "series_uid": "def-456",
    ...         "ope_no": "100.095",
    ...         "tool_id": "DICOM_TOOL"
    ...     },
    ...     {
    ...         "study_uid": "abc-123",
    ...         "series_uid": "ghi-789",
    ...         "ope_no": "100.095",
    ...         "tool_id": "DICOM_TOOL"
    ...     }
    ... ]
    >>> await post_ope_no(events, service, tasks)
    
    Notes
    -----
    此端點適用於外部系統批次上報事件的場景。
    不阻塞等待後台任務完成，立即返回。
    
    Flow:
        1. 接收事件列表（通過 Pydantic 驗證）
        2. 排程後台任務
        3. 立即返回確認
        4. 後台任務異步處理：寫入、檢查、轉遷
    """
    # 排程後台任務進行事件寫入和狀態檢查
    background_tasks.add_task(dcop_event_service.post_ope_no_task, data)
    
    return Response("post_ope_no")


@router.post(
    urls.SYNC_PROT_STUDY_TRANSFER_COMPLETE,
    status_code=200,
    summary="檢查 Study/Series 傳輸是否完成",
    description="檢查條件並觸發下一階段（進入轉檔）",
    response_description="已排程檢查任務的確認消息",
    tags=["State Transitions"],
)
async def post_check_study_series_transfer_complete(
    background_tasks: BackgroundTasks,
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    dcop_event_list: Optional[List[DCOPEventRequest]] = Body(default=None),
) -> Response:
    """
    檢查 Study/Series 傳輸是否完成，若完成則進入轉檔階段。
    
    此端點用於狀態轉遷：檢查所有 Series 是否已傳輸完成，
    若完成則建立 STUDY_CONVERTING 事件，進入 NIFTI 轉檔階段。
    
    Parameters
    ----------
    dcop_event_list : list[DCOPEventRequest], optional
        指定要檢查的事件列表。
        若為 None，則掃描資料庫中所有待檢查的 Study/Series。
    background_tasks : BackgroundTasks
        FastAPI 後台任務管理器。
    dcop_event_service : DCOPEventDicomService
        由依賴注入提供的服務實例。
    
    Returns
    -------
    Response
        已排程檢查任務的確認消息。
    
    Examples
    --------
    自動掃描所有待檢查 Study：
    
    >>> await post_check_study_series_transfer_complete(tasks, service)
    
    檢查指定的 Study：
    
    >>> events = [{"study_uid": "abc-123", "ope_no": "100.095"}]
    >>> await post_check_study_series_transfer_complete(
    ...     tasks, service, dcop_event_list=events
    ... )
    
    Notes
    -----
    這是一個 "推動狀態轉遷" 的 checkpoint API。
    Orthanc 系統完成傳輸後呼叫此端點，系統檢查並推進到轉檔階段。
    
    State Transition:
        SERIES_TRANSFER_COMPLETE → [檢查] → STUDY_CONVERTING
    
    Algorithm:
        1. 收集指定或掃描的 Study
        2. 對每個 Study，檢查其所有 Series 是否已完成傳輸
        3. 若全部完成，建立 STUDY_CONVERTING 事件
        4. 排程 NIFTI 轉檔工具執行
    """
    # 根據是否提供事件列表，選擇掃描或指定檢查
    if dcop_event_list is None:
        # 自動掃描資料庫中所有待檢查的 Study
        background_tasks.add_task(
            dcop_event_service.check_study_series_transfer_complete
        )
    else:
        # 檢查指定的 Study/Series
        background_tasks.add_task(
            dcop_event_service.check_study_series_transfer_complete, dcop_event_list
        )
    
    return Response("post_check_study_series_transfer_complete")


@router.post(
    urls.SYNC_PROT_STUDY_NIFTI_TOOL,
    status_code=200,
    summary="接收 NIFTI_TOOL 回報",
    description="接收外部 NIFTI 轉檔工具的執行結果",
    response_description="已排程更新任務的確認消息",
    tags=["Tool Integration"],
)
async def post_study_series_nifti_tool(
    data_list: List[DCOPEventNIFTITOOLRequest],
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
) -> Response:
    """
    接收 NIFTI_TOOL 回報並更新 Study/Series 狀態。
    
    NIFTI_TOOL 是獨立的外部服務，負責 DICOM to NIFTI 轉檔。
    執行完成後通過此端點回報結果，系統更新狀態並檢查轉檔是否全部完成。
    
    Parameters
    ----------
    data_list : list[DCOPEventNIFTITOOLRequest]
        NIFTI_TOOL 回報的結果列表。包含 ope_no、result_data 等。
    dcop_event_service : DCOPEventDicomService
        由依賴注入提供的服務實例。
    background_tasks : BackgroundTasks
        FastAPI 後台任務管理器。
    
    Returns
    -------
    Response
        已排程更新任務的確認消息。
    
    Examples
    --------
    NIFTI_TOOL 回報成功轉檔：
    
    >>> results = [
    ...     {
    ...         "ope_no": "200.195",
    ...         "tool_id": "NIFTI_TOOL",
    ...         "study_id": "HIS_001",
    ...         "result_data": {
    ...             "output_path": "/data/nifti/result.nii.gz",
    ...             "success": True
    ...         }
    ...     }
    ... ]
    >>> await post_study_series_nifti_tool(results, service, tasks)
    
    Notes
    -----
    NIFTI_TOOL 是鬆散耦合的外部系統，只提供 ope_no 和結果資料。
    系統根據 ope_no 反向查詢找到對應的 Study/Series，更新狀態。
    
    State Transition Flow:
        STUDY_CONVERTING → SERIES_CONVERTING → 
        [NIFTI_TOOL 執行] → SERIES_CONVERSION_COMPLETE → 
        [全部完成檢查] → STUDY_CONVERSION_COMPLETE
    
    Design:
        - 最小化外部工具的相依性
        - 使用 ope_no 進行反向查詢而非直接 API 呼叫
        - 支援批次處理多個轉檔結果
    """
    # 排程後台任務處理 NIFTI_TOOL 回報
    background_tasks.add_task(dcop_event_service.study_series_nifti_tool, data_list)
    
    return Response("post_study_nifti_tool")


@router.post(
    urls.SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID,
    status_code=200,
    summary="檢查 Study/Series NIFTI 轉檔是否完成",
    description="檢查條件並觸發下一階段（進入推論）",
    response_description="已排程檢查任務的確認消息",
    tags=["State Transitions"],
)
async def post_check_study_series_conversion_complete(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    background_tasks: BackgroundTasks,
    dcop_event_list: Optional[List[DCOPEventRequest]] = Body(default=None),
) -> Response:
    """
    檢查 Study/Series NIFTI 轉檔是否完成，若完成則進入推論階段。
    
    此端點用於狀態轉遷：檢查所有 Series 是否已完成轉檔，
    若完成則建立 STUDY_INFERENCE_READY 事件，進入推論準備階段。
    
    Parameters
    ----------
    dcop_event_list : list[DCOPEventRequest], optional
        指定要檢查的事件列表。
        若為 None，則掃描資料庫中所有待檢查的 Study/Series。
    dcop_event_service : DCOPEventDicomService
        由依賴注入提供的服務實例。
    background_tasks : BackgroundTasks
        FastAPI 後台任務管理器。
    
    Returns
    -------
    Response
        已排程檢查任務的確認消息。
    
    Examples
    --------
    自動掃描所有待檢查 Study：
    
    >>> await post_check_study_series_conversion_complete(service, tasks)
    
    檢查指定的 Study：
    
    >>> events = [{"study_uid": "abc-123", "ope_no": "200.195"}]
    >>> await post_check_study_series_conversion_complete(
    ...     service, tasks, dcop_event_list=events
    ... )
    
    Notes
    -----
    這是 "推動狀態轉遷" 的第二個 checkpoint API。
    NIFTI_TOOL 完成轉檔後呼叫此端點，系統檢查並推進到推論階段。
    
    State Transition:
        SERIES_CONVERSION_COMPLETE → [檢查] → STUDY_INFERENCE_READY
    
    Algorithm:
        1. 收集指定或掃描的 Study
        2. 對每個 Study，檢查其所有 Series 是否已完成轉檔
        3. 若全部完成，建立 STUDY_INFERENCE_READY 事件
        4. 排程推論工具執行
    """
    # 根據是否提供事件列表，選擇掃描或指定檢查
    if dcop_event_list is None:
        # 自動掃描資料庫中所有待檢查的 Study
        background_tasks.add_task(
            dcop_event_service.check_study_series_conversion_complete
        )
    else:
        # 檢查指定的 Study/Series
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
async def post_check_study_series_conversion_complete_by_id(
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
                cast(
                    FilterConfig,
                    {
                        # 多欄位搜索
                        "search": "params_data,result_data",
                        "search_ignore_case": True,
                        # 日期範圍過濾器
                        "created_at": True,  # type: ignore[dict-item]
                        # 集合過濾器
                        "in_fields": [
                            FieldNameType(name="tool_id", type_hint=str),
                            FieldNameType(name="ope_no", type_hint=str),
                            FieldNameType(name="study_uid", type_hint=str),
                            FieldNameType(name="study_id", type_hint=str),
                            FieldNameType(name="series_uid", type_hint=str),
                        ],
                        # 排序配置
                        "order_by": [  # type: ignore[dict-item]
                            "study_uid",
                            "ope_no",
                            "create_time",
                        ],
                        # 分頁配置
                        "pagination_type": "limit_offset",
                        "limit": 50,  # type: ignore[dict-item]
                        "offset": 0,  # type: ignore[dict-item]
                        # ID 過濾器
                        "id_filter": str,  # type: ignore[dict-item]
                    },
                )
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


@router.get(
    "/cache",
    status_code=200,
    summary="列出推論任務快取",
    description="查詢 Redis 中的推論任務快取鍵",
    response_description="活躍的推論任務列表",
    tags=["Cache Management"],
)
async def get_inference_cache():
    """
    列出推論任務快取鍵，用於偵錯和觀察推論隊列狀態。
    
    此端點連接 Redis 快取層，列出所有活躍的推論任務。
    對於監控和偵錯非常有用：可以觀察有多少 Study 正在等待或執行推論。
    
    Returns
    -------
    list
        推論任務列表，每項包含 [study_uid, study_id]。
    
    Examples
    --------
    查詢活躍的推論任務：
    
    >>> GET /sync/cache
    >>> [
    ...     ["abc-123", "HIS_001"],
    ...     ["def-456", "HIS_002"]
    ... ]
    
    Notes
    -----
    此端點用於操作監控和偵錯。可視為系統的 "推論隊列觀測窗口"。
    """
    redis_backend = FastAPICache.get_backend()
    redis_client = redis_backend.redis  # type: ignore[attr-defined]
    cache_prefix = get_sync_settings().cache.inference_prefix
    
    # 從 Redis 取得所有推論任務快取鍵
    cached_keys = await redis_client.keys(f"{cache_prefix}:*")
    logger.warning(f"cached_keys {cached_keys}")
    
    study_uids = []
    for key in cached_keys:
        # 快取鍵格式："{cache_prefix}:{study_uid},{study_id}"
        try:
            # 確保鍵是字符串格式（可能是 bytes）
            if isinstance(key, str):
                key_str = key
            else:
                key_str = key.decode("utf-8")
            
            # 解析鍵格式
            parts = key_str.split(":")
            if len(parts) >= 2 and parts[0] == cache_prefix:
                # 提取 study_uid 和 study_id
                uid_pair = parts[1].split(",")
                study_uids.append(uid_pair)
                
        except Exception as e:
            logger.info(f"Error parsing cache key {key}: {e}")

    return study_uids


@router.delete(
    "/cache",
    status_code=200,
    summary="清除推論任務快取",
    description="根據 study_id 或 study_uid 刪除 Redis 快取",
    response_description="刪除結果和統計",
    tags=["Cache Management"],
)
async def delete_events_complex(
    study_id: Optional[str] = Query(None),
    study_uid: Optional[OrthancID] = Query(None)
) -> dict:
    """
    清除指定 Study 的推論任務快取。
    
    此端點用於手動清除快取，適用於以下場景：
    1. 推論任務失敗需要重試
    2. 需要重新排隊執行
    3. 快取異常需要清理
    
    Parameters
    ----------
    study_id : str, optional
        醫院 HIS System 的 Study ID。
        若提供，刪除匹配此 ID 的所有快取。
    study_uid : OrthancID, optional
        DICOM Study UID。
        若提供，刪除匹配此 UID 的所有快取。
    
    Returns
    -------
    dict
        刪除結果，包含：
        - message: 操作完成消息
        - deleted_keys: 已刪除的快取鍵列表
        - deleted_count: 刪除數量
        - search_criteria: 搜尋條件
    
    Examples
    --------
    根據 Study UID 刪除：
    
    >>> DELETE /sync/cache?study_uid=abc-123
    >>> {
    ...     "message": "Cache deletion completed",
    ...     "deleted_keys": ["inference_task:abc-123,HIS_001"],
    ...     "deleted_count": 1,
    ...     "search_criteria": {"study_id": null, "study_uid": "abc-123"}
    ... }
    
    根據 Study ID 刪除：
    
    >>> DELETE /sync/cache?study_id=HIS_001
    
    Notes
    -----
    - 至少需要提供 study_id 或 study_uid 其中之一
    - 刪除操作會立即反映到 Redis
    - 刪除快取後需重新排隊推論任務
    
    Error Handling:
        若未提供任何搜尋條件，返回錯誤消息。
    """
    redis_backend = FastAPICache.get_backend()
    redis_client = redis_backend.redis  # type: ignore[attr-defined]
    cache_prefix = get_sync_settings().cache.inference_prefix

    # 早期驗證：至少提供一種搜尋條件
    if not study_id and not study_uid:
        return {"error": "需要提供 study_id 或 study_uid 其中之一"}

    # 取得所有推論任務快取鍵
    cached_keys = await redis_client.keys(f"{cache_prefix}:*")
    deleted_keys = []

    for key in cached_keys:
        try:
            # 確保鍵是字符串格式（可能是 bytes）
            if isinstance(key, str):
                key_str = key
            else:
                key_str = key.decode("utf-8")

            # 解析快取鍵格式："{cache_prefix}:{study_uid},{study_id}"
            parts = key_str.split(":")
            logger.info("parts {}".format(parts))
            
            if len(parts) >= 2 and parts[0] == cache_prefix:
                # 提取 study_uid 和 study_id
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
                        # 從 Redis 刪除快取鍵
                        await redis_client.delete(key)
                        deleted_keys.append(key_str)
                        logger.info(f"Deleted cache key: {key_str}")

        except Exception as e:
            logger.error(f"Error processing cache key {key}: {e}")

    return {
        "message": "快取刪除完成",
        "deleted_keys": deleted_keys,
        "deleted_count": len(deleted_keys),
        "search_criteria": {"study_id": study_id, "study_uid": study_uid},
    }


@router.get(
    "/query/study_series_ope_no_status",
    status_code=200,
    summary="查詢 Series 操作狀態（按操作編號）",
    description="查詢單一 Study 下所有 Series 的特定操作狀態",
    response_description="分頁的 Series 操作狀態列表",
    tags=["Status Queries"],
)
async def get_study_series_ope_no_status(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    limit: int = Query(20, ge=1, description="每頁結果數"),
    offset: int = Query(0, ge=0, description="分頁偏移量"),
    study_uid: Optional[OrthancID] = Query(None, description="Study UID"),
    ope_no: OpeNo = Query(..., description="操作編號"),
):
    """
    查詢單一 Study 下所有 Series 的特定操作狀態。
    
    此端點用於前端顯示 Study 的進度。例如查詢所有 Series 的轉檔完成情況。
    
    Parameters
    ----------
    study_uid : OrthancID
        Study UID。
    ope_no : OpeNo
        操作編號，例如 "100.095" 表示查詢傳輸完成的 Series。
    limit : int, default=20
        每頁結果數（1-100）。
    offset : int, default=0
        分頁偏移量。
    
    Returns
    -------
    list
        符合條件的 Series 狀態列表。
    
    Examples
    --------
    查詢 Study 下傳輸完成的 Series：
    
    >>> GET /sync/query/study_series_ope_no_status?
    ...     study_uid=abc-123&ope_no=100.095&limit=50
    """
    if study_uid is None:
        raise HTTPException(status_code=422, detail="study_uid is required")
    result = await dcop_event_service.get_stydy_series_ope_no_status(
        study_uid=study_uid,
        ope_no=ope_no,
        limit=limit,
        offset=offset,
    )
    return result


@router.get(
    "/query/stydy_ope_no_status",
    status_code=200,
    summary="查詢 Study 操作狀態（按操作編號）",
    description="查詢 Study 維度的特定操作狀態及所有 Series 狀態",
    response_description="分頁的 Study 操作狀態",
    tags=["Status Queries"],
)
async def get_stydy_ope_no_status(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    study_uid: Optional[OrthancID] = Query(None, description="Study UID"),
    ope_no: OpeNo = Query(..., description="操作編號"),
    limit: int = Query(20, ge=1, description="每頁結果數"),
    offset: int = Query(0, ge=0, description="分頁偏移量"),
) -> OffsetPagination[StydySeriesOpeNoStatus]:
    """
    查詢 Study 維度的特定操作狀態。
    
    此端點聚合查詢，返回 Study 及其所有 Series 的特定操作狀態。
    適用於前端顯示 Study 的完整流程進度。
    
    Parameters
    ----------
    study_uid : OrthancID
        Study UID。
    ope_no : OpeNo
        操作編號。
    limit : int, default=20
        每頁結果數。
    offset : int, default=0
        分頁偏移量。
    
    Returns
    -------
    OffsetPagination[StydySeriesOpeNoStatus]
        分頁結果。
    
    Examples
    --------
    查詢 Study 的轉檔完成狀態：
    
    >>> GET /sync/query/stydy_ope_no_status?
    ...     study_uid=abc-123&ope_no=200.200
    """
    if study_uid is None:
        raise HTTPException(status_code=422, detail="study_uid is required")
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
    summary="查詢已完成轉檔的 Study 列表",
    description="查詢轉檔完成的 Study 列表，方便前端顯示進度",
    response_description="已完成轉檔的 Study 列表",
    tags=["Status Queries"],
)
async def get_check_study_series_conversion_complete(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    study_uid: Optional[str] = Query(None, description="可選：過濾特定 Study"),
):
    """
    查詢已完成 NIFTI 轉檔的 Study 列表。
    
    此端點用於前端顯示哪些 Study 已完成轉檔，準備進入推論階段。
    
    Parameters
    ----------
    study_uid : str, optional
        可選過濾條件。若指定，只返回匹配的 Study。
    
    Returns
    -------
    list
        已完成轉檔的 Study 列表。
    
    Examples
    --------
    查詢所有已完成轉檔的 Study：
    
    >>> GET /sync/query/check_study_series_conversion_complete
    
    查詢特定 Study：
    
    >>> GET /sync/query/check_study_series_conversion_complete?study_uid=abc-123
    """
    result = await dcop_event_service.get_check_study_series_conversion_complete(
        study_uid=study_uid
    )
    return result
