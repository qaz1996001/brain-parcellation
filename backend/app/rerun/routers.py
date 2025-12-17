#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-12-17

ReRun 研究重新執行 API 路由模組

此模組定義了供客户端調用的 HTTP API 端點，用於觸發研究重新執行功能。
提供兩個主要路由：按 Study Rename ID 和按 Study UID 重新執行。

API 端點：
---------
1. POST /rerun/study/by-rename_id
   根據研究重命名 ID 重新執行整個流程

2. POST /rerun/study/by-uid
   根據研究 UID 重新執行整個流程

兩個端點都使用背景任務 (BackgroundTasks) 進行非同步執行，
立即向客户端回傳成功，實際處理在後台進行。

模組依賴：
--------
- FastAPI : Web 框架
- advanced_alchemy : SQLAlchemy 擴展
- ReRunStudyService : 重新執行服務
- DCOPEventDicomService : DICOM 事件服務

@author: sean Ho
"""

# 檔案位置提示：app/rerun/routers.py
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

# 建立 API 路由器實例
# 此路由器將被掛載到主應用程式中，處理 /rerun 路徑下的所有請求
router = APIRouter()


@router.post(
    urls.RERUN_PROT_STUDY_RENAME_ID,
    status_code=200,
    summary="根據 STUDY Rename ID 對 Study 重跑整個流程",
    description="""
    根據研究重命名 ID 列表，批量重新執行研究的整個處理流程。
    
    此端點接收一組研究重命名 ID，查詢對應的 Study UID，
    然後逐一重新執行，包括：
    1. 清理先前的結果和快取
    2. 建立新的處理事件
    3. 觸發 DICOM 系列資訊處理
    
    此操作為非同步後台任務，API 立即回傳成功。
    實際的重新執行在後台進行，可能需要數分鐘至數小時。
    """,
    response_description="立即回傳空響應，實際處理在後台進行",
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
    """
    根據 Study Rename ID 重新執行研究 (背景任務)
    
    Parameters
    ----------
    data_list : List[str]
        研究重命名 ID 列表
        示例：["study_rename_001", "study_rename_002"]
    
    re_event_service : ReRunStudyService
        重新執行服務實例 (由依賴注入系統提供)
    
    dcop_event_service : DCOPEventDicomService
        DICOM 事件服務實例 (由依賴注入系統提供)
    
    background_tasks : BackgroundTasks
        FastAPI 的背景任務隊列
    
    Returns
    -------
    Response
        HTTP 200 空響應
        Content-Type: text/plain
        Body: 空字符串
    
    Notes
    -----
    - 此端點使用 BackgroundTasks 進行非同步執行
    - 客户端無需等待長時間的処理，立即返回成功響應
    - 任何執行期間的錯誤都會被記錄到日誌，不會導致 HTTP 錯誤
    - 建議監控日誌或設定研究狀態輪詢機制來追蹤進度
    
    Examples
    --------
    cURL 請求：
    
    .. code-block:: bash
    
        curl -X POST "http://localhost:8000/rerun/study/by-rename_id" \\
            -H "Content-Type: application/json" \\
            -d '["study_rename_001", "study_rename_002"]'
    
    Python 客户端：
    
    .. code-block:: python
    
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/rerun/study/by-rename_id",
                json=["study_rename_001", "study_rename_002"]
            )
    
    See Also
    --------
    post_re_run_study_by_study_uid : 按 Study UID 重新執行
    ReRunStudyService.re_run_by_study_rename_id : 實際執行方法
    """
    # 將重新執行任務添加到後台任務隊列
    # FastAPI 會在回傳響應後立即執行此任務
    background_tasks.add_task(
        re_event_service.re_run_by_study_rename_id, data_list, dcop_event_service
    )
    
    # 立即回傳 200 OK，不等待任務完成
    return Response("")


@router.post(
    urls.RERUN_PROT_STUDY_UID,
    status_code=200,
    summary="根據 STUDY UID 對 Study 重跑整個流程",
    description="""
    根據研究 UID (DICOM 標準 UID 格式) 列表，批量重新執行研究的整個處理流程。
    
    此端點直接接受 Study UID 列表，無需額外查詢轉換。
    相比 by-rename_id 端點，此端點的執行更直接高效。
    
    重新執行流程包括：
    1. 清理先前的結果和快取
    2. 建立新的處理事件
    3. 觸發 DICOM 系列資訊處理
    
    此操作為非同步後台任務，API 立即回傳成功。
    """,
    response_description="立即回傳空響應，實際處理在後台進行",
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
    """
    根據 Study UID 重新執行研究 (背景任務)
    
    Parameters
    ----------
    request : PostStudyRequest
        HTTP 請求體，包含 Study UID 列表
        結構：{"ids": ["1.2.3.4.5", "1.2.3.4.6"]}
    
    re_event_service : ReRunStudyService
        重新執行服務實例 (由依賴注入系統提供)
    
    dcop_event_service : DCOPEventDicomService
        DICOM 事件服務實例 (由依賴注入系統提供)
    
    background_tasks : BackgroundTasks
        FastAPI 的背景任務隊列
    
    Returns
    -------
    Response
        HTTP 200 空響應
        Content-Type: text/plain
        Body: 空字符串
    
    Notes
    -----
    - 此端點為推薦使用的 API，相比 by-rename_id 更直接
    - 使用標準 DICOM UID 格式識別研究
    - 所有執行都在後台進行，客户端立即返回
    - 錯誤會被捕獲並記錄，不會導致 HTTP 錯誤回應
    - 建議實現客户端側的狀態輪詢機制來監控進度
    
    Examples
    --------
    cURL 請求：
    
    .. code-block:: bash
    
        curl -X POST "http://localhost:8000/rerun/study/by-uid" \\
            -H "Content-Type: application/json" \\
            -d '{"ids": ["1.2.3.4.5", "1.2.3.4.6"]}'
    
    Python 客户端：
    
    .. code-block:: python
    
        import httpx
        from backend.app.sync.schemas import PostStudyRequest
        
        async with httpx.AsyncClient() as client:
            request = PostStudyRequest(
                ids=["1.2.3.4.5", "1.2.3.4.6"]
            )
            response = await client.post(
                "http://localhost:8000/rerun/study/by-uid",
                json=request.model_dump()
            )
    
    See Also
    --------
    post_re_run_study_by_study_rename_id : 按 Study Rename ID 重新執行
    ReRunStudyService.re_run_by_study_uid : 實際執行方法
    PostStudyRequest : 請求體的 Pydantic 模型
    """
    # 從請求體中提取 Study UID 列表
    # request.ids 應為 List[OrthancID] 格式的 UID 列表
    
    # 將重新執行任務添加到後台任務隊列
    background_tasks.add_task(
        re_event_service.re_run_by_study_uid, request.ids, dcop_event_service
    )

    # 立即回傳 200 OK，不等待任務完成
    return Response("")
