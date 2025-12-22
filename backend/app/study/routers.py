"""
研究 (Study) 模組的 FastAPI 路由層。

此模組定義了所有與研究管理相關的 HTTP 端點，包括：
- 研究事件的複雜多條件搜索
- 事件的分頁和過濾
- 事件的排序

Architecture
-----------
遵循清晰的分層架構：
- routers.py: HTTP 端點定義和參數驗證
- service.py: 業務邏輯實現
- model.py: 資料庫模型
- schemas.py: 資料序列化方案

API 端點設計
-----------
所有端點都支援：
- 多欄位搜索
- 集合過濾器 (in 過濾)
- 日期範圍過濾
- 排序和分頁

Examples
--------
查詢所有研究事件：

    GET /study/list

複雜過濾範例：

    GET /study/list?searchString=keyword&tool_id=DICOM_TOOL&ope_no=100.095&limit=50&offset=0

按創建時間範圍過濾：

    GET /study/list?createTimeAfter=2024-01-01&createTimeBefore=2024-12-31

See Also
--------
backend.app.study.service : 業務邏輯層
backend.app.sync.schemas : 資料序列化方案
backend.app.config.deps : 過濾器配置
"""

import logging
from datetime import datetime
from typing import Annotated, cast
from advanced_alchemy.extensions.fastapi.providers import FieldNameType, FilterConfig
from fastapi import APIRouter, Depends
from advanced_alchemy.extensions.fastapi import service, filters

from backend.app.study import urls
from backend.app.sync.service import DCOPEventDicomService
from backend.app.sync.model import DCOPEventModel
from backend.app.sync.schemas import DCOPEventRequest, OrthancID
from backend.app.config.deps import provide_filters
from backend.app.database import alchemy

# 模組級日誌記錄器
logger = logging.getLogger(__name__)

# 建立 API 路由器實例
router = APIRouter()


@router.get(
    urls.STUDY_GET_LIST,
    status_code=200,
    summary="研究事件 - 複雜多條件搜索",
    response_model=service.OffsetPagination[DCOPEventRequest],
    tags=["Study"],
    description="查詢研究事件，支援多種過濾、搜索和排序選項。",
)
async def get_events_complex(
    dcop_event_service: Annotated[
        DCOPEventDicomService, Depends(alchemy.provide_service(DCOPEventDicomService))
    ],
    filters_list: Annotated[
        list[filters.FilterTypes],
        Depends(
            provide_filters(
                cast(
                    FilterConfig,
                    {
                        # 多欄位搜索 - 支援在多個欄位中同時搜索
                        "search": "params_data,result_data",
                        "search_ignore_case": True,
                        # 集合過濾器 - 支援 IN 過濾（多值）
                        "in_fields": [
                            FieldNameType(name="tool_id", type_hint=str),
                            FieldNameType(name="ope_no", type_hint=str),
                            FieldNameType(name="study_uid", type_hint=str),
                            FieldNameType(name="study_id", type_hint=str),
                            FieldNameType(name="series_uid", type_hint=str),
                        ],
                        # 日期範圍過濾器 - 支援 BEFORE/AFTER 過濾
                        # FieldNameType(name='update_time', type_hint=datetime)
                        "before_after_fields": [
                            FieldNameType(name="create_time", type_hint=datetime),
                        ],
                        # 排序配置 - 支援多欄位排序
                        "order_by": [
                            "study_uid",
                            "ope_no",
                            "create_time",
                        ],
                        # 分頁配置 - 使用 limit/offset 分頁
                        "pagination_type": "limit_offset",
                        "limit": 50,
                        "offset": 0,
                        # ID 過濾器 - 使用 Orthanc ID 格式驗證
                        "id_filter": OrthancID,
                    },
                )
            )
        ),
    ],
) -> service.OffsetPagination[DCOPEventModel]:
    """
    查詢研究事件 - 複雜多條件搜索。
    
    此端點提供靈活的搜索和過濾功能，支援：
    
    1. **多欄位搜索**
       - 在 params_data 和 result_data 中搜索
       - 不區分大小寫
    
    2. **集合過濾**
       - 按 tool_id 過濾（DICOM_TOOL、NIFTI_TOOL 等）
       - 按 ope_no 過濾（操作編號）
       - 按 study_uid、study_id、series_uid 過濾
    
    3. **日期範圍過濾**
       - 按創建時間進行 BEFORE/AFTER 過濾
    
    4. **排序**
       - 支援多欄位排序
       - 可組合升序和降序
    
    5. **分頁**
       - 使用 limit/offset 分頁
       - 默認每頁 50 條記錄
    
    Parameters
    ----------
    dcop_event_service : DCOPEventDicomService
        注入的服務依賴，通過 FastAPI 依賴注入系統提供。
    filters_list : list[FilterTypes]
        經過驗證和配置的過濾器列表，由 provide_filters 生成。
    
    Returns
    -------
    OffsetPagination[DCOPEventModel]
        分頁結果，包含：
        - items: 事件列表
        - total: 總記錄數
        - limit: 每頁記錄數
        - offset: 當前偏移量
    
    Examples
    --------
    查詢所有研究事件（無過濾）：
    
        GET /study/list
    
    搜索特定關鍵字：
    
        GET /study/list?searchString=keyword
    
    按多個欄位過濾：
    
        GET /study/list?tool_id=DICOM_TOOL&ope_no=100.095&limit=20&offset=0
    
    按日期範圍過濾：
    
        GET /study/list?createTimeAfter=2024-01-01&createTimeBefore=2024-12-31
    
    複雜過濾組合：
    
        GET /study/list?searchString=stroke&tool_id=NIFTI_TOOL,INFERENCE_TOOL&createTimeAfter=2024-01-01&limit=50&offset=0&orderBy=create_time
    
    Notes
    -----
    - 所有日期參數使用 ISO 8601 格式
    - 多值過濾使用逗號分隔
    - 搜索查詢是全文搜索，不是精確匹配
    - 分頁采用 limit/offset 方式
    
    See Also
    --------
    backend.app.study.service : 業務邏輯實現
    backend.app.sync.schemas : 資料模型定義
    """
    # 執行查詢並獲取結果
    results, total = await dcop_event_service.list_and_count(*filters_list)
    
    # 將結果轉換為響應格式並返回
    return dcop_event_service.to_schema(results, total, filters=filters_list)
