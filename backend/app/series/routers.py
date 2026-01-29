"""
Series 模組的 FastAPI 路由層。

此模組定義了所有與 DICOM 序列分析相關的 HTTP 端點，包括：
- 健康檢查
- 序列類型查詢
- 檔案路徑分析
- HTTP 上傳分析

Architecture
-----------
遵循清晰的分層架構：
- routers.py: HTTP 端點定義和參數驗證
- schemas.py: 資料序列化方案
- deps.py: 依賴注入配置

API 端點設計
-----------
所有端點都支援：
- 標準 HTTP 狀態碼
- OpenAPI 文檔自動生成
- 類型安全的參數驗證
- 清晰的錯誤訊息

Examples
--------
健康檢查：

    GET /series

取得序列類型列表：

    GET /series/types

透過檔案路徑分析：

    POST /series/dicom/analyze/by-path
    Body: ["/path/to/file1.dcm", "/path/to/file2.dcm"]

透過上傳分析：

    POST /series/dicom/analyze/by-upload
    Content-Type: multipart/form-data
    Files: dicom_file_list

See Also
--------
backend.app.series.schemas : 資料模型定義
backend.app.series.deps : 依賴注入配置
code_ai.dicom2nii.convert : DICOM 轉換管理器
"""

import io
import logging
from typing import List, Optional, TYPE_CHECKING
from pydantic import FilePath
from fastapi import APIRouter, Depends, Response, UploadFile
import pydicom

if TYPE_CHECKING:
    pass

from code_ai.dicom2nii.convert import ConvertManager
from code_ai.dicom2nii.convert.base import ImageOrientationProcessingStrategy
from code_ai.dicom2nii.convert.config import ImageOrientationEnum

from backend.app.series.schemas import SeriesResponse

logger = logging.getLogger(__name__)
from backend.app.series.schemas import (
    series_special_sort,
    series_perfusion_sort,
    series_structure_sort,
    series_functional_sort,
)
from backend.app.series.deps import get_rename_dicom_manager, get_dicom_orientation
from backend.app.series import urls

# 建立 API 路由器實例
router = APIRouter()


@router.get(
    urls.SERIES_GET_HEALTH_CHECK,
    status_code=200,
    summary="健康檢查",
    description="檢查 DICOM 序列分析服務是否正常運行",
    response_description="返回服務狀態訊息",
    
)
async def get_health_check() -> Response:
    """
    健康檢查端點 - 驗證服務可用性。
    
    此端點用於監控和健康檢查，確認 DICOM 序列分析服務
    是否正常運行。通常用於：
    - 負載均衡器健康檢查
    - 監控系統狀態檢查
    - 服務部署驗證
    
    Returns
    -------
    Response
        HTTP 200 響應，包含服務狀態訊息 "DICOM Service is running"。
    
    Examples
    --------
    >>> import httpx
    >>> async with httpx.AsyncClient() as client:
    ...     response = await client.get("/series")
    ...     print(response.text)
    DICOM Service is running
    
    Notes
    -----
    - 此端點不進行任何實際的 DICOM 處理
    - 響應時間極短，適合頻繁檢查
    - 可用於自動化監控腳本
    """
    return Response("DICOM Service is running")


@router.get(
    urls.SERIES_GET_AVAILABLE_SERIES_TYPES,
    summary="取得可用的 DICOM 序列類型",
    description="返回系統支援的所有 DICOM 序列類型名稱列表",
    response_description="DICOM 序列類型的字串陣列",
    response_model=List[str],
    
)
async def get_available_series_types() -> List[str]:
    """
    取得系統支援的所有 DICOM 序列類型列表。
    
    此端點返回系統能夠識別的所有 DICOM 序列類型，包括：
    
    1. **結構影像序列** (series_structure_sort)
       - T1, T2, DWI, ADC 等基礎序列
       - 各種方向和對比增強變體
       - 例如：T1_AXI, T1_COR, T1CE_AXI, T2FLAIR_AXI 等
    
    2. **特殊序列** (series_special_sort)
       - MRA（磁共振血管造影）
       - SWAN, eSWAN（磁敏感加權成像）
       - 例如：MRA_BRAIN, SWAN, eSWANmIP 等
    
    3. **灌注序列** (series_perfusion_sort)
       - DSC（動態磁敏感對比）
       - ASL（動脈自旋標記）
       - 例如：DSC, ASLSEQ, DSCCBF_COLOR 等
    
    4. **功能序列** (series_functional_sort)
       - RESTING（靜息態）
       - CVR（腦血管反應性）
       - DTI（擴散張量成像）
       - 例如：RESTING, CVR2000, DTI64D 等
    
    Returns
    -------
    List[str]
        所有支援的序列類型名稱列表，按類別排序。
        總數約 100+ 種序列類型。
    
    Examples
    --------
    >>> import httpx
    >>> async with httpx.AsyncClient() as client:
    ...     response = await client.get("/series/types")
    ...     types = response.json()
    ...     print(f"支援 {len(types)} 種序列類型")
    ...     print(types[:10])  # 顯示前 10 個
    支援 100+ 種序列類型
    ['ADC', 'DWI0', 'DWI1000', 'T1_AXI', 'T1_COR', ...]
    
    Notes
    -----
    - 返回的列表包含所有四個類別的序列類型
    - 序列類型名稱遵循標準命名規範
    - 此列表可用於前端下拉選單或驗證
    - 序列類型定義在 schemas.py 中維護
    
    See Also
    --------
    backend.app.series.schemas : 序列類型定義字典
    """
    # 收集所有序列類型
    series_list = []
    # 結構影像序列（T1, T2, DWI, ADC 等）
    series_list.extend(map(lambda x: x, series_structure_sort.keys()))
    # 特殊序列（MRA, SWAN, eSWAN 等）
    series_list.extend(map(lambda x: x, series_special_sort.keys()))
    # 灌注序列（DSC, ASL 等）
    series_list.extend(map(lambda x: x, series_perfusion_sort.keys()))
    # 功能序列（RESTING, CVR, DTI 等）
    series_list.extend(map(lambda x: x, series_functional_sort.keys()))
    return series_list


@router.post(
    urls.SERIES_ANALYZE_DICOM_FILES_BY_PATH,
    status_code=200,
    summary="透過檔案路徑分析 DICOM",
    description="根據提供的檔案路徑列表，分析 DICOM 序列類型和影像方向",
    response_description="包含分析結果的 DICOM 序列列表",
    response_model=List[SeriesResponse],
    
)
async def analyze_dicom_files_by_path(
    file_path_list: Optional[List[FilePath]],
    convert_manager: ConvertManager = Depends(get_rename_dicom_manager),
    dicom_orientation: ImageOrientationProcessingStrategy = Depends(
        get_dicom_orientation
    ),
) -> List[SeriesResponse]:
    """
    透過檔案路徑分析 DICOM 檔案。
    
    根據提供的檔案路徑列表，讀取並分析 DICOM 檔案，識別序列類型和影像方向。
    此方法僅讀取 DICOM 標頭資訊，不載入像素資料，以提升處理效能。
    
    Parameters
    ----------
    file_path_list : Optional[List[FilePath]]
        DICOM 檔案的完整路徑列表。
        路徑必須是伺服器可存取的有效路徑。
        若為 None 或空列表，返回空結果。
    convert_manager : ConvertManager, optional
        依賴注入的 DICOM 轉換管理器。
        用於識別序列類型。
    dicom_orientation : ImageOrientationProcessingStrategy, optional
        依賴注入的影像方向處理策略。
        用於分析影像的空間方向（軸位、矢狀位、冠狀位）。
    
    Returns
    -------
    List[SeriesResponse]
        每個檔案的分析結果列表，包含：
        - file_name: 檔案名稱
        - series_type: 識別的序列類型（如 "T1_AXI", "DWI0" 等）
        - series_orientation: 影像方向（如 "AXI", "COR", "SAG" 等）
        
        若無法識別序列類型，series_type 將設為 "unknown"。
    
    Side Effects
    -----------
    - 讀取檔案系統中的 DICOM 檔案
    - 檔案必須存在且可讀取
    
    Examples
    --------
    分析單個檔案：
    
    >>> file_paths = ["/data/dicom/scan1.dcm"]
    >>> response = await client.post(
    ...     "/series/dicom/analyze/by-path",
    ...     json=file_paths
    ... )
    >>> results = response.json()
    >>> print(results[0])
    {
        "file_name": "scan1.dcm",
        "series_type": "T1_AXI",
        "series_orientation": "AXI"
    }
    
    批次分析多個檔案：
    
    >>> file_paths = [
    ...     "/data/dicom/scan1.dcm",
    ...     "/data/dicom/scan2.dcm",
    ...     "/data/dicom/scan3.dcm"
    ... ]
    >>> response = await client.post(
    ...     "/series/dicom/analyze/by-path",
    ...     json=file_paths
    ... )
    >>> results = response.json()
    >>> for result in results:
    ...     print(f"{result['file_name']}: {result['series_type']}")
    
    Notes
    -----
    **效能優化:**
    - 使用 `stop_before_pixels=True` 僅讀取標頭，不載入像素資料
    - 大幅減少記憶體使用和處理時間
    
    **限制:**
    - 單次請求最多處理 100 個檔案
    - 超過 100 個檔案時，僅處理前 100 個
    
    **錯誤處理:**
    - 無法讀取的檔案會被跳過（需在調用方處理）
    - 無法識別的序列類型標記為 "unknown"
    
    **檔案路徑要求:**
    - 必須是伺服器可存取的絕對路徑
    - 檔案必須存在且具有讀取權限
    - 支援標準 DICOM 格式（.dcm, .dicom）
    
    See Also
    --------
    analyze_dicom_files_by_upload : 透過 HTTP 上傳分析
    get_available_series_types : 取得所有支援的序列類型
    """
    rename_dicom_list = []
    
    # 檢查 file_path_list 是否為 None
    if file_path_list is None:
        return rename_dicom_list
    
    # 限制批次處理數量（最多 100 個檔案）
    if len(file_path_list) > 100:
        file_path_list = file_path_list[:100]
    
    # 逐一處理每個檔案
    for file_path in file_path_list:
        try:
            # 讀取 DICOM 檔案（僅標頭，不載入像素）
            with open(file_path, mode="rb") as f:
                dcm_ds = pydicom.dcmread(f, stop_before_pixels=True)

                # 識別序列類型（添加錯誤處理）
                try:
                    rename_dicom = convert_manager.rename_dicom_path(dcm_ds)
                except Exception as e:
                    logger.warning(f"無法識別序列類型 {file_path}: {e}")
                    rename_dicom = ""

                # 分析影像方向（添加錯誤處理）
                try:
                    orientation = dicom_orientation.process(dcm_ds)
                except Exception as e:
                    logger.warning(f"無法分析影像方向 {file_path}: {e}")
                    orientation = ImageOrientationEnum.AXI

                # 建立響應物件
                rename_dicom_list.append(
                    SeriesResponse(
                        file_name=file_path.name,
                        series_type=rename_dicom if len(rename_dicom) > 0 else "unknown",
                        series_orientation=str(orientation.value),
                    )
                )
        except Exception as e:
            logger.error(f"無法讀取 DICOM 檔案 {file_path}: {e}")
            # 檔案無法讀取時，使用預設值
            rename_dicom_list.append(
                SeriesResponse(
                    file_name=file_path.name,
                    series_type="unknown",
                    series_orientation=str(ImageOrientationEnum.AXI.value),
                )
            )

    return rename_dicom_list


@router.post(
    urls.SERIES_ANALYZE_DICOM_FILES_BY_UPLOAD,
    status_code=200,
    summary="透過上傳檔案分析 DICOM",
    description="透過 HTTP 檔案上傳的方式，分析 DICOM 序列類型和影像方向",
    response_description="包含分析結果的 DICOM 序列列表",
    response_model=List[SeriesResponse],
)
async def analyze_dicom_files_by_upload(
    dicom_file_list: Optional[List[UploadFile]],
    convert_manager: ConvertManager = Depends(get_rename_dicom_manager),
    dicom_orientation: ImageOrientationProcessingStrategy = Depends(
        get_dicom_orientation
    ),
) -> List[SeriesResponse]:
    """
    透過 HTTP 上傳分析 DICOM 檔案。
    
    接收透過 HTTP multipart/form-data 上傳的 DICOM 檔案，進行序列類型和影像方向分析。
    此方法使用記憶體友善的串流處理方式，即時處理上傳的檔案，無需儲存到伺服器。
    
    Parameters
    ----------
    dicom_file_list : Optional[List[UploadFile]]
        透過 HTTP multipart/form-data 上傳的 DICOM 檔案列表。
        每個檔案必須是標準 DICOM 格式。
        若為 None 或空列表，返回空結果。
    convert_manager : ConvertManager, optional
        依賴注入的 DICOM 轉換管理器。
        用於識別序列類型。
    dicom_orientation : ImageOrientationProcessingStrategy, optional
        依賴注入的影像方向處理策略。
        用於分析影像的空間方向。
    
    Returns
    -------
    List[SeriesResponse]
        每個上傳檔案的分析結果列表，包含：
        - file_name: 原始檔案名稱
        - series_type: 識別的序列類型
        - series_orientation: 影像方向
        
        若無法識別序列類型，series_type 將設為 "unknown"。
    
    Side Effects
    -----------
    - 讀取上傳的檔案內容到記憶體
    - 檔案不會保存在伺服器上（僅在記憶體中處理）
    
    Examples
    --------
    使用 curl 上傳單個檔案：
    
    ```bash
    curl -X POST "http://api/series/dicom/analyze/by-upload" \
         -F "dicom_file_list=@scan1.dcm"
    ```
    
    使用 curl 上傳多個檔案：
    
    ```bash
    curl -X POST "http://api/series/dicom/analyze/by-upload" \
         -F "dicom_file_list=@scan1.dcm" \
         -F "dicom_file_list=@scan2.dcm" \
         -F "dicom_file_list=@scan3.dcm"
    ```
    
    使用 Python httpx 上傳：
    
    >>> import httpx
    >>> files = [
    ...     ("dicom_file_list", open("scan1.dcm", "rb")),
    ...     ("dicom_file_list", open("scan2.dcm", "rb"))
    ... ]
    >>> async with httpx.AsyncClient() as client:
    ...     response = await client.post(
    ...         "/series/dicom/analyze/by-upload",
    ...         files=files
    ...     )
    ...     results = response.json()
    ...     for result in results:
    ...         print(f"{result['file_name']}: {result['series_type']}")
    
    Notes
    -----
    **上傳要求:**
    - 檔案格式：標準 DICOM 格式 (.dcm, .dicom)
    - Content-Type: application/dicom 或 application/octet-stream
    - 檔案必須包含完整的 DICOM 標頭資訊
    
    **效能優化:**
    - 使用 `stop_before_pixels=True` 僅讀取標頭
    - 使用 BytesIO 進行記憶體串流處理
    - 不將檔案寫入磁盤，減少 I/O 開銷
    
    **限制:**
    - 單次上傳最多支援 100 個檔案
    - 超過 100 個檔案時，僅處理前 100 個
    - 檔案大小受伺服器記憶體限制
    
    **安全性:**
    - 檔案上傳後不會保存在伺服器上
    - 僅在記憶體中處理，處理完成後立即釋放
    - 建議在生產環境中設置檔案大小限制
    
    **錯誤處理:**
    - 無法解析的檔案會被跳過（需在調用方處理）
    - 無法識別的序列類型標記為 "unknown"
    - 損壞的 DICOM 檔案可能導致解析錯誤
    
    See Also
    --------
    analyze_dicom_files_by_path : 透過檔案路徑分析
    get_available_series_types : 取得所有支援的序列類型
    """
    rename_dicom_list = []
    
    # 檢查 dicom_file_list 是否為 None
    if dicom_file_list is None:
        return rename_dicom_list
    
    # 限制批次處理數量（最多 100 個檔案）
    if len(dicom_file_list) > 100:
        dicom_file_list = dicom_file_list[:100]
    
    # 逐一處理每個上傳的檔案
    for dicom_file in dicom_file_list:
        file_name = dicom_file.filename if dicom_file.filename is not None else "unknown"

        try:
            # 讀取檔案內容到記憶體（使用 BytesIO 進行串流處理）
            bytes_io = io.BytesIO(await dicom_file.read())
            bytes_io.seek(0)  # 重置指針到開頭

            # 讀取 DICOM 檔案（僅標頭，不載入像素）
            dcm_ds = pydicom.dcmread(bytes_io, stop_before_pixels=True)

            # 識別序列類型（添加錯誤處理）
            try:
                rename_dicom = convert_manager.rename_dicom_path(dcm_ds)
            except Exception as e:
                logger.warning(f"無法識別序列類型 {file_name}: {e}")
                rename_dicom = ""

            # 分析影像方向（添加錯誤處理）
            try:
                orientation = dicom_orientation.process(dcm_ds)
            except Exception as e:
                logger.warning(f"無法分析影像方向 {file_name}: {e}")
                orientation = ImageOrientationEnum.AXI

            # 建立響應物件
            rename_dicom_list.append(
                SeriesResponse(
                    file_name=file_name,
                    series_type=rename_dicom if len(rename_dicom) > 0 else "unknown",
                    series_orientation=str(orientation.value),
                )
            )
        except Exception as e:
            logger.error(f"無法讀取上傳的 DICOM 檔案 {file_name}: {e}")
            # 檔案無法讀取時，使用預設值
            rename_dicom_list.append(
                SeriesResponse(
                    file_name=file_name,
                    series_type="unknown",
                    series_orientation=str(ImageOrientationEnum.AXI.value),
                )
            )

    return rename_dicom_list
