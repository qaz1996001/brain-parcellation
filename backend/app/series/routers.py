# app/series/routers.py

import io
from typing import List,Optional, TYPE_CHECKING
from pydantic import FilePath
from fastapi import APIRouter, Depends, Response,UploadFile
import pydicom

if TYPE_CHECKING:
    import pathlib

from code_ai.dicom2nii.convert import ConvertManager
from code_ai.dicom2nii.convert.base import ImageOrientationProcessingStrategy

from backend.app.series.schemas import SeriesResponse
from backend.app.series.schemas import series_special_sort,series_perfusion_sort,series_structure_sort,series_functional_sort
from backend.app.series.deps import get_rename_dicom_manager,get_dicom_orientation
from backend.app.series import urls

router = APIRouter()


@router.get(urls.SERIES_GET_HEALTH_CHECK, status_code=200,
            summary="健康檢查",
            description="檢查 DICOM 服務是否正常運行",
            response_description="返回服務狀態訊息",
            )
async def get_health_check() -> Response:
    return Response("DICOM Service is running")



@router.get(urls.SERIES_GET_AVAILABLE_SERIES_TYPES,
            summary="取得可用的 DICOM series 類型",
            description="返回系統支援的所有 DICOM series名稱列表",
            response_description="DICOM series 的字串陣列",
            response_model=List[str],)
async def get_available_series_types() -> List[str]:
    series_list = []
    series_list.extend(map(lambda x:x,series_structure_sort.keys()))
    series_list.extend(map(lambda x: x, series_special_sort.keys()))
    series_list.extend(map(lambda x: x, series_perfusion_sort.keys()))
    series_list.extend(map(lambda x: x, series_functional_sort.keys()))
    return series_list


@router.post(urls.SERIES_ANALYZE_DICOM_FILES_BY_PATH, status_code=200,
             summary="透過檔案路徑分析 DICOM",
             description="根據提供的檔案路徑列表，分析 DICOM series 類型和影像方向",
             response_description="包含分析結果的DICOM series list",
             response_model=List[SeriesResponse],
             )
async def analyze_dicom_files_by_path(file_path_list :Optional[List[FilePath]],
                                      convert_manager:ConvertManager = Depends(get_rename_dicom_manager),
                                      dicom_orientation:ImageOrientationProcessingStrategy = Depends(get_dicom_orientation),) -> List[SeriesResponse]:
    """
    ## 透過檔案路徑分析 DICOM 檔案

    根據提供的檔案路徑列表，讀取並分析 DICOM 檔案，識別序列類型和影像方向。

    **功能特點:**
    - 支援批次處理多個檔案（最多 100 個）
    - 自動識別 MRI 序列類型（T1WI, T2WI, FLAIR, DWI 等）
    - 分析影像方向（軸位、矢狀位、冠狀位）
    - 僅讀取 DICOM 標頭，不載入影像像素資料以提升效能

    **參數:**
    - **file_path_list**: DICOM 檔案的完整路徑列表

    **返回值:**
    - 每個檔案的分析結果，包含檔案名、序列類型、影像方向

    **注意事項:**
    - 單次請求最多處理 100 個檔案
    - 檔案路徑必須是伺服器可存取的有效路徑
    - 無法識別的序列類型將標記為 'unknown'
    """

    rename_dicom_list = []
    if len(file_path_list) > 100:
        file_path_list = file_path_list[:100]
    for file_path in file_path_list:
        with open(file_path,mode='rb') as f:
            dcm_ds = pydicom.dcmread(f,stop_before_pixels=True)
            rename_dicom = convert_manager.rename_dicom_path(dcm_ds)
            orientation  = dicom_orientation.process(dcm_ds)

            rename_dicom_list.append(SeriesResponse(file_name = file_path.name,
                                                    series_type=rename_dicom if len(rename_dicom)>0 else 'unknown' ,
                                                    series_orientation = str(orientation.value)
                                                    ))
    # series = pd.Series(rename_dicom_list,name='rename_dicom')
    return rename_dicom_list



@router.post(urls.SERIES_ANALYZE_DICOM_FILES_BY_UPLOAD, status_code=200,
             summary="透過上傳檔案分析 DICOM",
             description="透過 HTTP 檔案上傳的方式，分析 DICOM series 類型和影像方向",
             response_description="包含分析結果的DICOM series list",
             response_model=List[SeriesResponse],
             )
async def analyze_dicom_files_by_upload(dicom_file_list :Optional[List[UploadFile]],
                                        convert_manager: ConvertManager = Depends(get_rename_dicom_manager),
                                        dicom_orientation: ImageOrientationProcessingStrategy = Depends(get_dicom_orientation),
                                        ) -> List[SeriesResponse]:
    """
    ## 透過上傳檔案分析 DICOM 檔案

    接收透過 HTTP multipart/form-data 上傳的 DICOM 檔案，進行序列類型和影像方向分析。

    **功能特點:**
    - 支援多檔案同時上傳分析（最多 100 個）
    - 即時處理上傳的檔案，無需儲存到伺服器
    - 自動識別各種 MRI 序列類型
    - 分析影像的空間方向資訊
    - 記憶體友善的串流處理方式

    **上傳要求:**
    - 檔案格式：標準 DICOM 格式 (.dcm, .dicom)
    - Content-Type: application/dicom 或 application/octet-stream
    - 檔案必須包含完整的 DICOM 標頭資訊

    **返回值:**
    - 每個上傳檔案的分析結果
    - 包含原始檔名、識別的序列類型、影像方向

    **注意事項:**
    - 單次上傳最多支援 100 個檔案
    - 僅讀取 DICOM 標頭以提升處理速度
    - 檔案上傳後不會保存在伺服器上
    - 無法識別的序列將標記為 'unknown'

    **使用範例:**
    ```bash
    curl -X POST "your-api-url/dicom/analyze/by-upload" \
         -F "dicom_file_list=@scan1.dcm" \
         -F "dicom_file_list=@scan2.dcm"
    ```
    """

    rename_dicom_list = []
    if len(dicom_file_list) > 100:
        dicom_file_list = dicom_file_list[:100]
    for dicom_file in dicom_file_list:
        bytes_io = io.BytesIO( await dicom_file.read())
        bytes_io.seek(0)
        dcm_ds = pydicom.dcmread(bytes_io, stop_before_pixels=True)
        rename_dicom = convert_manager.rename_dicom_path(dcm_ds)
        orientation = dicom_orientation.process(dcm_ds)

        rename_dicom_list.append(SeriesResponse(file_name=dicom_file.filename,
                                                series_type=rename_dicom if len(rename_dicom) > 0 else 'unknown',
                                                series_orientation=str(orientation.value)
                                                ))
    return rename_dicom_list