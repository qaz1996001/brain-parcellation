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


@router.get("/", status_code=200)
async def get_index() -> Response:
    return Response("Hello")


@router.get("/names", )
async def get_name_list() -> List[str]:
    series_list = []
    series_list.extend(map(lambda x:x,series_structure_sort.keys()))
    series_list.extend(map(lambda x: x, series_special_sort.keys()))
    series_list.extend(map(lambda x: x, series_perfusion_sort.keys()))
    series_list.extend(map(lambda x: x, series_functional_sort.keys()))
    return series_list


@router.post("/path", status_code=200)
async def post_dicom_path_list(file_path_list :Optional[List[FilePath]],
                               convert_manager:ConvertManager = Depends(get_rename_dicom_manager),
                               dicom_orientation:ImageOrientationProcessingStrategy = Depends(get_dicom_orientation),) -> List[SeriesResponse]:
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



@router.post("/file", status_code=200)
async def post_dicom_file_list(dicom_file_list :Optional[List[UploadFile]],
                               convert_manager: ConvertManager = Depends(get_rename_dicom_manager),
                               dicom_orientation: ImageOrientationProcessingStrategy = Depends(get_dicom_orientation),
                               ) -> List[SeriesResponse]:
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