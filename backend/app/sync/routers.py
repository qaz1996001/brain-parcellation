# app/sync/routers.py
import os
import pathlib
from typing import Annotated, Tuple, List

from fastapi import APIRouter, Depends, Response, BackgroundTasks

from .schemas import OrthancIDRequest, DCOPStatus, DCOPEventRequest, DCOPEventNIFTITOOLRequest
from backend.app.sync import urls
from .service import DCOPEventDicomService
from .model import DCOPEventModel
from ..database import alchemy

router = APIRouter()


@router.get(urls.SYNC_PROT_STUDY, status_code=200,
            summary="查詢已同步的 Study UUID ",
            description="",
            response_description="",)
async def get_study_uuid() -> Response:
    return Response("DICOM Service is running")


# @router.get(urls.SYNC_PROT_OPE_NO, status_code=200,
#              summary="根據 study series UUID ope_no 查詢資料",
#              description="根據 study series UUID ope_no 查詢資料",
#              response_description="",)
async def get_study_last(data :List[OrthancIDRequest],
                         dcop_event_service: Annotated[DCOPEventDicomService,
                         Depends(alchemy.provide_service(DCOPEventDicomService))],) -> Response:

    return Response("get_ope_no")



async def get_series_info(study_uid: str,dcop_event_service:DCOPEventDicomService):
    from code_ai.task.task_dicom2nii import dicom_to_nii
    from code_ai.task.schema.intput_params import Dicom2NiiParams
    from code_ai import load_dotenv
    load_dotenv()
    raw_dicom_path    = pathlib.Path(os.getenv("PATH_RAW_DICOM"))
    rename_dicom_path = pathlib.Path(os.getenv("PATH_RENAME_DICOM"))
    rename_nifti_path = pathlib.Path(os.getenv("PATH_RENAME_NIFTI"))
    study_uid_raw_dicom_path = raw_dicom_path.joinpath(study_uid)
    if study_uid_raw_dicom_path.exists():
        series_uid_path_list = sorted(study_uid_raw_dicom_path.iterdir())
        new_data_list = []
        for series_uid_path in series_uid_path_list:
            series_new_data = await DCOPEventModel.create_event(study_uid=study_uid,
                                                                series_uid=series_uid_path.name,
                                                                status=DCOPStatus.SERIES_NEW.name,
                                                                session=dcop_event_service.repository.session,)
            series_transferring_data = await DCOPEventModel.create_event(study_uid=study_uid,
                                                                         series_uid=series_uid_path.name,
                                                                         status=DCOPStatus.SERIES_TRANSFERRING.name,
                                                                         session=dcop_event_service.repository.session,)
            new_data_list.append(series_new_data)
            new_data_list.append(series_transferring_data)
        try:
            data_obj = await dcop_event_service.create_many(new_data_list,auto_commit= True)
        except:
            await dcop_event_service.repository.session.rollback()
        task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
                                      output_dicom_path=rename_dicom_path,
                                      output_nifti_path=rename_nifti_path, )

        task = dicom_to_nii.push(task_params.get_str_dict())
    return task


@router.post(urls.SYNC_PROT_STUDY, status_code=200,
             summary="更新同步的 Study UUID",
             description="傳入 STUDY_TRANSFER_COMPLETE Study UUID",
             response_description="",)
async def post_study_uuid(request:OrthancIDRequest,
                          dcop_event_service: Annotated[DCOPEventDicomService,
                          Depends(alchemy.provide_service(DCOPEventDicomService))],
                          background_tasks: BackgroundTasks
                          ) -> Response:


    result_list = await dcop_event_service.add_study_new(data_list=request.ids)
    print('result_list',result_list)
    background_tasks.add_task(dcop_event_service.dicom_tool_get_series_info,result_list)
    return result_list



# @router.delete(urls.SYNC_PROT_STUDY, status_code=200,
#                summary="delete Study UUID",
#                description="",
#                response_description="",)
async def delete_study_uuid(request:OrthancIDRequest) -> Response:
    print(request.ids)
    return Response("DICOM Service is running")



@router.get(urls.SYNC_PROT_OPE_NO, status_code=200,
             summary="根據 study series UUID ope_no 查詢資料",
             description="根據 study series UUID ope_no 查詢資料",
             response_description="",)
async def get_ope_no(data :List[DCOPEventRequest],
                     dcop_event_service: Annotated[DCOPEventDicomService,
                     Depends(alchemy.provide_service(DCOPEventDicomService))],) -> Response:

    return Response("get_ope_no")



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
async def post_check_study_series_transfer_complete(
        dcop_event_service: Annotated[DCOPEventDicomService,
        Depends(alchemy.provide_service(DCOPEventDicomService))],
        background_tasks: BackgroundTasks) -> Response:

    background_tasks.add_task(dcop_event_service.check_study_series_transfer_complete)
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




@router.post(urls.SYNC_PROT_STUDY_CONVERSION_COMPLETE,
             status_code=200,
             summary="檢查 study series nifti conversion complete",
             description="",
             response_description="",)
async def post_check_study_series_conversion_complete(
        dcop_event_service: Annotated[DCOPEventDicomService,
        Depends(alchemy.provide_service(DCOPEventDicomService))],
        background_tasks: BackgroundTasks) -> Response:

    background_tasks.add_task(dcop_event_service.check_study_series_conversion_complete)
    return Response("post_check_study_series_conversion_complete")

