import pathlib
from typing import List

from advanced_alchemy.extensions.fastapi import (
    repository,
    service,
)
from requests import session

from .model import DCOPEventModel
from .schemas import DCOPStatus, DCOPEventRequest


class DCOPEventDicomService(service.SQLAlchemyAsyncRepositoryService[DCOPEventModel]):
    """Author repository."""

    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        """Author repository."""

        model_type = DCOPEventModel

    repository_type = Repo


    async def post_ope_no_task(self,data :List[DCOPEventRequest]):
        from code_ai import load_dotenv
        load_dotenv()
        print('data',data)
        result_list = []
        for dcop_event in data:
            new_data = await DCOPEventModel.create_event_ope_no(tool_id    = dcop_event.tool_id,
                                                                study_uid  = dcop_event.study_uid,
                                                                series_uid = dcop_event.series_uid,
                                                                study_id   = dcop_event.study_id,
                                                                ope_no     = dcop_event.ope_no,
                                                                result_data= dcop_event.result_data,
                                                                params_data= dcop_event.params_data,
                                                                session    = self.repository.session)
            new_data_obj = await self.create(data=new_data)
            result_list.append(new_data_obj)
        try:
            data_obj = await self.create_many(result_list,auto_commit= True)
        except:
            await self.repository.session.rollback()
        return data_obj


    async def check_study_transfer_complete_task(self):
        """
            檢查 study 下的 series 是否都傳輸完成
            1.series 完成傳輸添加 SERIES_TRANSFER_COMPLETE  的記錄
            2.所有series都到了SERIES_TRANSFER_COMPLETE， 添加 STUDY_TRANSFER_COMPLETE 的記錄
        """
        from code_ai import load_dotenv
        load_dotenv()
        session = self.repository.session




    async def get_series_info(self, study_uid: str):
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
                                                                    session=dcop_event_service.repository.session, )
                series_transferring_data = await DCOPEventModel.create_event(study_uid=study_uid,
                                                                             series_uid=series_uid_path.name,
                                                                             status=DCOPStatus.SERIES_TRANSFERRING.name,
                                                                             session=dcop_event_service.repository.session, )
                new_data_list.append(series_new_data)
                new_data_list.append(series_transferring_data)
            try:
                data_obj = await cls.create_many(new_data_list, auto_commit=True)
            except:
                await cls.repository.session.rollback()
            task_params = Dicom2NiiParams(sub_dir=study_uid_raw_dicom_path,
                                          output_dicom_path=rename_dicom_path,
                                          output_nifti_path=rename_nifti_path, )

            task = dicom_to_nii.push(task_params.get_str_dict())
        return task