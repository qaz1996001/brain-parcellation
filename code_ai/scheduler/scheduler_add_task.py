import os
import pathlib
from datetime import datetime

import nb_log
from funboost import BoosterParams, ConcurrentModeEnum, BrokerEnum, Booster
from sqlalchemy.orm import Session


from code_ai.utils.database import get_sqla_helper
from code_ai.utils.model import RawDicomToNiiInference

logger_add_raw_dicom_to_nii_inference = (nb_log.LogManager('add_raw_dicom_to_nii_inference').
                                         get_logger_and_add_handlers(log_filename='add_raw_dicom_to_nii_inference.log'))


@Booster(BoosterParams(queue_name='add_raw_dicom_to_nii_inference_queue',
                       broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
                       concurrent_mode=ConcurrentModeEnum.SOLO,
                       qps=1,))
def add_raw_dicom_to_nii_inference():
    from code_ai import load_dotenv
    from code_ai.task.task_dicom2nii import dicom_to_nii
    from code_ai.task.task_pipeline import task_pipeline_inference
    from code_ai.task.schema.intput_params import Dicom2NiiParams
    load_dotenv()
    PATH_RAW_DICOM    = os.getenv("PATH_RAW_DICOM")
    PATH_RENAME_DICOM = os.getenv("PATH_RENAME_DICOM")
    PATH_RENAME_NIFTI = os.getenv("PATH_RENAME_NIFTI")

    input_dicom = pathlib.Path(PATH_RAW_DICOM)
    output_dicom_path = pathlib.Path(PATH_RENAME_DICOM)
    output_nifti_path = pathlib.Path(PATH_RENAME_NIFTI)

    input_dicom_list = sorted(input_dicom.iterdir())
    input_dicom_list = list(filter(lambda x: x.is_dir(), input_dicom_list))
    result_list = []
    if len(input_dicom_list) == 0:
        input_dicom_list = [input_dicom]
    enginex, sqla_helper = get_sqla_helper()
    #
    session: Session = sqla_helper.session
    for input_dicom_path in input_dicom_list:
        task_params = Dicom2NiiParams(
            sub_dir=input_dicom_path,
            output_dicom_path=output_dicom_path,
            output_nifti_path=output_nifti_path, )
    #  檢查重複
        with session :
            result = session.query(RawDicomToNiiInference).filter(RawDicomToNiiInference.sub_dir==str(task_params.sub_dir),
                                                                   RawDicomToNiiInference.output_dicom_path == str(task_params.output_dicom_path),
                                                                   RawDicomToNiiInference.output_nifti_path == str(task_params.output_nifti_path),
                                                         ).one_or_none()
    #   沒有重複
        if result is None:
            # 發任務
            task = dicom_to_nii.push(task_params.get_str_dict())
            result_list.append(task)
            with session:
                created_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                session.add(RawDicomToNiiInference(sub_dir=str(task_params.sub_dir),
                                                   name = 'dicom_to_nii_queue',
                                                   output_dicom_path=str(task_params.output_dicom_path),
                                                   output_nifti_path=str(task_params.output_nifti_path),
                                                   created_time = created_time_str
                                                   ))
                session.commit()
        else:
    #   重複任務
            continue
        logger_add_raw_dicom_to_nii_inference.debug('task_params = {}'.format(task_params.get_str_dict()))
    for async_result in result_list:
        async_result.set_timeout(3600)
    result_list = [async_result.result for async_result in result_list]
    logger_add_raw_dicom_to_nii_inference.debug('result_list')
    if len(result_list) > 0:
        output_nifti_path_list = list(map(lambda x:output_nifti_path.joinpath(os.path.basename(x)),
                                          result_list))
        for nifti_study_path in output_nifti_path_list:
            dicom_study_path = output_dicom_path.joinpath(nifti_study_path.name)
            nifti_study_path_str = str(nifti_study_path)
            dicom_study_path_str = str(dicom_study_path)
            #  檢查重複
            with session:
                result = session.query(RawDicomToNiiInference).filter(
                    RawDicomToNiiInference.sub_dir == None,
                    RawDicomToNiiInference.output_dicom_path == dicom_study_path_str,
                    RawDicomToNiiInference.output_nifti_path == nifti_study_path_str,
                    ).one_or_none()
            #   沒有重複
            if result is None:
                # 發任務
                task_pipeline_result = task_pipeline_inference.push({'nifti_study_path': str(nifti_study_path_str),
                                                                     'dicom_study_path': str(dicom_study_path_str),
                                                                     })
                with session:
                    created_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                    session.add(RawDicomToNiiInference(
                                                       name='task_pipeline_inference_queue',
                                                       sub_dir=None,
                                                       output_dicom_path=dicom_study_path_str,
                                                       output_nifti_path=nifti_study_path_str,
                                                       created_time=created_time_str
                                                       ))
                    session.commit()





if __name__ == '__main__':
    print(add_raw_dicom_to_nii_inference.push())
