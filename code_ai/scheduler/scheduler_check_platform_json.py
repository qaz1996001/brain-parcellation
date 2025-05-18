import json
import os
import pathlib
from datetime import datetime

import nb_log
from funboost import BoosterParams, ConcurrentModeEnum, BrokerEnum, Booster
from sqlalchemy.orm import Session

from code_ai.pipeline import upload_json
from code_ai.utils.database import get_sqla_helper
from code_ai.utils.model import RawDicomToNiiInference

logger_ = (nb_log.LogManager('check_platform_json').
           get_logger_and_add_handlers(log_filename='check_platform_json.log'))


@Booster(BoosterParams(queue_name='check_platform_json_queue',
                       broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
                       concurrent_mode=ConcurrentModeEnum.SOLO,
                       qps=1,))
def check_platform_json():
    from code_ai import load_dotenv
    load_dotenv()
    path_process = os.getenv("PATH_PROCESS")
    deep_cmd_tools_path = os.path.join(path_process, 'Deep_cmd_tools')
    cmd_json_path_list = sorted(pathlib.Path(deep_cmd_tools_path).iterdir())
    enginex, sqla_helper = get_sqla_helper()
    for cmd_json_path in cmd_json_path_list:
        if os.path.exists(cmd_json_path):
            with open(cmd_json_path, 'r') as f:
                data_list = json.load(f)
            session: Session = sqla_helper.session
            for cmd_data in data_list:
                output_dicom_path = str(cmd_data['input_dicom_dir'][0])
                output_nifti_path = os.path.dirname( cmd_data['output_list'][0])
                task_params = RawDicomToNiiInference( name = 'check_platform_json_queue',
                                                      sub_dir=None,
                                                      output_nifti_path = output_nifti_path,
                                                      output_dicom_path = output_dicom_path
                                                      )
                #  檢查重複
                with session:
                    result = session.query(RawDicomToNiiInference).filter(
                        RawDicomToNiiInference.name == task_params.name,
                        RawDicomToNiiInference.output_dicom_path == str(task_params.output_dicom_path),
                        RawDicomToNiiInference.output_nifti_path == str(task_params.output_nifti_path),
                        ).one_or_none()
                    continue
                #   沒有重複
                if result is None:
                    created_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                    raw_params = RawDicomToNiiInference(sub_dir=None,
                                                        name='check_platform_json_queue',
                                                        output_dicom_path=str(task_params.output_dicom_path),
                                                        output_nifti_path=str(task_params.output_nifti_path),
                                                        created_time=created_time_str)
                    try:
                        upload_json(cmd_data['study_id'],
                                    cmd_data['name'])
                        session.add(raw_params)
                        session.commit()
                    except :
                        nii_file_list = list(filter(lambda x: str(x).endswith('nii.gz'), cmd_data['output_list']))
                        platform_json_list = list(
                            map(lambda x: str(x).replace('.nii.gz', '_platform_json.json'), nii_file_list))
                        
                        platform_json_list = list(map(lambda x: x[0],os.path.exists(x[1]), enumerate(platform_json_list)))
                    finally:
                        pass

        else:
            logger_.debug(f'{cmd_json_path} is not exist')






if __name__ == '__main__':
    print(check_platform_json.push())
