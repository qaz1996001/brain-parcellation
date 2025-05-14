import logging
import re
import shutil
import os
import pathlib
import subprocess
from typing import List, Dict

from funboost import BrokerEnum, Booster
from pydicom import dcmread

from code_ai.dicom2nii.convert import ModalityProcessingStrategy, MRAcquisitionTypeProcessingStrategy, \
    MRRenameSeriesProcessingStrategy
from code_ai.dicom2nii.convert import DwiProcessingStrategy, ADCProcessingStrategy, EADCProcessingStrategy, \
    SWANProcessingStrategy
from code_ai.dicom2nii.convert import ESWANProcessingStrategy, MRABrainProcessingStrategy, MRANeckProcessingStrategy
from code_ai.dicom2nii.convert import MRAVRBrainProcessingStrategy, MRAVRNeckProcessingStrategy, T1ProcessingStrategy
from code_ai.dicom2nii.convert import T2ProcessingStrategy, ASLProcessingStrategy, DSCProcessingStrategy
from code_ai.dicom2nii.convert import RestingProcessingStrategy, DTIProcessingStrategy, CVRProcessingStrategy
from code_ai.dicom2nii.convert import NullEnum
from code_ai.dicom2nii.convert import MRSeriesRenameEnum

from code_ai.dicom2nii.convert import dicom_rename_mr_postprocess
from code_ai.dicom2nii.convert import convert_nifti_postprocess
from code_ai.task.schema import intput_params
from code_ai.task.task_params import BoosterParamsMyRABBITMQ
from code_ai.utils.database import save_result_status_to_sqlalchemy

def get_output_study(dicom_ds):
    if dicom_ds is None:
        return None
    study_folder_name = get_study_folder_name(dicom_ds)
    if not study_folder_name:
        return None
    return study_folder_name


def check_dicom_instance_number_at_last(output_study_instance: pathlib.Path) -> bool:
    if output_study_instance.exists():
        with open(output_study_instance, mode='rb') as dcm:
            dicom_ds = dcmread(dcm, stop_before_pixels=True)
            # (0020,0013)	Instance Number	187
            # (0020,1002)	Images In Acquisition	192
            instance_number_tag = dicom_ds.get((0x20, 0x13), False)
            images_acquisition_tag = dicom_ds.get((0x20, 0x1002), False)
            if instance_number_tag and images_acquisition_tag:
                instance_number = instance_number_tag.value
                images_acquisition = images_acquisition_tag.value
                if instance_number == images_acquisition:
                    return True
    return False


def get_series_folder_list(study_path: pathlib.Path, exclude_dicom_series) -> List[pathlib.Path]:
    series_folder_list = []
    for series_folder in study_path.iterdir():
        if series_folder.is_dir() and series_folder.name != '.meta':
            if series_folder.name not in exclude_dicom_series:
                series_folder_list.append(series_folder)
    return series_folder_list


def get_study_folder_name(dicom_ds):
    # Implement actual logic based on DICOM attributes
    modality = dicom_ds[0x08, 0x60].value
    patient_id = dicom_ds[0x10, 0x20].value
    accession_number = dicom_ds[0x08, 0x50].value
    study_date = dicom_ds.get((0x08, 0x20), None)
    if study_date is None:
        return None
    else:
        study_date = study_date.value
    return f'{patient_id}_{study_date}_{modality}_{accession_number}'


def rename_dicom_file(instance_path,
                      processing_strategy_list,
                      modality_processing_strategy,
                      mr_acquisition_type_processing_strategy):
    with open(instance_path, mode='rb') as dcm:
        dicom_ds = dcmread(dcm, stop_before_pixels=True,force=True)
    if dicom_ds is None:
        return tuple(['', ''])
    # Simulating renaming logic
    modality_enum = modality_processing_strategy.process(dicom_ds=dicom_ds)
    mr_acquisition_type_enum = mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
    for processing_strategy in processing_strategy_list:
        if modality_enum == processing_strategy.modality:
            for mr_acquisition_type in processing_strategy.mr_acquisition_type:
                if mr_acquisition_type_enum == mr_acquisition_type:
                    series_enum = processing_strategy.process(dicom_ds=dicom_ds)
                    if series_enum is not NullEnum.NULL:
                        output_study = get_output_study(dicom_ds)
                        return series_enum.value, output_study


def copy_dicom_file(input_tuple, instance_path, output_path):
    if input_tuple is None or len(input_tuple[0]) == 0 or input_tuple[1] is None:
        return
    rename_series = input_tuple[0]
    output_study = input_tuple[1]
    output_study_series = output_path.joinpath(output_study, rename_series)
    output_study_series.mkdir(exist_ok=True, parents=True)
    os.makedirs(output_study_series, exist_ok=True)
    output_study_instance: pathlib.Path = output_study_series.joinpath(instance_path.name)
    if output_study_series.is_dir():
        if output_study_instance.exists():
            return output_study_instance
        else:
            with open(instance_path, mode='rb') as instance:
                with open(output_study_instance, 'wb+') as output_instance:
                    shutil.copyfileobj(instance, output_instance)
            return output_study_instance


def file_processing(func_params: Dict[str, any]):
    study_folder_path = func_params.get('study_folder_path')
    post_process_manager = func_params.get('post_process_manager')
    if study_folder_path is None:
        return
    post_process_manager.post_process(study_folder_path)


@Booster(BoosterParamsMyRABBITMQ(queue_name='call_dcm2niix_queue',
                                 user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
                                 qps=10, ))
def call_dcm2niix(func_params: Dict[str, any]):
    task_params = intput_params.CallDcm2niixParams.model_validate(func_params,
                                                                  strict=False)
    output_series_file_path = task_params.output_series_file_path
    output_series_path = task_params.output_series_path
    series_path = task_params.series_path
    output_series_path.parent.mkdir(exist_ok=True, parents=True)
    cmd_str = f'dcm2niix -z y -f {output_series_path.name} -o {output_series_path.parent} {series_path}'
    process = subprocess.Popen(args=cmd_str, cwd='/', shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    pattern = re.compile(r"DICOM as (.*)\s[(]", flags=re.MULTILINE)
    match_result = pattern.search(stdout.decode())

    if match_result is None:
        return f'call_dcm2niix {stdout.decode()}'
    else:
        str_result = match_result.groups()[0]
        dcm2niix_output_path = pathlib.Path(f'{str_result}.nii.gz')
        if dcm2niix_output_path.name != output_series_path:
            try:
                # Rename the output file and corresponding JSON file
                dcm2niix_output_path.rename(output_series_file_path)
                dcm2niix_json_path = pathlib.Path(str(dcm2niix_output_path).replace('.nii.gz', '.json'))
                output_series_json_path = pathlib.Path(str(output_series_file_path).replace('.nii.gz', '.json'))
                if dcm2niix_json_path.exists():
                    dcm2niix_json_path.unlink()
                if output_series_json_path.exists():
                    output_series_json_path.unlink()
            except FileExistsError:
                print(rf'FileExistsError {series_path}')
        return output_series_path.name


@Booster(BoosterParamsMyRABBITMQ(queue_name='dicom_2_nii_file_queue',
                                 qps=10, ))
def dicom_2_nii_file(func_params: Dict[str, any]):
    task_params = intput_params.Dicom2NiiFileParams.model_validate(func_params,
                                                                   strict=False)
    dicom_study_folder_path = task_params.dicom_study_folder_path
    output_nifti_path = task_params.output_nifti_path
    FILE_SIZE = 500
    if dicom_study_folder_path is None:
        return
    series_list = list(filter(lambda series_path: series_path.name != '.meta', dicom_study_folder_path.iterdir()))
    workflows = []
    for series_path in series_list:
        if series_path.name in Dicm2NiixConverter.exclude_set:
            continue
        output_series_path = pathlib.Path(
            f'{str(series_path).replace(str(dicom_study_folder_path.parent), str(output_nifti_path))}')
        output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
        call_dcm2niix_params = intput_params.CallDcm2niixParams(output_series_file_path=output_series_file_path,
                                                                output_series_path=output_series_path,
                                                                series_path=series_path)
        if output_series_file_path.exists():
            if output_series_file_path.stat().st_size < FILE_SIZE:
                output_series_file_path.unlink()
            #     result = call_dcm2niix.push(call_dcm2niix_params.get_str_dict())
            #     workflows.append(result)
            # else:
            #     continue
            result = call_dcm2niix.push(call_dcm2niix_params.get_str_dict())
            workflows.append(result)
        else:
            result = call_dcm2niix.push(call_dcm2niix_params.get_str_dict())
            workflows.append(result)
    result_list = [async_result.result for async_result in workflows]
    nifti_study_folder_path = output_nifti_path.joinpath(dicom_study_folder_path.name)
    file_processing(func_params=dict(study_folder_path=nifti_study_folder_path,
                                     post_process_manager=ConvertManager.nifti_post_process_manager))
    return nifti_study_folder_path


@Booster(BoosterParamsMyRABBITMQ(queue_name='process_instances_queue',
                                 qps=100,
                                 log_level=logging.WARNING,
                                 # user_custom_record_process_info_func=save_result_status_to_sqlalchemy
                                 ))
def process_instances(func_params: Dict[str, any]):
    task_params = intput_params.ProcessInstancesParams.model_validate(func_params,
                                                                      strict=False)
    instance = task_params.instance
    output_dicom_path = task_params.output_dicom_path
    rename_dicom_file_tuple = rename_dicom_file(instance,
                                                ConvertManager.processing_strategy_list,
                                                ConvertManager.modality_processing_strategy,
                                                ConvertManager.mr_acquisition_type_processing_strategy)
    copy_dicom_file_tuple = copy_dicom_file(rename_dicom_file_tuple,
                                            instance,
                                            output_dicom_path)
    if copy_dicom_file_tuple:
        return str(copy_dicom_file_tuple)
    return None


def process_dir_next(sub_dir: pathlib.Path, output_dicom_path: pathlib.Path):
    instances_list = list(sub_dir.rglob('*.dcm'))
    if len(instances_list) == 0:
        instances_list = sorted(sub_dir.rglob('*'))
        instances_list = list(filter(lambda x: x.is_file(), instances_list))
        instance_path: pathlib.Path = instances_list[0]
    else:
        instance_path: pathlib.Path = instances_list[0]

    with open(instance_path, mode='rb') as dcm:
        dicom_ds = dcmread(dcm, stop_before_pixels=True)
        study_folder_name = get_study_folder_name(dicom_ds)
        study_folder_path = output_dicom_path.joinpath(study_folder_name)
        file_processing(func_params=dict(study_folder_path=study_folder_path,
                                         post_process_manager=ConvertManager.dicom_post_process_manager))
        dicom_study_folder_path = study_folder_path
    return dicom_study_folder_path


@Booster(BoosterParamsMyRABBITMQ(queue_name='process_dir_queue',
                                 user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
                                 qps=10,))
def process_dir(func_params: Dict[str, any]):
    task_params = intput_params.Dicom2NiiParams.model_validate(func_params,
                                                               strict=False)
    sub_dir = task_params.sub_dir
    output_dicom_path = task_params.output_dicom_path
    instances_list = sorted(sub_dir.rglob('*.dcm'))
    if len(instances_list) > 0:
        async_result_list = [process_instances.push(intput_params.ProcessInstancesParams(instance=instances,
                                                                                         output_dicom_path=output_dicom_path).get_str_dict())
                             for instances in instances_list]
    else:
        instances_list = sorted(sub_dir.rglob('*'))
        instances_list = list(filter(lambda x: x.is_file(), instances_list))
        async_result_list = [process_instances.push(intput_params.ProcessInstancesParams(instance=instances,
                                                                                         output_dicom_path=output_dicom_path).get_str_dict())
                             for instances in instances_list]
    for async_result in async_result_list:
        async_result.set_timeout(3600)

    result_list = [async_result.result for async_result in async_result_list]
    dicom_study_folder_path = process_dir_next(sub_dir, output_dicom_path)

    if task_params.output_nifti_path is not None:
        output_nifti_path = task_params.output_nifti_path
        dicom_2_nii_file_param = intput_params.Dicom2NiiFileParams(dicom_study_folder_path=dicom_study_folder_path,
                                                                   output_nifti_path=output_nifti_path)

        dicom_2_nii_file_result = dicom_2_nii_file.push(dicom_2_nii_file_param.get_str_dict())
        dicom_2_nii_file_result.set_timeout(3600)
        dicom_2_nii_file_result.get()
    return dicom_study_folder_path


@Booster(BoosterParamsMyRABBITMQ(queue_name='dicom_to_nii_queue',
                                 user_custom_record_process_info_func = save_result_status_to_sqlalchemy,
                                 qps=10,))
def dicom_to_nii(func_params: Dict[str, any]):

    task_params = intput_params.Dicom2NiiParams.model_validate(func_params,
                                                               strict=False)
    # 1. raw dicom -> rename dicom
    if task_params.sub_dir is not None:
        result = process_dir.push(func_params)
        result.set_timeout(3600)
        data = result.get()
    else:
        # rename dicom -> rename nifti
        dicom_2_nii_file_param = intput_params.Dicom2NiiFileParams(
            dicom_study_folder_path=task_params.output_dicom_path,
            output_nifti_path=task_params.output_nifti_path)
        result = dicom_2_nii_file.push(dicom_2_nii_file_param.get_str_dict())
        result.set_timeout(3600)
        data = result.get()

    return data


# @Booster('dicom_rename_queue',
#          broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10)
@Booster(BoosterParamsMyRABBITMQ(queue_name='dicom_rename_queue',
                                 qps=10,))
def dicom_rename(func_params: Dict[str, any]):
    process_dir_result = process_dir.push(func_params)
    return process_dir_result


# @Booster('nii_file_processing_queue',
#          broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10)
@Booster(BoosterParamsMyRABBITMQ(queue_name='nii_file_processing_queue',
                                 qps=10,))
def nii_file_processing(func_params: Dict[str, any]):
    print('nii_file_processing', func_params)
    study_folder_path = func_params.get('study_folder_path')
    if study_folder_path is None:
        return
    ConvertManager.nifti_post_process_manager.post_process(pathlib.Path(study_folder_path))


class ConvertManager:
    modality_processing_strategy: ModalityProcessingStrategy = ModalityProcessingStrategy()
    mr_acquisition_type_processing_strategy: MRAcquisitionTypeProcessingStrategy = MRAcquisitionTypeProcessingStrategy()
    processing_strategy_list: List[MRRenameSeriesProcessingStrategy] = [DwiProcessingStrategy(),
                                                                        ADCProcessingStrategy(),
                                                                        EADCProcessingStrategy(),
                                                                        SWANProcessingStrategy(),
                                                                        ESWANProcessingStrategy(),
                                                                        MRABrainProcessingStrategy(),
                                                                        MRANeckProcessingStrategy(),
                                                                        MRAVRBrainProcessingStrategy(),
                                                                        MRAVRNeckProcessingStrategy(),
                                                                        T1ProcessingStrategy(),
                                                                        T2ProcessingStrategy(),
                                                                        ASLProcessingStrategy(),
                                                                        DSCProcessingStrategy(),
                                                                        RestingProcessingStrategy(),
                                                                        CVRProcessingStrategy(),
                                                                        DTIProcessingStrategy()]
    dicom_post_process_manager = dicom_rename_mr_postprocess.PostProcessManager()
    nifti_post_process_manager = convert_nifti_postprocess.PostProcessManager()


class Dicm2NiixConverter:
    exclude_set = {
        MRSeriesRenameEnum.MRAVR_BRAIN.value,
        MRSeriesRenameEnum.MRAVR_NECK.value,

        # DSCSeriesRenameEnum.DSC.value,
        # DSCSeriesRenameEnum.rCBV.value,
        # DSCSeriesRenameEnum.rCBF.value,
        # DSCSeriesRenameEnum.MTT.value,
        #
        # ASLSEQSeriesRenameEnum.ASLSEQ.value,
        # ASLSEQSeriesRenameEnum.ASLPROD.value,
        #
        # ASLSEQSeriesRenameEnum.ASLSEQATT.value,
        # ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR.value,
        #
        # ASLSEQSeriesRenameEnum.ASLSEQCBF.value,
        # ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR.value,
        #
        # ASLSEQSeriesRenameEnum.ASLSEQPW.value,
    }
