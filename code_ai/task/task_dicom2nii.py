import re
import shutil
import os
import pathlib
import subprocess
from typing import List
import dcm2niix
from pydicom import dcmread
from celery import Celery, chain, group, shared_task
from . import app

from code_ai.dicom2nii.convert import ModalityProcessingStrategy,MRAcquisitionTypeProcessingStrategy, MRRenameSeriesProcessingStrategy
from code_ai.dicom2nii.convert import DwiProcessingStrategy, ADCProcessingStrategy,EADCProcessingStrategy,SWANProcessingStrategy
from code_ai.dicom2nii.convert import ESWANProcessingStrategy,MRABrainProcessingStrategy,MRANeckProcessingStrategy
from code_ai.dicom2nii.convert import MRAVRBrainProcessingStrategy,MRAVRNeckProcessingStrategy,T1ProcessingStrategy
from code_ai.dicom2nii.convert import T2ProcessingStrategy, ASLProcessingStrategy,DSCProcessingStrategy
from code_ai.dicom2nii.convert import RestingProcessingStrategy, DTIProcessingStrategy, CVRProcessingStrategy
from code_ai.dicom2nii.convert import NullEnum
from code_ai.dicom2nii.convert import MRSeriesRenameEnum
from ..dicom2nii.convert import dicom_rename_mr_postprocess
from ..dicom2nii.convert import convert_nifti_postprocess


def get_output_study(dicom_ds):
    if dicom_ds is None:
        return None
    study_folder_name = get_study_folder_name(dicom_ds)
    if not study_folder_name:
        return None
    return study_folder_name


def check_dicom_instance_number_at_last(output_study_instance:pathlib.Path) -> bool:
    if output_study_instance.exists():
        with open(output_study_instance, mode='rb') as dcm:
            dicom_ds = dcmread(dcm, stop_before_pixels=True)
            # (0020,0013)	Instance Number	187
            # (0020,1002)	Images In Acquisition	192
            instance_number_tag = dicom_ds.get((0x20, 0x13),False)
            images_acquisition_tag = dicom_ds.get((0x20, 0x1002),False)
            if instance_number_tag and images_acquisition_tag:
                instance_number = instance_number_tag.value
                images_acquisition = images_acquisition_tag.value
                if instance_number == images_acquisition:
                    return True
    return False


def get_series_folder_list(study_path: pathlib.Path,exclude_dicom_series) -> List[pathlib.Path]:
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


def chunk_list(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]



# @app.task(bind=True,rate_limit='250/s',priority=100)
# def rename_dicom_file(self,instance_path, processing_strategy_list,modality_processing_strategy,mr_acquisition_type_processing_strategy):
#     try:
#         with open(instance_path, mode='rb') as dcm:
#             dicom_ds = dcmread(dcm, stop_before_pixels=True)
#         if dicom_ds is None:
#             return tuple(['',''])
#         # Simulating renaming logic
#         modality_enum = modality_processing_strategy.process(dicom_ds=dicom_ds)
#         mr_acquisition_type_enum = mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
#         for processing_strategy in processing_strategy_list:
#             if modality_enum == processing_strategy.modality:
#                 for mr_acquisition_type in processing_strategy.mr_acquisition_type:
#                     if mr_acquisition_type_enum == mr_acquisition_type:
#                         series_enum = processing_strategy.process(dicom_ds=dicom_ds)
#                         if series_enum is not NullEnum.NULL:
#                             output_study = get_output_study(dicom_ds)
#                             return series_enum.value,output_study
#         return tuple(['',''])
#     except (InvalidDicomError, BytesLengthException):
#         print(f"Invalid DICOM file: {instance_path}")
#         return None
#     except Exception as e:
#         self.retry(countdown=5, max_retries=5)  # 重試任務
#
#
# @app.task(bind=True,rate_limit='100/s',priority=80)
# def copy_dicom_file(self,input_tuple,instance_path,output_path):
#     try:
#         if input_tuple is None or len(input_tuple[0]) == 0 or input_tuple[1] is None:
#             return
#         rename_series = input_tuple[0]
#         output_study = input_tuple[1]
#         output_study_series = output_path.joinpath(output_study,rename_series)
#         output_study_series.mkdir(exist_ok=True,parents=True)
#         os.makedirs(output_study_series, exist_ok=True)
#         output_study_instance:pathlib.Path = output_study_series.joinpath(instance_path.name)
#         if output_study_series.is_dir():
#             if output_study_instance.exists():
#                 return
#             else:
#                 with open(instance_path,mode='rb') as instance:
#                     with open(output_study_instance,'wb+') as output_instance:
#                         shutil.copyfileobj(instance,output_instance)
#                 return output_study_instance
#     except Exception as e:
#         print('input_tuple',input_tuple)
#         self.retry(countdown=5, max_retries=5)  # 重試任務
#     except OSError :
#         self.retry(countdown=30, max_retries=5)
#
#
# @app.task(bind=True,rate_limit='100/s',priority=70)
# def dicom_file_processing(self,input_tuple,output_nifti_path,post_process_manager):
#     if input_tuple is None:
#         return
#
#     output_study_instance: pathlib.Path = input_tuple
#     output_study_path = output_study_instance.parent.parent
#     post_process_manager.post_process(output_study_path)
#     if check_dicom_instance_number_at_last(output_study_instance):
#         print('dicom_file_processing input_tuple ', input_tuple)
#         print('dicom_file_processing output_nifti_path ', output_nifti_path)
#         print('dicom_file_processing post_process_manager ', post_process_manager)
#         output_study_series = output_study_instance.parent
#         print('check_dicom_instance_number_at_last',output_study_series)
#         dicom_2_nii_file.apply_async(args=(output_study_series, output_nifti_path))
#     # try:
#     #     pass
#     # except :
#     #     self.retry(countdown=5, max_retries=5)  # 重試任務
#
#
# @app.task(bind=True,rate_limit='0.3/s',priority=60)
# def dicom_2_nii_file(self,study_path,output_path):
#     print('dicom_2_nii_file self ', self)
#     print('dicom_2_nii_file study_path ', study_path)
#     print('dicom_2_nii_file output_path ', output_path)



#         self.retry(countdown=5, max_retries=5)  # 重試任務
#
#
# @app.task(bind=True,rate_limit='100/s',priority=80)
# def copy_dicom_file(self,input_tuple,instance_path,output_path):
#     try:
#         if input_tuple is None or len(input_tuple[0]) == 0 or input_tuple[1] is None:
#             return
#         rename_series = input_tuple[0]
#         output_study = input_tuple[1]
#         output_study_series = output_path.joinpath(output_study,rename_series)
#         output_study_series.mkdir(exist_ok=True,parents=True)
#         os.makedirs(output_study_series, exist_ok=True)
#         output_study_instance:pathlib.Path = output_study_series.joinpath(instance_path.name)
#         if output_study_series.is_dir():
#             if output_study_instance.exists():
#                 return
#             else:
#                 with open(instance_path,mode='rb') as instance:
#                     with open(output_study_instance,'wb+') as output_instance:
#                         shutil.copyfileobj(instance,output_instance)
#                 return output_study_instance
#     except Exception as e:
#         print('input_tuple',input_tuple)
#         self.retry(countdown=5, max_retries=5)  # 重試任務
#     except OSError :
#         self.retry(countdown=30, max_retries=5)



def rename_dicom_file(instance_path,
                      processing_strategy_list,
                      modality_processing_strategy,
                      mr_acquisition_type_processing_strategy):
    with open(instance_path, mode='rb') as dcm:
        dicom_ds = dcmread(dcm, stop_before_pixels=True)
    if dicom_ds is None:
        return tuple(['',''])
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
                        return series_enum.value,output_study



def copy_dicom_file(input_tuple,instance_path,output_path):
    if input_tuple is None or len(input_tuple[0]) == 0 or input_tuple[1] is None:
        return
    rename_series = input_tuple[0]
    output_study = input_tuple[1]
    output_study_series = output_path.joinpath(output_study,rename_series)
    output_study_series.mkdir(exist_ok=True,parents=True)
    os.makedirs(output_study_series, exist_ok=True)
    output_study_instance:pathlib.Path = output_study_series.joinpath(instance_path.name)
    if output_study_series.is_dir():
        if output_study_instance.exists():
            return
        else:
            with open(instance_path,mode='rb') as instance:
                with open(output_study_instance,'wb+') as output_instance:
                    shutil.copyfileobj(instance,output_instance)
            return output_study_instance


def file_processing(study_folder_path,post_process_manager):
    if study_folder_path is None:
        return
    post_process_manager.post_process(study_folder_path)


# @app.task(bind=True,rate_limit='1/s',priority=55)
@shared_task(bind=True,rate_limit='16/s',priority=55)
def call_dcm2niix(self,output_series_file_path,output_series_path,series_path):
    output_series_path.parent.mkdir(exist_ok=True, parents=True)
    cmd_str = f'{dcm2niix.bin} -z y -f {output_series_path.name} -o {output_series_path.parent} {series_path}'
    process = subprocess.Popen(args=cmd_str, cwd='/', shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    pattern = re.compile(r"DICOM as (.*)\s[(]", flags=re.MULTILINE)
    match_result = pattern.search(stdout.decode())
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
            # dcm2niix_json_path.rename(output_series_json_path)
        except FileExistsError:
            print(rf'FileExistsError {series_path}')
    return output_series_path.name


# @app.task(bind=True,rate_limit='1/s',priority=55)
@shared_task(bind=True,rate_limit='8/s',priority=55)
def dicom_2_nii_file(self,dicom_study_folder_path,nifti_output_path):
    FILE_SIZE = 500
    if dicom_study_folder_path is None:
        return
    series_list = list(filter(lambda series_path: series_path.name != '.meta', dicom_study_folder_path.iterdir()))
    workflows = []
    for series_path in series_list:
        if series_path.name in Dicm2NiixConverter.exclude_set:
            continue
        output_series_path = pathlib.Path(
            f'{str(series_path).replace(str(dicom_study_folder_path.parent),str(nifti_output_path))}')
        output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
        if output_series_file_path.exists():
            if output_series_file_path.stat().st_size < FILE_SIZE:
                output_series_file_path.unlink()
                workflows.append(call_dcm2niix.s(output_series_file_path, output_series_path, series_path))
            else:
                continue
        else:
            workflows.append(call_dcm2niix.s(output_series_file_path,output_series_path,series_path))
    job = group(workflows).apply()
    nifti_study_folder_path = nifti_output_path.joinpath(dicom_study_folder_path.name)
    file_processing(nifti_study_folder_path,ConvertManager.nifti_post_process_manager)
    return job



# @app.task(bind=True,rate_limit='500/s',priority=58)
@shared_task(bind=True,rate_limit='500/s',priority=58,ignore_result=True)
def process_instances(self, instances_list,output_dicom_path):
    try:
        for instance in instances_list:
            rename_dicom_file_tuple = rename_dicom_file(instance,
                                            ConvertManager.processing_strategy_list,
                                            ConvertManager.modality_processing_strategy,
                                            ConvertManager.mr_acquisition_type_processing_strategy)
            copy_dicom_file_tuple = copy_dicom_file(rename_dicom_file_tuple,
                                                    instance,
                                                    output_dicom_path)
    except:
        self.retry(countdown=60, max_retries=5)  # 重試任務


@app.task(rate_limit='300/s',priority=60)
def process_dir(sub_dir:pathlib.Path, output_dicom_path:pathlib.Path):
    instances_list = list(sub_dir.rglob('*.dcm'))
    workflows = []
    for chunk in chunk_list(instances_list, 64):
        workflows.append(process_instances.s(chunk,output_dicom_path))
    # job = group(workflows).apply()
    from celery.result import GroupResult
    res :GroupResult = group(workflows).apply()
    print('res',res,res.successful())

    if len(instances_list) > 0:
        instance_path: pathlib.Path = instances_list[0]
        with open(instance_path, mode='rb') as dcm:

            dicom_ds = dcmread(dcm, stop_before_pixels=True)
            study_folder_name = get_study_folder_name(dicom_ds)
            study_folder_path = output_dicom_path.joinpath(study_folder_name)
            print('instance_path', instance_path)
            print('study_folder_name', study_folder_name)
            print('study_folder_path', study_folder_path)
            file_processing(study_folder_path,
                                  ConvertManager.dicom_post_process_manager)
            dicom_study_folder_path = study_folder_path
        return dicom_study_folder_path


@app.task
def celery_workflow(input_dicom_str, output_dicom_str, output_nifti_str):
    if isinstance(input_dicom_str,str):
        input_dicom_path: pathlib.Path = pathlib.Path(input_dicom_str)
    else:
        input_dicom_path = input_dicom_str
    if isinstance(output_dicom_str,str):
        output_dicom_path: pathlib.Path = pathlib.Path(output_dicom_str)
    else:
        output_dicom_path = output_dicom_str
    if isinstance(output_nifti_str,str):
        output_nifti_path: pathlib.Path = pathlib.Path(output_nifti_str)
    else:
        output_nifti_path = output_nifti_str

    workflows = []
    is_dir_flag = all(list(map(lambda x: x.is_dir(), input_dicom_path.iterdir())))
    print('is_dir_flag', is_dir_flag)
    if is_dir_flag:
        for sub_dir in list(input_dicom_path.iterdir()):
            workflow = chain(process_dir.s(sub_dir,output_dicom_path),
                             dicom_2_nii_file.s(output_nifti_path))
            workflows.append(workflow)
    else:
        for sub_dir in list(input_dicom_path.iterdir()):
            if sub_dir.is_dir():
                for sub_dir in list(input_dicom_path.iterdir()):
                    workflow = chain(process_dir.s(sub_dir,output_dicom_path),
                                     dicom_2_nii_file.s(output_nifti_path)
                                     )
                    workflows.append(workflow)
    job = group(workflows).apply_async()
    return job


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

    # def __init__(self, input_dicom_path, output_dicom_path, output_nifti_path):
    #     self._input_dicom_path = pathlib.Path(input_dicom_path)
    #     self.output_dicom_path = pathlib.Path(output_dicom_path)
    #     self.output_nifti_path = pathlib.Path(output_nifti_path)
    #
    # def run(self):
    #     is_dir_flag = all(list(map(lambda x: x.is_dir(), self.input_dicom_path.iterdir())))
    #     print('is_dir_flag', is_dir_flag)
    #     if is_dir_flag:
    #         for sub_dir in list(self.input_dicom_path.iterdir()):
    #             instances_list = list(sub_dir.rglob('*.dcm'))
    #             for chunk in chunk_list(instances_list, 128):
    #                 self.process_instances(chunk)
    #     else:
    #         instances_list = list(self.input_dicom_path.rglob('*.dcm'))
    #         # Process in chunks of 128 instances
    #         for chunk in chunk_list(instances_list, 128):
    #             self.process_instances(chunk)
    #
    # def process_instances(self, instances_list):
    #     workflows = []
    #     for instance in instances_list:
    #         workflow = chain(rename_dicom_file.s(instance,
    #                                              self.processing_strategy_list,
    #                                              self.modality_processing_strategy,
    #                                              self.mr_acquisition_type_processing_strategy),
    #                          copy_dicom_file.s(instance, self.output_dicom_path),
    #                          dicom_file_processing.s(self.output_nifti_path,self.post_process_manager)
    #                          )
    #         workflows.append(workflow)
    #
    #     # Execute the workflows as a group
    #     job = group(workflows).apply_async()
    #     job.get()  # Wait for all tasks to complete
    #
    #
    # @property
    # def input_dicom_path(self):
    #     return self._input_dicom_path
    #
    # @input_dicom_path.setter
    # def input_dicom_path(self, value):
    #     self._input_dicom_path = pathlib.Path(value)


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


