import re
import shutil
import os
import pathlib
import traceback
from typing import List

import dcm2niix
from pydicom import dcmread
from pydicom.dataset import FileDataset
from pydicom.errors import InvalidDicomError, BytesLengthException

from celery import Celery, chain, group
from . import app

from code_ai.dicom2nii.convert import ModalityProcessingStrategy,MRAcquisitionTypeProcessingStrategy, MRRenameSeriesProcessingStrategy
from code_ai.dicom2nii.convert import DwiProcessingStrategy, ADCProcessingStrategy,EADCProcessingStrategy,SWANProcessingStrategy
from code_ai.dicom2nii.convert import ESWANProcessingStrategy,MRABrainProcessingStrategy,MRANeckProcessingStrategy
from code_ai.dicom2nii.convert import MRAVRBrainProcessingStrategy,MRAVRNeckProcessingStrategy,T1ProcessingStrategy
from code_ai.dicom2nii.convert import T2ProcessingStrategy, ASLProcessingStrategy,DSCProcessingStrategy
from code_ai.dicom2nii.convert import RestingProcessingStrategy, DTIProcessingStrategy, CVRProcessingStrategy
from code_ai.dicom2nii.convert import NullEnum
from code_ai.dicom2nii.convert import MRSeriesRenameEnum
from ..dicom2nii.convert.dicom_rename_mr_postprocess import PostProcessManager


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


def get_series_folder_list(study_path: pathlib.Path) -> List[pathlib.Path]:
        # series_folder_list = list(study_path.iterdir())
        series_folder_list = []
        for series_folder in study_path.iterdir():
            if series_folder.is_dir() and series_folder.name != '.meta':
                if series_folder.name not in self.exclude_dicom_series:
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



@app.task(bind=True,rate_limit='250/s',priority=100)
def rename_dicom_file(self,instance_path, processing_strategy_list,modality_processing_strategy,mr_acquisition_type_processing_strategy):
    try:
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
        return tuple(['',''])
    except (InvalidDicomError, BytesLengthException):
        print(f"Invalid DICOM file: {instance_path}")
        return None
    except Exception as e:
        self.retry(countdown=5, max_retries=5)  # 重試任務


# @app.task(bind=True,rate_limit='8/m')
# def call_dicom_to_nii(self,output_study_series,output_nifti_path):
#     try:
#         pass
#     except :
#         print('self', self)
#         self.retry(countdown=60, max_retries=5)


@app.task(bind=True,rate_limit='50/s',priority=80)
def copy_dicom_file(self,input_tuple,instance_path,output_path):
    try:
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
    except Exception as e:
        print('input_tuple',input_tuple)
        self.retry(countdown=5, max_retries=5)  # 重試任務
    except OSError :
        self.retry(countdown=30, max_retries=5)


@app.task(bind=True,rate_limit='50/s',priority=70)
def dicom_file_processing(self,input_tuple,output_nifti_path,post_process_manager):
    if input_tuple is None:
        return

    output_study_instance: pathlib.Path = input_tuple
    post_process_manager.post_process()
    if check_dicom_instance_number_at_last(output_study_instance):
        # output_study_instance.parts
        # in 0.06680197801324539s: None
        # [2025-01-22 11:54:20,155: WARNING/MainProcess] dicom_file_processing input_tuple
        # [2025-01-22 11:54:20,155: WARNING/MainProcess]
        # [2025-01-22 11:54:20,155: WARNING/MainProcess] /mnt/e/rename_dicom/02695350_20240109_MR_21210300104/T2_COR/1.2.840.113619.2.475.5554020.7707121.17035.1704705833.242.dcm
        # [2025-01-22 11:54:20,155: WARNING/MainProcess] dicom_file_processing output_nifti_path
        # [2025-01-22 11:54:20,155: WARNING/MainProcess]
        # [2025-01-22 11:54:20,155: WARNING/MainProcess] /mnt/e/rename_nifti
        # [2025-01-22 11:54:20,155: WARNING/MainProcess] dicom_file_processing post_process_manager
        # [2025-01-22 11:54:20,155: WARNING/MainProcess]
        # [2025-01-22 11:54:20,155: WARNING/MainProcess] <code_ai.dicom2nii.convert.dicom_rename_mr_postprocess.PostProcessManager object at 0x7f42443e93c0>
        print('dicom_file_processing input_tuple ', input_tuple)
        print('dicom_file_processing output_nifti_path ', output_nifti_path)
        print('dicom_file_processing post_process_manager ', post_process_manager)


        output_study_series = output_study_instance.parent
        print('check_dicom_instance_number_at_last',output_study_series)
        # dicom_2_nii_file.apply_async(args=(output_study_series, output_nifti_path))
    # try:
    #     pass
    # except :
    #     self.retry(countdown=5, max_retries=5)  # 重試任務


@app.task(bind=True,rate_limit='0.28/s',priority=60)
def dicom_2_nii_file(self,input_tuple,study_path,output_path):
    # Celery 中的 rate_limit 使用格式為 {次數}/{時間}，時間單位可以是秒（s）、分（m）、小時（h）。
    # rate_limit: 設置為 '8/30s'，表示每 30 秒最多執行 8 個任務。
    try:
        pass
    except Exception as e:
        print('input_tuple',input_tuple)
        self.retry(countdown=5, max_retries=5)  # 重試任務
    except OSError :
        self.retry(countdown=30, max_retries=5)


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
    post_process_manager = PostProcessManager()

    def __init__(self, input_dicom_path, output_dicom_path, output_nifti_path):
        self._input_dicom_path = pathlib.Path(input_dicom_path)
        self.output_dicom_path = pathlib.Path(output_dicom_path)
        self.output_nifti_path = pathlib.Path(output_nifti_path)

    def run(self):
        is_dir_flag = all(list(map(lambda x: x.is_dir(), self.input_dicom_path.iterdir())))
        print('is_dir_flag', is_dir_flag)
        if is_dir_flag:
            for sub_dir in list(self.input_dicom_path.iterdir()):
                instances_list = list(sub_dir.rglob('*.dcm'))
                for chunk in chunk_list(instances_list, 128):
                    self.process_instances(chunk)
        else:
            instances_list = list(self.input_dicom_path.rglob('*.dcm'))
            # Process in chunks of 128 instances
            for chunk in chunk_list(instances_list, 128):
                self.process_instances(chunk)

    def process_instances(self, instances_list):
        workflows = []
        for instance in instances_list:
            workflow = chain(rename_dicom_file.s(instance,
                                                 self.processing_strategy_list,
                                                 self.modality_processing_strategy,
                                                 self.mr_acquisition_type_processing_strategy),
                             copy_dicom_file.s(instance, self.output_dicom_path),
                             dicom_file_processing.s(self.output_nifti_path,self.post_process_manager)
                             )
            workflows.append(workflow)

        # Execute the workflows as a group
        job = group(workflows).apply_async()
        job.get()  # Wait for all tasks to complete


    @property
    def input_dicom_path(self):
        return self._input_dicom_path

    @input_dicom_path.setter
    def input_dicom_path(self, value):
        self._input_dicom_path = pathlib.Path(value)


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

    def __init__(self, input_path, output_path):
        """
        Initialize the Dcm2NiixConverter.

        Parameters
        ----------
        input_path : str or pathlib.Path
            Path to the input DICOM files.
        output_path : str or pathlib.Path
            Path to the output directory for NIfTI files.
        """
        self.input_path = pathlib.Path(input_path)
        self.output_path = pathlib.Path(output_path)

    def run_cmd(self, output_series_path, series_path):
        """
        Run the dcm2niix command to convert DICOM to NIfTI.

        Parameters
        ----------
        output_series_path : pathlib.Path
            Path to the output series.
        series_path : pathlib.Path
            Path to the input DICOM series.

        Returns
        -------
        str
            The result of the conversion.
        """
        output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
        cmd_str = f'{dcm2niix.bin} -z y -f {output_series_path.name} -o {output_series_path.parent} {series_path}'

        completed_process = subprocess.run(cmd_str, capture_output=True)
        pattern = re.compile(r"DICOM as (.*)\s[(]", flags=re.MULTILINE)
        match_result = pattern.search(completed_process.stdout.decode())
        str_result = match_result.groups()[0]
        dcm2niix_output_path = pathlib.Path(f'{str_result}.nii.gz')

        if dcm2niix_output_path.name != output_series_path:
            try:
                # Rename the output file and corresponding JSON file
                dcm2niix_output_path.rename(output_series_file_path)
                dcm2niix_json_path = pathlib.Path(str(dcm2niix_output_path).replace('.nii.gz', '.json'))
                output_series_json_path = pathlib.Path(str(output_series_file_path).replace('.nii.gz', '.json'))
                dcm2niix_json_path.rename(output_series_json_path)
            except FileExistsError:
                print(rf'FileExistsError {series_path}')
        return str_result

    # def copy_meta_dir(self, study_path: pathlib.Path):
    #     meta_path = study_path.joinpath('.meta')
    #     output_study_path = pathlib.Path(f'{str(study_path).replace(str(study_path.parent), str(self.output_path))}')
    #     if meta_path.exists():
    #         shutil.copytree(meta_path,output_study_path.joinpath('.meta'),dirs_exist_ok=True)
    #
    # def convert_dicom_to_nifti(self, executor: Executor = None):
    #     """
    #     Convert DICOM files to NIfTI format.
    #
    #     Parameters
    #     ----------
    #     executor : concurrent.futures.Executor, optional
    #         Executor for parallel execution.
    #     """
    #     instances = list(self.input_path.rglob('*.dcm'))
    #     study_list = list(set(list(map(lambda x: x.parent.parent, instances))))
    #     future_list = []
    #     for study_path in study_list:
    #         series_list = list(filter(lambda series_path : series_path.name != '.meta' ,study_path.iterdir()))
    #         for series_path in series_list:
    #             if series_path.name in self.exclude_set:
    #                 continue
    #             output_series_path = pathlib.Path(
    #                 f'{str(series_path).replace(str(study_path.parent), str(self.output_path))}')
    #             output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
    #             if output_series_file_path.exists():
    #                 print(output_series_file_path)
    #                 continue
    #             else:
    #                 os.makedirs(output_series_path.parent, exist_ok=True)
    #                 if executor:
    #                     future = executor.submit(self.run_cmd, output_series_path, series_path)
    #                     future_list.append(future)
    #                 else:
    #                     self.run_cmd(output_series_path=output_series_path, series_path=series_path)
    #         self.copy_meta_dir(study_path=study_path)