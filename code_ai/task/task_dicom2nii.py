import re
import shutil
import os
import pathlib
import traceback
from typing import List

import dcm2niix
from pydicom import dcmread
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




@app.task(bind=True,rate_limit='250/s')
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


def get_output_study(dicom_ds):
    if dicom_ds is None:
        return None
    study_folder_name = get_study_folder_name(dicom_ds)
    if not study_folder_name:
        return None
    return study_folder_name


@app.task(bind=True,rate_limit='100/s')
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

    except Exception as e:
        print('input_tuple',input_tuple)
        self.retry(countdown=5, max_retries=5)  # 重試任務
    except OSError :
        self.retry(countdown=30, max_retries=5)


@app.task(bind=True,rate_limit='10/s')
def dicom_2_nii_file(self,input_tuple,study_path,output_path):
    try:
        pass
    except Exception as e:
        print('input_tuple',input_tuple)
        self.retry(countdown=5, max_retries=5)  # 重試任務
    except OSError :
        self.retry(countdown=30, max_retries=5)


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

    def __init__(self, input_path, output_path):
        self._input_path = pathlib.Path(input_path)
        self.output_path = pathlib.Path(output_path)

    def run(self):
        is_dir_flag = all(list(map(lambda x: x.is_dir(), self.input_path.iterdir())))
        print('is_dir_flag', is_dir_flag)
        if is_dir_flag:
            for sub_dir in list(self.input_path.iterdir()):
                instances_list = list(sub_dir.rglob('*.dcm'))
                for chunk in chunk_list(instances_list, 64):
                    self.process_instances(chunk, sub_dir.name)
        else:
            instances_list = list(self.input_path.rglob('*.dcm'))
            # Process in chunks of 1000 instances
            for chunk in chunk_list(instances_list, 64):
                self.process_instances(chunk, self.input_path.name)

    def process_instances(self, instances_list, dir_name):
        workflows = []
        for instance in instances_list:
            workflow = chain(rename_dicom_file.s(instance,
                                                 self.processing_strategy_list,
                                                 self.modality_processing_strategy,
                                                 self.mr_acquisition_type_processing_strategy),
                             copy_dicom_file.s(instance,self.output_path)
            )
            workflows.append(workflow)

        # Execute the workflows as a group
        job = group(workflows).apply_async()
        job.get()  # Wait for all tasks to complete

    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self, value):
        self._input_path = pathlib.Path(value)


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


#
# class Dicm2NiixConverter:
#     def __init__(self, input_path, output_path):
#         """
#         Initialize the Dcm2NiixConverter.
#
#         Parameters
#         ----------
#         input_path : str or pathlib.Path
#             Path to the input DICOM files.
#         output_path : str or pathlib.Path
#             Path to the output directory for NIfTI files.
#         """
#         self.input_path = pathlib.Path(input_path)
#         self.output_path = pathlib.Path(output_path)
#         self.exclude_set = {
#             MRSeriesRenameEnum.MRAVR_BRAIN.value,
#             MRSeriesRenameEnum.MRAVR_NECK.value,
#
#             # DSCSeriesRenameEnum.DSC.value,
#             # DSCSeriesRenameEnum.rCBV.value,
#             # DSCSeriesRenameEnum.rCBF.value,
#             # DSCSeriesRenameEnum.MTT.value,
#             #
#             # ASLSEQSeriesRenameEnum.ASLSEQ.value,
#             # ASLSEQSeriesRenameEnum.ASLPROD.value,
#             #
#             # ASLSEQSeriesRenameEnum.ASLSEQATT.value,
#             # ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR.value,
#             #
#             # ASLSEQSeriesRenameEnum.ASLSEQCBF.value,
#             # ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR.value,
#             #
#             # ASLSEQSeriesRenameEnum.ASLSEQPW.value,
#         }
#
#     def run_cmd(self, output_series_path, series_path):
#         """
#         Run the dcm2niix command to convert DICOM to NIfTI.
#
#         Parameters
#         ----------
#         output_series_path : pathlib.Path
#             Path to the output series.
#         series_path : pathlib.Path
#             Path to the input DICOM series.
#
#         Returns
#         -------
#         str
#             The result of the conversion.
#         """
#         output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
#         cmd_str = f'{dcm2niix.bin} -z y -f {output_series_path.name} -o {output_series_path.parent} {series_path}'
#
#         completed_process = subprocess.run(cmd_str, capture_output=True)
#         pattern = re.compile(r"DICOM as (.*)\s[(]", flags=re.MULTILINE)
#         match_result = pattern.search(completed_process.stdout.decode())
#         str_result = match_result.groups()[0]
#         dcm2niix_output_path = pathlib.Path(f'{str_result}.nii.gz')
#
#         if dcm2niix_output_path.name != output_series_path:
#             try:
#                 # Rename the output file and corresponding JSON file
#                 dcm2niix_output_path.rename(output_series_file_path)
#                 dcm2niix_json_path = pathlib.Path(str(dcm2niix_output_path).replace('.nii.gz', '.json'))
#                 output_series_json_path = pathlib.Path(str(output_series_file_path).replace('.nii.gz', '.json'))
#                 dcm2niix_json_path.rename(output_series_json_path)
#             except FileExistsError:
#                 print(rf'FileExistsError {series_path}')
#         return str_result
#
#     def copy_meta_dir(self, study_path: pathlib.Path):
#         meta_path = study_path.joinpath('.meta')
#         output_study_path = pathlib.Path(f'{str(study_path).replace(str(study_path.parent), str(self.output_path))}')
#         if meta_path.exists():
#             shutil.copytree(meta_path,output_study_path.joinpath('.meta'),dirs_exist_ok=True)
#
#     def convert_dicom_to_nifti(self, executor: Executor = None):
#         """
#         Convert DICOM files to NIfTI format.
#
#         Parameters
#         ----------
#         executor : concurrent.futures.Executor, optional
#             Executor for parallel execution.
#         """
#         instances = list(self.input_path.rglob('*.dcm'))
#         study_list = list(set(list(map(lambda x: x.parent.parent, instances))))
#         future_list = []
#         for study_path in study_list:
#             series_list = list(filter(lambda series_path : series_path.name != '.meta' ,study_path.iterdir()))
#             for series_path in series_list:
#                 if series_path.name in self.exclude_set:
#                     continue
#                 output_series_path = pathlib.Path(
#                     f'{str(series_path).replace(str(study_path.parent), str(self.output_path))}')
#                 output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
#                 if output_series_file_path.exists():
#                     print(output_series_file_path)
#                     continue
#                 else:
#                     os.makedirs(output_series_path.parent, exist_ok=True)
#                     if executor:
#                         future = executor.submit(self.run_cmd, output_series_path, series_path)
#                         future_list.append(future)
#                     else:
#                         self.run_cmd(output_series_path=output_series_path, series_path=series_path)
#             self.copy_meta_dir(study_path=study_path)
