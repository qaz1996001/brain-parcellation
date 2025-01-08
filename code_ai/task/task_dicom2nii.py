import shutil
import os
import pathlib
import traceback
from typing import List

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

@app.task
def read_dicom_file(instance_path):
    try:
        dicom_ds = dcmread(str(instance_path), stop_before_pixels=True)
        return dicom_ds
    except (InvalidDicomError, BytesLengthException):
        print(f"Invalid DICOM file: {instance_path}")
        return None


@app.task
def get_output_study(dicom_ds, output_path):
    if dicom_ds is None:
        return None
    study_folder_name = get_study_folder_name(dicom_ds)
    if not study_folder_name:
        return None
    output_study = output_path.joinpath(study_folder_name)
    return output_study


@app.task
def rename_dicom_file(dicom_ds, processing_strategy_list,modality_processing_strategy,mr_acquisition_type_processing_strategy):
    if dicom_ds is None:
        return ''
    # Simulating renaming logic
    modality_enum = modality_processing_strategy.process(dicom_ds=dicom_ds)
    mr_acquisition_type_enum = mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
    for processing_strategy in processing_strategy_list:
        if modality_enum == processing_strategy.modality:
            for mr_acquisition_type in processing_strategy.mr_acquisition_type:
                if mr_acquisition_type_enum == mr_acquisition_type:
                    series_enum = processing_strategy.process(dicom_ds=dicom_ds)
                    if series_enum is not NullEnum.NULL:
                        return series_enum.value
    return ''


@app.task
def copy_dicom_file(instance_path, output_study, rename_series):
    if len(rename_series) == 0 or output_study is None:
        return
    os.makedirs(output_study, exist_ok=True)
    output_study_series = output_study.joinpath(rename_series)
    os.makedirs(output_study_series, exist_ok=True)
    output_study_instance = output_study_series.joinpath(instance_path.name)
    shutil.copyfile(instance_path, output_study_instance)


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
        if is_dir_flag:
            for sub_dir in self.input_path.iterdir():
                instances_list = list(sub_dir.rglob('*.dcm'))
                self.process_instances(instances_list, sub_dir.name)
        else:
            instances_list = list(self.input_path.rglob('*.dcm'))
            # Process in chunks of 1000 instances
            for chunk in chunk_list(instances_list, 1000):
                self.process_instances(chunk, self.input_path.name)

    def process_instances(self, instances_list, dir_name):
        workflows = []
        for instance in instances_list:
            workflow = chain(
                read_dicom_file.s(instance),
                get_output_study.s(self.output_path),
                rename_dicom_file.s(self.processing_strategy_list),
                copy_dicom_file.s(instance)
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

