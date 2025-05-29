import logging
import re
import shutil
import os
import pathlib
from typing import List


from pydicom import dcmread

from code_ai.dicom2nii.convert import ModalityProcessingStrategy, MRAcquisitionTypeProcessingStrategy, \
    MRRenameSeriesProcessingStrategy
from code_ai.dicom2nii.convert import DwiProcessingStrategy, ADCProcessingStrategy, EADCProcessingStrategy, \
    SWANProcessingStrategy
from code_ai.dicom2nii.convert import ESWANProcessingStrategy, MRABrainProcessingStrategy, MRANeckProcessingStrategy
from code_ai.dicom2nii.convert import MRAVRBrainProcessingStrategy, MRAVRNeckProcessingStrategy, T1ProcessingStrategy
from code_ai.dicom2nii.convert import T2ProcessingStrategy, ASLProcessingStrategy, DSCProcessingStrategy
from code_ai.dicom2nii.convert import RestingProcessingStrategy, DTIProcessingStrategy, CVRProcessingStrategy

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


if __name__ == '__main__':
    # instance_path = pathlib.Path('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/4fd61075-0d21921e-6f493a03-06f34542-791c1ef8/MR000518.dcm')
    instance_path = pathlib.Path('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/580ada15-b16f5062-e11479b7-3cc015a5-ca9d5b71/MR000269.dcm')

    # /mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/580ada15-b16f5062-e11479b7-3cc015a5-ca9d5b71/MR000269.dcm
    with open(instance_path, mode='rb') as dcm:
        dicom_ds = dcmread(dcm, stop_before_pixels=True,force=True)
    modality_enum = ConvertManager.modality_processing_strategy.process(dicom_ds=dicom_ds)
    mr_acquisition_type_enum = ConvertManager.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
    for processing_strategy in ConvertManager.processing_strategy_list:
        if modality_enum == processing_strategy.modality:
            for mr_acquisition_type in processing_strategy.mr_acquisition_type:
                if mr_acquisition_type_enum == mr_acquisition_type:
                    series_enum = processing_strategy.process(dicom_ds=dicom_ds)
                    print(series_enum)


    print(dicom_ds)

