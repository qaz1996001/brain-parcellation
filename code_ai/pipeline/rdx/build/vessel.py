import pathlib
from typing import Dict, Any, Union, List

import numpy as np
import pandas as pd
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir

from code_ai.pipeline.dicomseg import utils

from code_ai.pipeline.rdx.schema import VesselDilatedDetectionItem,VesselDilatedDetectionResponse
from .base import PredictionBaseBuilder


class VesselDilatedBuilder(PredictionBaseBuilder[VesselDilatedDetectionItem, VesselDilatedDetectionResponse]):
    """
    動脈瘤檢測結果 Builder
    """

    model_class = VesselDilatedDetectionResponse

    @staticmethod
    def use_create_dicom_seg_file(path_nii: pathlib.Path,
                                  series_name: str,
                                  output_folder: pathlib.Path,
                                  image: Any,
                                  first_dcm: FileDataset | DicomDir,
                                  source_images: List[FileDataset | DicomDir], ):
        # Load prediction data from NIfTI file
        if series_name == "MRA_BRAIN":
            new_nifti_array = utils.get_array_to_dcm_axcodes(path_nii.joinpath(f'Vessel.nii.gz'))

        else:
            new_nifti_array = utils.get_array_to_dcm_axcodes(path_nii.joinpath(f'{series_name}_vessel.nii.gz'))

        series_name_add_vessel = "{}_Vessel".format(series_name)
        pred_data_unique = np.unique(new_nifti_array)
        if len(pred_data_unique) < 1:
            return None
        else:
            pred_data_unique = pred_data_unique[1:]  # Exclude background value (0)
        result_list = utils.create_dicom_seg_file(pred_data_unique,
                                                  new_nifti_array,
                                                  series_name_add_vessel,
                                                  output_folder,
                                                  image,
                                                  first_dcm,
                                                  source_images)
        return result_list

    def build_detection(self,
                        source_image: Union[FileDataset, DicomDir],
                        prediction_result: Dict[str, Any] = {},
                        *args, **kwargs) -> VesselDilatedDetectionItem:
        """
        構建單個動脈瘤檢測項目

        Args:
            source_image: DICOM 影像

        Returns:
            AneurysmDetectionItem
        """
        # 從 DICOM 提取必要資訊
        series_instance_uid = source_image.get((0x0020, 0x000E)).value
        sop_instance_uid = source_image.get((0x0008, 0x0018)).value

        # 構建 detection 字典
        detection_dict = {
            'series_instance_uid': series_instance_uid,
            'sop_instance_uid': sop_instance_uid,
            'label':  'vessel',
        }

        return VesselDilatedDetectionItem.model_validate(detection_dict)

