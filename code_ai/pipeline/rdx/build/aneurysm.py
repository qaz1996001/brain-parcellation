import pathlib
from typing import Dict, Any, Union, List

import numpy as np
import pandas as pd
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir

from code_ai.pipeline.dicomseg import utils

from code_ai.pipeline.rdx.schema import AneurysmDetectionItem, AneurysmDetectionResponse
from .base import PredictionBaseBuilder


class AneurysmDetectionBuilder(PredictionBaseBuilder[AneurysmDetectionItem, AneurysmDetectionResponse]):
    """
    動脈瘤檢測結果 Builder
    """

    model_class = AneurysmDetectionResponse

    angle_key_mapping = {
        'MIP_Pitch':'pitch_angle',
        'MIP_Yaw':'yaw_angle'
    }

    @staticmethod
    def get_excel_to_pred_json(excel_file_path: str,
                               intput_list: List[Dict[str, Any]]):
        """
        intput_list:Dict[str,str] -> {"series_name": "MIP_Pitch",
             #                        "pred_nii_path": "*.nii.gz"}
        """
        pred_json_list = []
        df = pd.read_excel(excel_file_path)
        aneurysm_number_count = df['Aneurysm_Number'].iloc()[0]
        for intput_dict in intput_list:
            series_name_pred = []
            for aneurysm_number in range(1, aneurysm_number_count + 1):
                pred = utils.get_array_to_dcm_axcodes(intput_dict['pred_nii_path'])
                have_labels = np.where(pred == aneurysm_number)[0]
                sub_location = df[f"{aneurysm_number}_SubLocation"].iloc()[0]
                segment_attribute = {
                    "mask_index": str(aneurysm_number),
                    "mask_name": 'A' + str(aneurysm_number),
                    "diameter": str(round(df[f"{aneurysm_number}_size"].iloc()[0], 1)),
                    "type": 'saccular',
                    "location": str(df[f"{aneurysm_number}_Location"].iloc()[0]),
                    "sub_location": sub_location if pd.notna(sub_location) else "",
                    "probability": round(df[f"{aneurysm_number}_Prob_max"].iloc()[0], 2),
                    "main_seg_slice": int(np.median(have_labels) + 1),
                }
                series_name_pred.append(segment_attribute)
            pred_json_list.append({"series_name": intput_dict['series_name'],
                                   "data": series_name_pred
                                   })
        return pred_json_list


        return pred_json_list

    @staticmethod
    def use_create_dicom_seg_file(path_nii: pathlib.Path,
                                  series_name: str,
                                  output_folder: pathlib.Path,
                                  image: Any,
                                  first_dcm: FileDataset | DicomDir,
                                  source_images: List[FileDataset | DicomDir], ):
        # Load prediction data from NIfTI file
        if series_name == "MRA_BRAIN":
            new_nifti_array = utils.get_array_to_dcm_axcodes(path_nii.joinpath(f'Pred.nii.gz'))
        else:
            new_nifti_array = utils.get_array_to_dcm_axcodes(path_nii.joinpath(f'{series_name}_pred.nii.gz'))
        pred_data_unique = np.unique(new_nifti_array)
        if len(pred_data_unique) < 1:
            return None
        else:
            pred_data_unique = pred_data_unique[1:]  # Exclude background value (0)
        result_list = utils.create_dicom_seg_file(pred_data_unique,
                                                  new_nifti_array,
                                                  series_name,
                                                  output_folder,
                                                  image,
                                                  first_dcm,
                                                  source_images)
        return result_list

    @classmethod
    def merge_pitch_yaw_angle(cls, pred_json_list:List[Dict[str, Any]],series_name:str = 'MRA_BRAIN') -> Dict[str, Any]:

        mra_brain_json = next(filter(lambda x: x['series_name'] == series_name, pred_json_list))
        pitch_yaw_list = list(filter(lambda x: x['series_name'] != series_name, pred_json_list))

        angle_list     = [ list(map(lambda x: {'key':pred_json['series_name'],
                                               x['mask_name']:x['main_seg_slice'] * 3 -3 }, pred_json['data']))for pred_json in pitch_yaw_list]
        mra_brain_json['data'] = [{**brain_json_data[0],
                                   **{cls.angle_key_mapping[xx['key']]: xx[brain_json_data[0]['mask_name']]
                                      for xx in brain_json_data[1:]}} for brain_json_data in zip(mra_brain_json['data'], *angle_list)]

        return mra_brain_json

    def build_detection(self,
                        source_image: Union[FileDataset, DicomDir],
                        prediction_result: Dict[str, Any],
                        *args, **kwargs) -> AneurysmDetectionItem:
        """
        構建單個動脈瘤檢測項目

        Args:
            source_image: DICOM 影像
            prediction_result: 包含以下鍵值的字典:
                - type: 動脈瘤類型
                - location: 位置
                - diameter: 直徑
                - main_seg_slice: 主要分割切片
                - probability: 機率
                - pitch_angle: 俯仰角
                - yaw_angle: 偏航角
                - mask_index: mask 索引
                - sub_location: 子位置

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
            'label': prediction_result.get('mask_name', 'aneurysm'),
            'type': prediction_result['type'],
            'location': prediction_result['location'],
            'diameter': prediction_result['diameter'],
            'main_seg_slice': prediction_result['main_seg_slice'],
            'probability': prediction_result['probability'],
            'pitch_angle': prediction_result.get('pitch_angle', 0),
            'yaw_angle': prediction_result.get('yaw_angle', 0),
            'mask_index': prediction_result['mask_index'],
            'sub_location': prediction_result.get('sub_location', ""),
        }

        return AneurysmDetectionItem.model_validate(detection_dict)

