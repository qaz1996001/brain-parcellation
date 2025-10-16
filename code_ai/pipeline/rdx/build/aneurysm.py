from typing import Dict, Any, Union
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir

from pipeline.rdx.schema import AneurysmDetectionItem, AneurysmDetectionResponse
from . import PredictionBaseBuilder


class AneurysmDetectionBuilder(PredictionBaseBuilder[AneurysmDetectionItem, AneurysmDetectionResponse]):
    """
    動脈瘤檢測結果 Builder
    """

    model_class = AneurysmDetectionResponse

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
            'label': prediction_result.get('label', 'aneurysm'),
            'type': prediction_result['type'],
            'location': prediction_result['location'],
            'diameter': prediction_result['diameter'],
            'main_seg_slice': prediction_result['main_seg_slice'],
            'probability': prediction_result['probability'],
            'pitch_angle': prediction_result['pitch_angle'],
            'yaw_angle': prediction_result['yaw_angle'],
            'mask_index': prediction_result['mask_index'],
            'sub_location': prediction_result['sub_location'],
        }

        return AneurysmDetectionItem.model_validate(detection_dict)