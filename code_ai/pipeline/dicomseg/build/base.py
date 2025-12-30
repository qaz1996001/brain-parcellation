import abc
from typing import Dict, Any, List, Union, Generic
from typing_extensions import Self
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir
from code_ai.pipeline.dicomseg.schema import DetectionT, PredictionT

# 定義泛型變數


class PredictionBaseBuilder(Generic[DetectionT, PredictionT], metaclass=abc.ABCMeta):
    """
    用於構建 PredictionBaseResponse 的抽象基類

    泛型參數:
        DetectionT: DetectionsBaseResponse 的子類
        PredictionT: PredictionBaseResponse 的子類
    """

    model_class: type[PredictionT]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._detections: List[DetectionT] = []
        self._prediction_dict: Dict[str, Any] = {}
        self.reset()

    def reset(self):
        """重置 builder 狀態"""
        self._prediction_dict: Dict[str, Any] = {}
        self._detections: List[DetectionT] = []

    def set_patient_info(self, source_images: List[Union[FileDataset, DicomDir]]) -> Self:
        """從 DICOM 提取患者基本資訊"""
        if not source_images:
            raise ValueError("source_images cannot be empty")

        dicom_ds = source_images[0]

        # 提取患者和檢查資訊
        patient_id = dicom_ds.get((0x0010, 0x0020)).value
        study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
        series_instance_uid = dicom_ds.get((0x0020, 0x000E)).value

        self._prediction_dict.update({
            'patient_id': patient_id,
            'study_instance_uid': study_instance_uid,
            'series_instance_uid': series_instance_uid,
        })

        return self

    def set_model_id(self, model_id: str) -> Self:
        """設定模型 ID"""
        self._prediction_dict['model_id'] = model_id
        return self

    @abc.abstractmethod
    def build_detection(self,
                        source_image: Union[FileDataset, DicomDir],
                        prediction_result: Dict[str, Any],
                        *args, **kwargs) -> DetectionT:
        """
        構建單個 detection item

        Args:
            source_image: DICOM 影像
            prediction_result: 預測結果字典

        Returns:
            DetectionT: Detection 物件
        """
        pass

    def set_detections(self,
                       source_images: List[Union[FileDataset, DicomDir]],
                       prediction_results: List[Dict[str, Any]],
                       *args, **kwargs) -> Self:
        """
        批次構建 detections

        Args:
            source_images: DICOM 影像列表
            prediction_results: 預測結果列表
        """

        for source_image, pred_result in zip(source_images, prediction_results):
            print('pred_result',pred_result)
            detection = self.build_detection(source_image, pred_result, *args, **kwargs)
            self._detections.append(detection)

        return self

    def build(self) -> PredictionT:
        """
        構建最終的 Prediction 物件

        Returns:
            PredictionT: 完整的預測結果物件
        """
        # 將 detections 加入字典
        self._prediction_dict['detections'] = self._detections

        # 驗證並創建實例
        instance = self.model_class.model_validate(self._prediction_dict)

        # 重置 builder
        self.reset()

        return instance
