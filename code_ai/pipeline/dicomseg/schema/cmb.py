import uuid
from typing import List
from uuid import UUID

from pydantic import field_serializer, Field, PositiveFloat, PositiveInt, confloat, model_serializer
from .base import DetectionsBaseResponse, PredictionBaseResponse



class CmbDetectionItem(DetectionsBaseResponse):
    annotated_series_instance_uid : str = Field(...)
    type                          : str = Field(...)
    location                      : str = Field(...)
    diameter                      : PositiveFloat = Field(...)
    main_seg_slice                : PositiveInt   = Field(...)
    probability                   : confloat(ge=0.0, le=1.0, allow_inf_nan=True) = Field(...)
    mask_index                    : int           = Field(...)


    _field_order = ['annotated_series_instance_uid',
                    'series_instance_uid',
                    'sop_instance_uid',
                    'label',
                    'type',
                    'location',
                    'diameter',
                    'main_seg_slice',
                    'probability',
                    'mask_index',
                    ]


    @field_serializer('diameter')
    def serialize_diameter(self, value: float) -> float:
        """序列化時保留 4 位小數"""
        return round(value, 4)

    @model_serializer(mode='wrap')
    def ordered_dump(self, handler):
        data = handler(self)
        return {k: data[k] for k in self._field_order}


class CmbDetectionResponse(PredictionBaseResponse[CmbDetectionItem]):
    inference_id              :UUID = Field(default_factory=uuid.uuid4)
    input_study_instance_uid  : List[str] = Field(...)
    input_series_instance_uid : List[str] = Field(...)
    study_instance_uid        : str = Field(None,exclude=True)
    series_instance_uid       : str = Field(None,exclude=True)

    _field_order = ['inference_id',
                    'inference_timestamp',
                    'input_study_instance_uid',
                    'input_series_instance_uid',
                    'model_id',
                    'patient_id',
                    'detections',]

    @model_serializer(mode='wrap')
    def ordered_dump(self, handler):
        data = handler(self)
        return {k: data[k] for k in self._field_order}