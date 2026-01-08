"""Base models for prediction and detection responses."""

import datetime
from typing import List, Optional, Generic, TypeVar

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    field_serializer,
)

__all__ = [
    "DetectionsBaseResponse",
    "PredictionBaseResponse",
    "InferenceCompleteRequest",
    "DetectionT",
    "PredictionT",
]


# =============================================================================
# Detection Models
# =============================================================================

class DetectionsBaseResponse(BaseModel):
    """Base response model for detection results."""

    series_instance_uid: str = Field(...)
    sop_instance_uid: str = Field(...)
    label: str = Field(...)


# TypeVar 定義在類別之後，可以直接引用
DetectionT = TypeVar("DetectionT", bound=DetectionsBaseResponse)


# =============================================================================
# Prediction Models
# =============================================================================

class PredictionBaseResponse(BaseModel, Generic[DetectionT]):
    """Base response model for prediction results with generic detection type."""

    model_config = ConfigDict(from_attributes=True)

    inference_timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    patient_id: str = Field(...)
    study_instance_uid: str = Field(...)
    series_instance_uid: str = Field(...)
    model_id: str = Field(...)
    detections: List[Optional[DetectionT]] = Field(...)

    @field_validator("inference_timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime.datetime) -> datetime.datetime:
        """確保 timestamp 為 UTC"""
        if v.tzinfo is None:
            return v.replace(tzinfo=datetime.timezone.utc)
        elif v.tzinfo != datetime.timezone.utc:
            return v.astimezone(datetime.timezone.utc)
        return v

    @field_serializer("inference_timestamp")
    def serialize_timestamp(self, value: datetime.datetime) -> str:
        """序列化為 ISO 8601 格式的 UTC 字串"""
        utc_time = value.astimezone(datetime.timezone.utc)
        return utc_time.isoformat()


# TypeVar 定義在類別之後
PredictionT = TypeVar("PredictionT", bound=PredictionBaseResponse)