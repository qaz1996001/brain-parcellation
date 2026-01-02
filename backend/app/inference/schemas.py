# app/inference/schemas.py
import datetime
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
from uuid import UUID, uuid4


# ============================================================================
# Model Selection Schemas
# ============================================================================

class ModelIdentifier(BaseModel):
    """Model selection by name+version or model_id (UUID).

    Either (model_name + model_version) or model_id must be provided.
    model_id takes precedence if both are present.
    """
    model_name: Optional[str] = Field(None, description="Model name (e.g., 'aneurysm_model')")
    model_version: Optional[str] = Field(None, description="Model version (e.g., 'v1.2.0')")
    model_id: Optional[UUID] = Field(None, description="Model UUID (overrides name+version)")

    @field_validator('model_id', mode='before')
    def validate_model_selection(cls, v, info):
        """Ensure either model_id or (model_name + model_version) is provided."""
        data = info.data
        if v is None and (data.get('model_name') is None or data.get('model_version') is None):
            raise ValueError("Either model_id or (model_name + model_version) required")
        return v


# ============================================================================
# Request Schemas
# ============================================================================

class SeriesInferenceRequest(BaseModel):
    """Single study inference request with selected series and model."""
    study_uid: str = Field(..., description="Study instance UID (Orthanc ID format)")
    series_uids: List[str] = Field(..., min_length=1, description="Series instance UIDs to process")
    model_id: Optional[UUID] = Field(None, description="Model UUID (overrides name+version)")
    model_name: Optional[str] = Field(None, description="Model name")
    model_version: Optional[str] = Field(None, description="Model version")

    @field_validator('series_uids')
    def validate_series_uids(cls, v):
        if not v or len(v) == 0:
            raise ValueError("series_uids cannot be empty")
        return v


class BatchInferenceRequest(BaseModel):
    """Batch of multiple study inference requests."""
    requests: List[SeriesInferenceRequest] = Field(..., min_length=1, description="Array of study requests")

    @field_validator('requests')
    def validate_requests(cls, v):
        if not v or len(v) == 0:
            raise ValueError("requests array cannot be empty")
        return v


# ============================================================================
# Response Schemas
# ============================================================================

class SeriesValidationResult(BaseModel):
    """Validation results for series readiness."""
    accepted: List[str] = Field(default_factory=list, description="Series UIDs ready for inference")
    rejected: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Series UIDs not ready with reasons"
    )


class QueueStatus(BaseModel):
    """Task queue position and estimated wait time."""
    position: int = Field(..., description="Queue position (total across both queues)")
    estimated_wait_seconds: int = Field(..., description="Estimated wait time in seconds")


class InferenceResponse(BaseModel):
    """Response for single study inference request."""
    study_uid: str
    inference_id: Optional[UUID] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_id: Optional[UUID] = None
    queue_status: Optional[QueueStatus] = None
    series_validation: SeriesValidationResult
    cache_status: Literal["new", "cached"] = "new"
    status: Literal["queued", "partial", "rejected"]


class BatchInferenceResponse(BaseModel):
    """Response for batch inference request."""
    batch_id: UUID = Field(default_factory=uuid4, description="Batch tracking ID")
    results: List[InferenceResponse] = Field(..., description="Results for each study request")


# ============================================================================
# Status Query Schemas
# ============================================================================

class InferenceStatusResponse(BaseModel):
    """Status response for inference query."""
    inference_id: UUID
    study_uid: str
    series_uids: List[str]
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    status: str = Field(..., description="Current DCOP status code")
    params_data: Optional[Dict[str, Any]] = None
    result_data: Optional[Dict[str, Any]] = None
    timestamp: datetime.datetime


class BatchStatusResponse(BaseModel):
    """Aggregated status for batch request."""
    batch_id: UUID
    total: int
    queued: int
    running: int
    completed: int
    failed: int
    statuses: List[InferenceStatusResponse]


# ============================================================================
# Callback Schemas
# ============================================================================

class InferenceCallbackRequest(BaseModel):
    """Callback payload from GPU worker matching radax requirements."""
    studyInstanceUid: str = Field(..., alias="studyInstanceUid")
    modelName: str = Field(..., alias="modelName")
    inferenceId: UUID = Field(..., alias="inferenceId")
    result: Literal["success", "failed"]
    resultData: Optional[Dict[str, Any]] = Field(None, alias="resultData")

    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Cache Management Schemas
# ============================================================================

class CacheEntry(BaseModel):
    """Cache entry information."""
    cache_key: str
    inference_id: UUID
    study_uid: str
    series_uids: List[str]
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_id: Optional[UUID] = None
    timestamp: datetime.datetime
    status: str


class CacheListResponse(BaseModel):
    """Response for cache listing."""
    entries: List[CacheEntry]
    total: int


class CacheDeleteResponse(BaseModel):
    """Response for cache deletion."""
    deleted_count: int
    cache_keys: List[str]
