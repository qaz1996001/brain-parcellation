"""Unit tests for code_ai.pipeline.rdx.schema.base module.

Tests cover:
- DetectionsBaseResponse: Basic detection item validation
- PredictionBaseResponse: Prediction container with UTC timestamp handling
- InferenceCompleteRequest: Callback request validation

Following Linus-style patterns:
- Explicit test names describing behavior
- Separate tests for success and failure paths
- Validation of edge cases and boundary conditions
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from code_ai.pipeline.rdx.schema.base import (
    DetectionsBaseResponse,
    InferenceCompleteRequest,
    PredictionBaseResponse,
)


class TestDetectionsBaseResponse:
    """Test suite for DetectionsBaseResponse base model."""

    def test_valid_instantiation(self) -> None:
        """Valid data creates instance successfully."""
        detection = DetectionsBaseResponse(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
        )

        assert detection.series_instance_uid == "1.2.840.113619.2.1.2.2.1"
        assert detection.sop_instance_uid == "1.2.840.113619.2.1.2.3.1"
        assert detection.label == "aneurysm"

    def test_missing_required_fields_raises_validation_error(self) -> None:
        """Missing required fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionsBaseResponse()

        errors = exc_info.value.errors()
        missing_fields = {err["loc"][0] for err in errors}
        assert "series_instance_uid" in missing_fields
        assert "sop_instance_uid" in missing_fields
        assert "label" in missing_fields

    def test_partial_missing_fields_raises_validation_error(self) -> None:
        """Partial data with missing fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionsBaseResponse(
                series_instance_uid="1.2.840.113619.2.1.2.2.1"
            )

        errors = exc_info.value.errors()
        missing_fields = {err["loc"][0] for err in errors}
        assert "sop_instance_uid" in missing_fields
        assert "label" in missing_fields

    def test_serialization_to_dict(self) -> None:
        """Model serializes to dictionary correctly."""
        detection = DetectionsBaseResponse(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="vessel_dilation",
        )

        result = detection.model_dump()

        assert isinstance(result, dict)
        assert result["series_instance_uid"] == "1.2.840.113619.2.1.2.2.1"
        assert result["sop_instance_uid"] == "1.2.840.113619.2.1.2.3.1"
        assert result["label"] == "vessel_dilation"

    def test_json_serialization(self) -> None:
        """Model serializes to JSON correctly."""
        detection = DetectionsBaseResponse(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="test",
        )

        json_str = detection.model_dump_json()

        assert isinstance(json_str, str)
        assert "1.2.840.113619.2.1.2.2.1" in json_str
        assert "test" in json_str


class TestPredictionBaseResponse:
    """Test suite for PredictionBaseResponse generic container."""

    def test_valid_instantiation_with_utc_timestamp(
        self, mock_dicom_dataset, utc_timestamp
    ) -> None:
        """Valid data with UTC timestamp creates instance successfully."""
        prediction = PredictionBaseResponse[DetectionsBaseResponse](
            inference_timestamp=utc_timestamp,
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="aneurysm_v1.0",
            detections=[],
        )

        assert prediction.patient_id == "TEST_PATIENT_001"
        assert prediction.model_id == "aneurysm_v1.0"
        assert prediction.inference_timestamp.tzinfo == timezone.utc

    def test_default_timestamp_is_utc(self) -> None:
        """Default inference_timestamp is UTC-aware."""
        prediction = PredictionBaseResponse[DetectionsBaseResponse](
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="test_model",
            detections=[],
        )

        assert prediction.inference_timestamp.tzinfo == timezone.utc

    def test_naive_timestamp_converted_to_utc(self, naive_timestamp) -> None:
        """Naive timestamp (no timezone) is assumed UTC."""
        prediction = PredictionBaseResponse[DetectionsBaseResponse](
            inference_timestamp=naive_timestamp,
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="test_model",
            detections=[],
        )

        assert prediction.inference_timestamp.tzinfo == timezone.utc
        # Verify time values match (only timezone added)
        assert prediction.inference_timestamp.year == naive_timestamp.year
        assert prediction.inference_timestamp.month == naive_timestamp.month
        assert prediction.inference_timestamp.day == naive_timestamp.day
        assert prediction.inference_timestamp.hour == naive_timestamp.hour

    def test_non_utc_timestamp_converted_to_utc(self, non_utc_timestamp) -> None:
        """Non-UTC timezone-aware timestamp is converted to UTC."""
        prediction = PredictionBaseResponse[DetectionsBaseResponse](
            inference_timestamp=non_utc_timestamp,
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="test_model",
            detections=[],
        )

        assert prediction.inference_timestamp.tzinfo == timezone.utc
        # Verify conversion preserves absolute time
        expected_utc = non_utc_timestamp.astimezone(timezone.utc)
        assert prediction.inference_timestamp == expected_utc

    def test_timestamp_serialization_format(self, utc_timestamp) -> None:
        """Timestamp serializes to ISO 8601 format."""
        prediction = PredictionBaseResponse[DetectionsBaseResponse](
            inference_timestamp=utc_timestamp,
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="test_model",
            detections=[],
        )

        json_data = prediction.model_dump(mode="json")

        # ISO 8601 format includes 'T' separator and timezone info
        assert "T" in json_data["inference_timestamp"]
        assert json_data["inference_timestamp"].endswith(
            "+00:00"
        ) or json_data["inference_timestamp"].endswith("Z")

    def test_missing_required_fields_raises_validation_error(self) -> None:
        """Missing required fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionBaseResponse[DetectionsBaseResponse]()

        errors = exc_info.value.errors()
        missing_fields = {err["loc"][0] for err in errors}
        assert "patient_id" in missing_fields
        assert "study_instance_uid" in missing_fields
        assert "series_instance_uid" in missing_fields
        assert "model_id" in missing_fields
        assert "detections" in missing_fields

    def test_detections_list_with_items(self) -> None:
        """Detections list with items validates correctly."""
        detection1 = DetectionsBaseResponse(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
        )
        detection2 = DetectionsBaseResponse(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.2",
            label="vessel",
        )

        prediction = PredictionBaseResponse[DetectionsBaseResponse](
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="test_model",
            detections=[detection1, detection2],
        )

        assert len(prediction.detections) == 2
        assert prediction.detections[0].label == "aneurysm"
        assert prediction.detections[1].label == "vessel"

    def test_empty_detections_list(self) -> None:
        """Empty detections list is valid."""
        prediction = PredictionBaseResponse[DetectionsBaseResponse](
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="test_model",
            detections=[],
        )

        assert prediction.detections == []
        assert len(prediction.detections) == 0


class TestInferenceCompleteRequest:
    """Test suite for InferenceCompleteRequest callback model."""

    def test_valid_instantiation(self) -> None:
        """Valid data creates instance successfully."""
        request = InferenceCompleteRequest(
            studyInstanceUid="1.2.840.113619.2.1.2.1.1",
            seriesInstanceUid="1.2.840.113619.2.1.2.2.1",
        )

        assert request.studyInstanceUid == "1.2.840.113619.2.1.2.1.1"
        assert request.seriesInstanceUid == "1.2.840.113619.2.1.2.2.1"

    def test_missing_required_fields_raises_validation_error(self) -> None:
        """Missing required fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InferenceCompleteRequest()

        errors = exc_info.value.errors()
        missing_fields = {err["loc"][0] for err in errors}
        assert "studyInstanceUid" in missing_fields
        assert "seriesInstanceUid" in missing_fields

    def test_partial_missing_fields_raises_validation_error(self) -> None:
        """Partial data with missing fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InferenceCompleteRequest(
                studyInstanceUid="1.2.840.113619.2.1.2.1.1"
            )

        errors = exc_info.value.errors()
        missing_fields = {err["loc"][0] for err in errors}
        assert "seriesInstanceUid" in missing_fields

    def test_serialization_to_dict(self) -> None:
        """Model serializes to dictionary correctly."""
        request = InferenceCompleteRequest(
            studyInstanceUid="1.2.840.113619.2.1.2.1.1",
            seriesInstanceUid="1.2.840.113619.2.1.2.2.1",
        )

        result = request.model_dump()

        assert isinstance(result, dict)
        assert result["studyInstanceUid"] == "1.2.840.113619.2.1.2.1.1"
        assert result["seriesInstanceUid"] == "1.2.840.113619.2.1.2.2.1"

    def test_json_serialization(self) -> None:
        """Model serializes to JSON correctly."""
        request = InferenceCompleteRequest(
            studyInstanceUid="1.2.840.113619.2.1.2.1.1",
            seriesInstanceUid="1.2.840.113619.2.1.2.2.1",
        )

        json_str = request.model_dump_json()

        assert isinstance(json_str, str)
        assert "1.2.840.113619.2.1.2.1.1" in json_str
        assert "1.2.840.113619.2.1.2.2.1" in json_str