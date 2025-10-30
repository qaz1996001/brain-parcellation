"""Unit tests for code_ai.pipeline.rdx.schema.vessel module.

Tests cover:
- VesselDilatedDetectionItem: Vessel dilation detection validation
- VesselDilatedDetectionResponse: Container for vessel detections

Following Linus-style patterns:
- Test basic functionality and inheritance
- Prepare for future field additions
- Verify base class integration
"""

from datetime import timezone

import pytest
from pydantic import ValidationError

from code_ai.pipeline.rdx.schema.vessel import (
    VesselDilatedDetectionItem,
    VesselDilatedDetectionResponse,
)


class TestVesselDilatedDetectionItem:
    """Test suite for VesselDilatedDetectionItem model."""

    def test_valid_instantiation(self) -> None:
        """Valid vessel data creates instance successfully."""
        detection = VesselDilatedDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="vessel_dilation",
        )

        assert detection.series_instance_uid == "1.2.840.113619.2.1.2.2.1"
        assert detection.sop_instance_uid == "1.2.840.113619.2.1.2.3.1"
        assert detection.label == "vessel_dilation"

    def test_inherits_base_fields(self) -> None:
        """VesselDilatedDetectionItem includes base class fields."""
        detection = VesselDilatedDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="vessel",
        )

        # Base fields from DetectionsBaseResponse
        assert hasattr(detection, "series_instance_uid")
        assert hasattr(detection, "sop_instance_uid")
        assert hasattr(detection, "label")

    def test_missing_required_fields_raises_validation_error(self) -> None:
        """Missing required base fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            VesselDilatedDetectionItem()

        errors = exc_info.value.errors()
        missing_fields = {err["loc"][0] for err in errors}
        # Only base class required fields
        assert "series_instance_uid" in missing_fields
        assert "sop_instance_uid" in missing_fields
        assert "label" in missing_fields

    def test_serialization_to_dict(self) -> None:
        """Model serializes to dictionary correctly."""
        detection = VesselDilatedDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="vessel",
        )

        result = detection.model_dump()

        assert isinstance(result, dict)
        assert result["series_instance_uid"] == "1.2.840.113619.2.1.2.2.1"
        assert result["label"] == "vessel"

    def test_json_serialization(self) -> None:
        """Model serializes to JSON correctly."""
        detection = VesselDilatedDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="vessel_dilation",
        )

        json_str = detection.model_dump_json()

        assert isinstance(json_str, str)
        assert "vessel_dilation" in json_str


class TestVesselDilatedDetectionResponse:
    """Test suite for VesselDilatedDetectionResponse container."""

    def test_valid_instantiation_with_detections(self) -> None:
        """Valid response with vessel detections creates successfully."""
        detection = VesselDilatedDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="vessel_dilation",
        )

        response = VesselDilatedDetectionResponse(
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="vessel_v1.0",
            detections=[detection],
        )

        assert len(response.detections) == 1
        assert response.detections[0].label == "vessel_dilation"

    def test_multiple_detections(self) -> None:
        """Response supports multiple vessel detections."""
        detection1 = VesselDilatedDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="vessel_dilation",
        )
        detection2 = VesselDilatedDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.2",
            label="vessel_narrowing",
        )

        response = VesselDilatedDetectionResponse(
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="vessel_v1.0",
            detections=[detection1, detection2],
        )

        assert len(response.detections) == 2
        assert response.detections[0].label == "vessel_dilation"
        assert response.detections[1].label == "vessel_narrowing"

    def test_empty_detections_list(self) -> None:
        """Response accepts empty detections list."""
        response = VesselDilatedDetectionResponse(
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="vessel_v1.0",
            detections=[],
        )

        assert response.detections == []

    def test_inherits_timestamp_validation(self, naive_timestamp) -> None:
        """Response inherits UTC timestamp validation from base."""
        response = VesselDilatedDetectionResponse(
            inference_timestamp=naive_timestamp,
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="vessel_v1.0",
            detections=[],
        )

        # Should convert to UTC
        assert response.inference_timestamp.tzinfo == timezone.utc

    def test_missing_required_fields_raises_validation_error(self) -> None:
        """Missing required fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            VesselDilatedDetectionResponse()

        errors = exc_info.value.errors()
        missing_fields = {err["loc"][0] for err in errors}
        assert "patient_id" in missing_fields
        assert "study_instance_uid" in missing_fields
        assert "model_id" in missing_fields
        assert "detections" in missing_fields

    def test_serialization_to_json(self) -> None:
        """Response serializes to JSON correctly."""
        detection = VesselDilatedDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="vessel_dilation",
        )

        response = VesselDilatedDetectionResponse(
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="vessel_v1.0",
            detections=[detection],
        )

        json_str = response.model_dump_json()

        assert isinstance(json_str, str)
        assert "TEST_PATIENT_001" in json_str
        assert "vessel_v1.0" in json_str
        assert "vessel_dilation" in json_str