"""Unit tests for code_ai.pipeline.rdx.schema.aneurysm module.

Tests cover:
- AneurysmDetectionItem: Aneurysm-specific detection validation
- AneurysmDetectionResponse: Container for aneurysm detections

Following Linus-style patterns:
- Test field validation constraints (PositiveFloat, PositiveInt, confloat)
- Test serialization behavior (diameter rounding)
- Test inheritance from base models
"""

from datetime import timezone

import pytest
from pydantic import ValidationError

from code_ai.pipeline.rdx.schema.aneurysm import (
    AneurysmDetectionItem,
    AneurysmDetectionResponse,
)


class TestAneurysmDetectionItem:
    """Test suite for AneurysmDetectionItem model."""

    def test_valid_instantiation(self, aneurysm_prediction_result) -> None:
        """Valid aneurysm data creates instance successfully."""
        detection = AneurysmDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
            **aneurysm_prediction_result,
        )

        assert detection.type == "saccular"
        assert detection.location == "MCA"
        assert detection.diameter == 5.234
        assert detection.main_seg_slice == 42
        assert detection.probability == 0.95
        assert detection.pitch_angle == 45
        assert detection.yaw_angle == 30
        assert detection.mask_index == 1
        assert detection.sub_location == "M1_segment"

    def test_diameter_must_be_positive(self) -> None:
        """Diameter field rejects non-positive values."""
        with pytest.raises(ValidationError) as exc_info:
            AneurysmDetectionItem(
                series_instance_uid="1.2.840.113619.2.1.2.2.1",
                sop_instance_uid="1.2.840.113619.2.1.2.3.1",
                label="aneurysm",
                type="saccular",
                location="MCA",
                diameter=-1.5,  # Invalid: negative
                main_seg_slice=42,
                probability=0.95,
                pitch_angle=45,
                yaw_angle=30,
                mask_index=1,
                sub_location="M1_segment",
            )

        errors = exc_info.value.errors()
        assert any(err["loc"][0] == "diameter" for err in errors)

    def test_diameter_zero_rejected(self) -> None:
        """Diameter field rejects zero (must be positive)."""
        with pytest.raises(ValidationError) as exc_info:
            AneurysmDetectionItem(
                series_instance_uid="1.2.840.113619.2.1.2.2.1",
                sop_instance_uid="1.2.840.113619.2.1.2.3.1",
                label="aneurysm",
                type="saccular",
                location="MCA",
                diameter=0.0,  # Invalid: zero
                main_seg_slice=42,
                probability=0.95,
                pitch_angle=45,
                yaw_angle=30,
                mask_index=1,
                sub_location="M1_segment",
            )

        errors = exc_info.value.errors()
        assert any(err["loc"][0] == "diameter" for err in errors)

    def test_main_seg_slice_must_be_positive(self) -> None:
        """main_seg_slice field rejects non-positive integers."""
        with pytest.raises(ValidationError) as exc_info:
            AneurysmDetectionItem(
                series_instance_uid="1.2.840.113619.2.1.2.2.1",
                sop_instance_uid="1.2.840.113619.2.1.2.3.1",
                label="aneurysm",
                type="saccular",
                location="MCA",
                diameter=5.2,
                main_seg_slice=0,  # Invalid: zero
                probability=0.95,
                pitch_angle=45,
                yaw_angle=30,
                mask_index=1,
                sub_location="M1_segment",
            )

        errors = exc_info.value.errors()
        assert any(err["loc"][0] == "main_seg_slice" for err in errors)

    def test_probability_within_bounds(self) -> None:
        """Probability field accepts values between 0.0 and 1.0."""
        detection = AneurysmDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
            type="saccular",
            location="MCA",
            diameter=5.2,
            main_seg_slice=42,
            probability=0.0,  # Minimum bound
            pitch_angle=45,
            yaw_angle=30,
            mask_index=1,
            sub_location="M1_segment",
        )
        assert detection.probability == 0.0

        detection2 = AneurysmDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
            type="saccular",
            location="MCA",
            diameter=5.2,
            main_seg_slice=42,
            probability=1.0,  # Maximum bound
            pitch_angle=45,
            yaw_angle=30,
            mask_index=1,
            sub_location="M1_segment",
        )
        assert detection2.probability == 1.0

    def test_probability_above_one_rejected(self) -> None:
        """Probability field rejects values above 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            AneurysmDetectionItem(
                series_instance_uid="1.2.840.113619.2.1.2.2.1",
                sop_instance_uid="1.2.840.113619.2.1.2.3.1",
                label="aneurysm",
                type="saccular",
                location="MCA",
                diameter=5.2,
                main_seg_slice=42,
                probability=1.5,  # Invalid: > 1.0
                pitch_angle=45,
                yaw_angle=30,
                mask_index=1,
                sub_location="M1_segment",
            )

        errors = exc_info.value.errors()
        assert any(err["loc"][0] == "probability" for err in errors)

    def test_probability_below_zero_rejected(self) -> None:
        """Probability field rejects negative values."""
        with pytest.raises(ValidationError) as exc_info:
            AneurysmDetectionItem(
                series_instance_uid="1.2.840.113619.2.1.2.2.1",
                sop_instance_uid="1.2.840.113619.2.1.2.3.1",
                label="aneurysm",
                type="saccular",
                location="MCA",
                diameter=5.2,
                main_seg_slice=42,
                probability=-0.1,  # Invalid: < 0.0
                pitch_angle=45,
                yaw_angle=30,
                mask_index=1,
                sub_location="M1_segment",
            )

        errors = exc_info.value.errors()
        assert any(err["loc"][0] == "probability" for err in errors)

    def test_diameter_serialization_rounds_to_four_decimals(self) -> None:
        """Diameter serializes with 4 decimal places."""
        detection = AneurysmDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
            type="saccular",
            location="MCA",
            diameter=5.23456789,  # More than 4 decimals
            main_seg_slice=42,
            probability=0.95,
            pitch_angle=45,
            yaw_angle=30,
            mask_index=1,
            sub_location="M1_segment",
        )

        serialized = detection.model_dump(mode="json")

        assert serialized["diameter"] == 5.2346  # Rounded to 4 decimals

    def test_missing_required_fields_raises_validation_error(self) -> None:
        """Missing required fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AneurysmDetectionItem(
                series_instance_uid="1.2.840.113619.2.1.2.2.1",
                sop_instance_uid="1.2.840.113619.2.1.2.3.1",
                label="aneurysm",
            )

        errors = exc_info.value.errors()
        missing_fields = {err["loc"][0] for err in errors}
        required_fields = {
            "type",
            "location",
            "diameter",
            "main_seg_slice",
            "probability",
            "pitch_angle",
            "yaw_angle",
            "mask_index",
            "sub_location",
        }
        assert required_fields.issubset(missing_fields)

    def test_inherits_base_fields(self) -> None:
        """AneurysmDetectionItem includes base class fields."""
        detection = AneurysmDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
            type="saccular",
            location="MCA",
            diameter=5.2,
            main_seg_slice=42,
            probability=0.95,
            pitch_angle=45,
            yaw_angle=30,
            mask_index=1,
            sub_location="M1_segment",
        )

        # Base fields from DetectionsBaseResponse
        assert hasattr(detection, "series_instance_uid")
        assert hasattr(detection, "sop_instance_uid")
        assert hasattr(detection, "label")


class TestAneurysmDetectionResponse:
    """Test suite for AneurysmDetectionResponse container."""

    def test_valid_instantiation_with_detections(
        self, aneurysm_prediction_result
    ) -> None:
        """Valid response with aneurysm detections creates successfully."""
        detection = AneurysmDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
            **aneurysm_prediction_result,
        )

        response = AneurysmDetectionResponse(
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="aneurysm_v1.0",
            detections=[detection],
        )

        assert len(response.detections) == 1
        assert response.detections[0].type == "saccular"
        assert response.detections[0].diameter == 5.234

    def test_multiple_detections(self, aneurysm_prediction_result) -> None:
        """Response supports multiple aneurysm detections."""
        detection1 = AneurysmDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
            **aneurysm_prediction_result,
        )
        detection2 = AneurysmDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.2",
            label="aneurysm",
            type="fusiform",
            location="ACA",
            diameter=3.1,
            main_seg_slice=50,
            probability=0.88,
            pitch_angle=20,
            yaw_angle=15,
            mask_index=2,
            sub_location="A1_segment",
        )

        response = AneurysmDetectionResponse(
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="aneurysm_v1.0",
            detections=[detection1, detection2],
        )

        assert len(response.detections) == 2
        assert response.detections[0].location == "MCA"
        assert response.detections[1].location == "ACA"

    def test_empty_detections_list(self) -> None:
        """Response accepts empty detections list."""
        response = AneurysmDetectionResponse(
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="aneurysm_v1.0",
            detections=[],
        )

        assert response.detections == []

    def test_inherits_timestamp_validation(self, naive_timestamp) -> None:
        """Response inherits UTC timestamp validation from base."""
        response = AneurysmDetectionResponse(
            inference_timestamp=naive_timestamp,
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="aneurysm_v1.0",
            detections=[],
        )

        # Should convert to UTC
        assert response.inference_timestamp.tzinfo == timezone.utc

    def test_serialization_to_json(self, aneurysm_prediction_result) -> None:
        """Response serializes to JSON correctly."""
        detection = AneurysmDetectionItem(
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            sop_instance_uid="1.2.840.113619.2.1.2.3.1",
            label="aneurysm",
            **aneurysm_prediction_result,
        )

        response = AneurysmDetectionResponse(
            patient_id="TEST_PATIENT_001",
            study_instance_uid="1.2.840.113619.2.1.2.1.1",
            series_instance_uid="1.2.840.113619.2.1.2.2.1",
            model_id="aneurysm_v1.0",
            detections=[detection],
        )

        json_str = response.model_dump_json()

        assert isinstance(json_str, str)
        assert "TEST_PATIENT_001" in json_str
        assert "aneurysm_v1.0" in json_str
        assert "saccular" in json_str