"""Unit tests for code_ai.pipeline.rdx.build.aneurysm module.

Tests cover:
- AneurysmDetectionBuilder: Builder pattern for aneurysm detections
- DICOM data extraction and validation
- Fluent interface and state management
- Error handling for missing data

Following Linus-style patterns:
- Test builder state transitions
- Verify fluent interface returns Self
- Test error paths with invalid DICOM data
"""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pydicom.dataset import Dataset
from pydicom.tag import Tag

from code_ai.pipeline.rdx.build.aneurysm import AneurysmDetectionBuilder
from code_ai.pipeline.rdx.schema.aneurysm import AneurysmDetectionResponse


class TestAneurysmDetectionBuilder:
    """Test suite for AneurysmDetectionBuilder."""

    @pytest.fixture
    def builder(self) -> AneurysmDetectionBuilder:
        """Create fresh builder instance for each test."""
        return AneurysmDetectionBuilder()

    @pytest.fixture
    def mock_dicom_list(self, mock_dicom_dataset) -> list:
        """Create list containing mock DICOM dataset."""
        return [mock_dicom_dataset]

    def test_builder_initialization(self, builder) -> None:
        """Builder initializes with empty state."""
        assert builder._detections == []
        assert builder._prediction_dict == {}
        assert builder.model_class == AneurysmDetectionResponse

    def test_reset_clears_state(self, builder) -> None:
        """reset() clears builder internal state."""
        # Add some state
        builder._prediction_dict = {"patient_id": "TEST"}
        builder._detections = [MagicMock()]

        # Reset
        builder.reset()

        assert builder._detections == []
        assert builder._prediction_dict == {}

    def test_set_patient_info_extracts_dicom_uids(
        self, builder, mock_dicom_list
    ) -> None:
        """set_patient_info() extracts patient and study UIDs from DICOM."""
        result = builder.set_patient_info(mock_dicom_list)

        assert result is builder  # Fluent interface
        assert builder._prediction_dict["patient_id"] == "TEST_PATIENT_001"
        assert (
            builder._prediction_dict["study_instance_uid"]
            == "1.2.840.113619.2.1.2.1.1"
        )
        assert (
            builder._prediction_dict["series_instance_uid"]
            == "1.2.840.113619.2.1.2.2.1"
        )

    def test_set_patient_info_empty_list_raises_error(self, builder) -> None:
        """set_patient_info() raises ValueError for empty list."""
        with pytest.raises(ValueError, match="source_images cannot be empty"):
            builder.set_patient_info([])

    def test_set_model_id_stores_model_id(self, builder) -> None:
        """set_model_id() stores model identifier."""
        result = builder.set_model_id("aneurysm_v1.0")

        assert result is builder  # Fluent interface
        assert builder._prediction_dict["model_id"] == "aneurysm_v1.0"

    def test_build_detection_creates_detection_item(
        self, builder, mock_dicom_dataset, aneurysm_prediction_result
    ) -> None:
        """build_detection() creates AneurysmDetectionItem from DICOM and prediction."""

        detection = builder.build_detection(
            mock_dicom_dataset, aneurysm_prediction_result
        )

        assert detection.series_instance_uid == "1.2.840.113619.2.1.2.2.1"
        assert detection.sop_instance_uid == "1.2.840.113619.2.1.2.3.1"
        assert detection.type == "saccular"
        assert detection.location == "MCA"
        assert detection.diameter == 5.234
        assert detection.main_seg_slice == 42
        assert detection.probability == 0.95

    def test_build_detection_missing_prediction_field_raises_error(
        self, builder, mock_dicom_dataset
    ) -> None:
        """build_detection() raises KeyError for missing prediction fields."""
        incomplete_result = {
            "type": "saccular",
            # Missing required fields: location, diameter, etc.
        }

        with pytest.raises(KeyError):
            builder.build_detection(mock_dicom_dataset, incomplete_result)

    def test_build_detection_invalid_dicom_tag_raises_error(
        self, builder, aneurysm_prediction_result
    ) -> None:
        """build_detection() raises AttributeError for invalid DICOM tags."""
        # Create DICOM dataset missing required tags
        invalid_dicom = Dataset()
        invalid_dicom.PatientID = "TEST"
        # Missing SeriesInstanceUID and SOPInstanceUID

        with pytest.raises(AttributeError):
            builder.build_detection(invalid_dicom, aneurysm_prediction_result)

    def test_set_detections_batch_processing(
        self,
        builder,
        mock_dicom_dataset,
        aneurysm_prediction_result,
    ) -> None:
        """set_detections() processes multiple DICOM and prediction pairs."""
        # Create second DICOM and prediction
        dicom2 = Dataset()
        dicom2.PatientID = "TEST_PATIENT_002"
        dicom2.StudyInstanceUID = "1.2.840.113619.2.1.2.1.2"
        dicom2.SeriesInstanceUID = "1.2.840.113619.2.1.2.2.2"
        dicom2.SOPInstanceUID = "1.2.840.113619.2.1.2.3.2"

        prediction2 = {
            "type": "fusiform",
            "location": "ACA",
            "diameter": 3.5,
            "main_seg_slice": 50,
            "probability": 0.88,
            "pitch_angle": 20,
            "yaw_angle": 15,
            "mask_index": 2,
            "sub_location": "A1_segment",
        }

        result = builder.set_detections(
            [mock_dicom_dataset, dicom2],
            [aneurysm_prediction_result, prediction2],
        )

        assert result is builder  # Fluent interface
        assert len(builder._detections) == 2
        assert builder._detections[0].location == "MCA"
        assert builder._detections[1].location == "ACA"

    def test_fluent_interface_chaining(
        self,
        builder,
        mock_dicom_list,
        mock_dicom_dataset,
        aneurysm_prediction_result,
    ) -> None:
        """Builder methods support fluent interface chaining."""
        result = (
            builder.set_patient_info(mock_dicom_list)
            .set_model_id("aneurysm_v1.0")
            .set_detections([mock_dicom_dataset], [aneurysm_prediction_result])
        )

        assert result is builder
        assert builder._prediction_dict["patient_id"] == "TEST_PATIENT_001"
        assert builder._prediction_dict["model_id"] == "aneurysm_v1.0"
        assert len(builder._detections) == 1

    def test_build_creates_response_and_resets(
        self,
        builder,
        mock_dicom_list,
        mock_dicom_dataset,
        aneurysm_prediction_result,
    ) -> None:
        """build() creates AneurysmDetectionResponse and resets builder state."""
        builder.set_patient_info(mock_dicom_list).set_model_id(
            "aneurysm_v1.0"
        ).set_detections([mock_dicom_dataset], [aneurysm_prediction_result])

        response = builder.build()

        # Verify response
        assert isinstance(response, AneurysmDetectionResponse)
        assert response.patient_id == "TEST_PATIENT_001"
        assert response.model_id == "aneurysm_v1.0"
        assert len(response.detections) == 1
        assert response.detections[0].location == "MCA"

        # Verify reset
        assert builder._detections == []
        assert builder._prediction_dict == {}

    def test_build_without_required_fields_raises_validation_error(
        self, builder
    ) -> None:
        """build() raises ValidationError when required fields missing."""
        # Only set model_id, missing patient_info and detections
        builder.set_model_id("aneurysm_v1.0")

        with pytest.raises(ValidationError):
            builder.build()

    def test_build_with_empty_detections(
        self, builder, mock_dicom_list
    ) -> None:
        """build() accepts empty detections list."""
        builder.set_patient_info(mock_dicom_list).set_model_id("aneurysm_v1.0")

        # Manually set empty detections
        builder._detections = []

        response = builder.build()

        assert isinstance(response, AneurysmDetectionResponse)
        assert len(response.detections) == 0

    def test_multiple_build_cycles(
        self,
        builder,
        mock_dicom_list,
        mock_dicom_dataset,
        aneurysm_prediction_result,
    ) -> None:
        """Builder supports multiple build cycles with reset between."""
        # First build
        builder.set_patient_info(mock_dicom_list).set_model_id(
            "aneurysm_v1.0"
        ).set_detections([mock_dicom_dataset], [aneurysm_prediction_result])
        response1 = builder.build()

        # Second build with different data
        builder.set_patient_info(mock_dicom_list).set_model_id(
            "aneurysm_v2.0"
        ).set_detections([mock_dicom_dataset], [aneurysm_prediction_result])
        response2 = builder.build()

        assert response1.model_id == "aneurysm_v1.0"
        assert response2.model_id == "aneurysm_v2.0"
        # Verify both are independent instances
        assert response1 is not response2

    def test_build_detection_default_label(
        self, builder, mock_dicom_dataset
    ) -> None:
        """build_detection() uses default label 'aneurysm' when not provided."""
        prediction_without_label = {
            "type": "saccular",
            "location": "MCA",
            "diameter": 5.2,
            "main_seg_slice": 42,
            "probability": 0.95,
            "pitch_angle": 45,
            "yaw_angle": 30,
            "mask_index": 1,
            "sub_location": "M1_segment",
        }

        detection = builder.build_detection(
            mock_dicom_dataset, prediction_without_label
        )

        assert detection.label == "aneurysm"

    def test_build_detection_custom_label(
        self, builder, mock_dicom_dataset, aneurysm_prediction_result
    ) -> None:
        """build_detection() uses custom label when provided."""
        aneurysm_prediction_result["label"] = "custom_aneurysm"

        detection = builder.build_detection(
            mock_dicom_dataset, aneurysm_prediction_result
        )

        assert detection.label == "custom_aneurysm"