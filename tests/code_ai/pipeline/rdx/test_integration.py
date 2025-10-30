"""Integration tests for code_ai.pipeline.rdx module.

Tests cover end-to-end workflows:
- Complete builder workflow from DICOM to AneurysmDetectionResponse
- Multiple detection handling
- Timestamp consistency across predictions
- Real-world usage scenarios

Following Linus-style patterns:
- Test complete workflows, not individual components
- Verify cross-component integration
- Test realistic usage scenarios
"""

from datetime import datetime, timezone

import pytest
from pydicom.dataset import Dataset

from code_ai.pipeline.rdx.build.aneurysm import AneurysmDetectionBuilder
from code_ai.pipeline.rdx.schema.aneurysm import (
    AneurysmDetectionItem,
    AneurysmDetectionResponse,
)


class TestAneurysmBuilderIntegration:
    """Integration tests for complete aneurysm detection workflow."""

    def test_complete_single_detection_workflow(
        self, mock_dicom_dataset, aneurysm_prediction_result
    ) -> None:
        """Complete workflow: DICOM + prediction → AneurysmDetectionResponse."""
        builder = AneurysmDetectionBuilder()

        response = (
            builder.set_patient_info([mock_dicom_dataset])
            .set_model_id("aneurysm_v1.0")
            .set_detections([mock_dicom_dataset], [aneurysm_prediction_result])
            .build()
        )

        # Verify response structure
        assert isinstance(response, AneurysmDetectionResponse)
        assert response.patient_id == "TEST_PATIENT_001"
        assert response.study_instance_uid == "1.2.840.113619.2.1.2.1.1"
        assert response.model_id == "aneurysm_v1.0"
        assert response.inference_timestamp.tzinfo == timezone.utc

        # Verify detection
        assert len(response.detections) == 1
        detection = response.detections[0]
        assert isinstance(detection, AneurysmDetectionItem)
        assert detection.type == "saccular"
        assert detection.location == "MCA"
        assert detection.diameter == 5.234

    def test_complete_multiple_detections_workflow(self) -> None:
        """Complete workflow with multiple detections from different slices."""
        # Create multiple DICOM datasets
        dicom1 = Dataset()
        dicom1.PatientID = "MULTI_TEST_001"
        dicom1.StudyInstanceUID = "1.2.840.113619.2.1.2.1.10"
        dicom1.SeriesInstanceUID = "1.2.840.113619.2.1.2.2.10"
        dicom1.SOPInstanceUID = "1.2.840.113619.2.1.2.3.10"

        dicom2 = Dataset()
        dicom2.PatientID = "MULTI_TEST_001"
        dicom2.StudyInstanceUID = "1.2.840.113619.2.1.2.1.10"
        dicom2.SeriesInstanceUID = "1.2.840.113619.2.1.2.2.10"
        dicom2.SOPInstanceUID = "1.2.840.113619.2.1.2.3.11"

        dicom3 = Dataset()
        dicom3.PatientID = "MULTI_TEST_001"
        dicom3.StudyInstanceUID = "1.2.840.113619.2.1.2.1.10"
        dicom3.SeriesInstanceUID = "1.2.840.113619.2.1.2.2.10"
        dicom3.SOPInstanceUID = "1.2.840.113619.2.1.2.3.12"

        # Create predictions for different aneurysms
        predictions = [
            {
                "type": "saccular",
                "location": "MCA",
                "diameter": 5.2,
                "main_seg_slice": 42,
                "probability": 0.95,
                "pitch_angle": 45,
                "yaw_angle": 30,
                "mask_index": 1,
                "sub_location": "M1_segment",
            },
            {
                "type": "fusiform",
                "location": "ACA",
                "diameter": 3.5,
                "main_seg_slice": 50,
                "probability": 0.88,
                "pitch_angle": 20,
                "yaw_angle": 15,
                "mask_index": 2,
                "sub_location": "A1_segment",
            },
            {
                "type": "saccular",
                "location": "PCA",
                "diameter": 4.1,
                "main_seg_slice": 60,
                "probability": 0.92,
                "pitch_angle": 35,
                "yaw_angle": 25,
                "mask_index": 3,
                "sub_location": "P1_segment",
            },
        ]

        builder = AneurysmDetectionBuilder()

        response = (
            builder.set_patient_info([dicom1])
            .set_model_id("aneurysm_multi_v1.0")
            .set_detections([dicom1, dicom2, dicom3], predictions)
            .build()
        )

        # Verify response
        assert len(response.detections) == 3
        assert response.patient_id == "MULTI_TEST_001"

        # Verify each detection
        assert response.detections[0].location == "MCA"
        assert response.detections[0].type == "saccular"
        assert response.detections[1].location == "ACA"
        assert response.detections[1].type == "fusiform"
        assert response.detections[2].location == "PCA"
        assert response.detections[2].diameter == 4.1

    def test_multiple_build_cycles_with_different_patients(
        self, aneurysm_prediction_result
    ) -> None:
        """Builder reuse across multiple patients maintains data separation."""
        builder = AneurysmDetectionBuilder()

        # Patient 1
        dicom_p1 = Dataset()
        dicom_p1.PatientID = "PATIENT_001"
        dicom_p1.StudyInstanceUID = "1.2.840.113619.2.1.2.1.1"
        dicom_p1.SeriesInstanceUID = "1.2.840.113619.2.1.2.2.1"
        dicom_p1.SOPInstanceUID = "1.2.840.113619.2.1.2.3.1"

        response1 = (
            builder.set_patient_info([dicom_p1])
            .set_model_id("aneurysm_v1.0")
            .set_detections([dicom_p1], [aneurysm_prediction_result])
            .build()
        )

        # Patient 2
        dicom_p2 = Dataset()
        dicom_p2.PatientID = "PATIENT_002"
        dicom_p2.StudyInstanceUID = "1.2.840.113619.2.1.2.1.2"
        dicom_p2.SeriesInstanceUID = "1.2.840.113619.2.1.2.2.2"
        dicom_p2.SOPInstanceUID = "1.2.840.113619.2.1.2.3.2"

        response2 = (
            builder.set_patient_info([dicom_p2])
            .set_model_id("aneurysm_v1.0")
            .set_detections([dicom_p2], [aneurysm_prediction_result])
            .build()
        )

        # Verify data separation
        assert response1.patient_id == "PATIENT_001"
        assert response2.patient_id == "PATIENT_002"
        assert response1 is not response2
        assert response1.detections[0] is not response2.detections[0]

    def test_timestamp_consistency_across_predictions(self) -> None:
        """Inference timestamp is consistent within single response."""
        dicom = Dataset()
        dicom.PatientID = "TIME_TEST_001"
        dicom.StudyInstanceUID = "1.2.840.113619.2.1.2.1.1"
        dicom.SeriesInstanceUID = "1.2.840.113619.2.1.2.2.1"
        dicom.SOPInstanceUID = "1.2.840.113619.2.1.2.3.1"

        prediction = {
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

        builder = AneurysmDetectionBuilder()

        response = (
            builder.set_patient_info([dicom])
            .set_model_id("aneurysm_v1.0")
            .set_detections([dicom], [prediction])
            .build()
        )

        # Verify timestamp is UTC
        assert response.inference_timestamp.tzinfo == timezone.utc

        # Verify timestamp is recent (within last minute)
        now = datetime.now(timezone.utc)
        time_diff = (now - response.inference_timestamp).total_seconds()
        assert time_diff < 60  # Within 1 minute

    def test_json_serialization_round_trip(
        self, mock_dicom_dataset, aneurysm_prediction_result
    ) -> None:
        """Complete workflow with JSON serialization and deserialization."""
        builder = AneurysmDetectionBuilder()

        original_response = (
            builder.set_patient_info([mock_dicom_dataset])
            .set_model_id("aneurysm_v1.0")
            .set_detections([mock_dicom_dataset], [aneurysm_prediction_result])
            .build()
        )

        # Serialize to JSON
        json_data = original_response.model_dump_json()
        assert isinstance(json_data, str)

        # Deserialize back
        restored_response = AneurysmDetectionResponse.model_validate_json(
            json_data
        )

        # Verify round-trip preserves data
        assert restored_response.patient_id == original_response.patient_id
        assert restored_response.model_id == original_response.model_id
        assert len(restored_response.detections) == len(
            original_response.detections
        )
        assert (
            restored_response.detections[0].diameter
            == original_response.detections[0].diameter
        )

    def test_detection_with_boundary_probability_values(
        self, mock_dicom_dataset
    ) -> None:
        """Integration test with minimum and maximum probability values."""
        builder = AneurysmDetectionBuilder()

        # Prediction with minimum probability
        min_prob_prediction = {
            "type": "saccular",
            "location": "MCA",
            "diameter": 5.2,
            "main_seg_slice": 42,
            "probability": 0.0,  # Minimum valid
            "pitch_angle": 45,
            "yaw_angle": 30,
            "mask_index": 1,
            "sub_location": "M1_segment",
        }

        response_min = (
            builder.set_patient_info([mock_dicom_dataset])
            .set_model_id("aneurysm_v1.0")
            .set_detections([mock_dicom_dataset], [min_prob_prediction])
            .build()
        )

        assert response_min.detections[0].probability == 0.0

        # Prediction with maximum probability
        max_prob_prediction = {
            "type": "saccular",
            "location": "MCA",
            "diameter": 5.2,
            "main_seg_slice": 42,
            "probability": 1.0,  # Maximum valid
            "pitch_angle": 45,
            "yaw_angle": 30,
            "mask_index": 1,
            "sub_location": "M1_segment",
        }

        response_max = (
            builder.set_patient_info([mock_dicom_dataset])
            .set_model_id("aneurysm_v1.0")
            .set_detections([mock_dicom_dataset], [max_prob_prediction])
            .build()
        )

        assert response_max.detections[0].probability == 1.0

    def test_empty_detections_workflow(self, mock_dicom_dataset) -> None:
        """Workflow with no detections (no aneurysms found)."""
        builder = AneurysmDetectionBuilder()

        response = (
            builder.set_patient_info([mock_dicom_dataset])
            .set_model_id("aneurysm_v1.0")
            .build()
        )

        # Manually set empty detections before build
        builder._detections = []
        response = (
            builder.set_patient_info([mock_dicom_dataset])
            .set_model_id("aneurysm_v1.0")
            .build()
        )

        assert len(response.detections) == 0
        assert response.patient_id == "TEST_PATIENT_001"