"""Fixtures for code_ai.pipeline.rdx tests.

Provides reusable test fixtures for DICOM data, prediction results,
and builder instances following Linus-style testing patterns.
"""

from datetime import datetime, timezone
from typing import Any, Dict

import pytest
from pydicom.dataset import Dataset


@pytest.fixture
def mock_dicom_dataset() -> Dataset:
    """Create mock DICOM dataset with required tags.

    Returns:
        Dataset with PatientID, StudyInstanceUID, SeriesInstanceUID,
        and SOPInstanceUID tags for testing.
    """
    ds = Dataset()
    ds.PatientID = "TEST_PATIENT_001"
    ds.StudyInstanceUID = "1.2.840.113619.2.1.2.1.1"
    ds.SeriesInstanceUID = "1.2.840.113619.2.1.2.2.1"
    ds.SOPInstanceUID = "1.2.840.113619.2.1.2.3.1"
    return ds


@pytest.fixture
def aneurysm_prediction_result() -> Dict[str, Any]:
    """Create valid aneurysm prediction result dictionary.

    Returns:
        Dictionary containing all required fields for AneurysmDetectionItem:
        type, location, diameter, main_seg_slice, probability, angles, etc.
    """
    return {
        "type": "saccular",
        "location": "MCA",
        "diameter": 5.234,
        "main_seg_slice": 42,
        "probability": 0.95,
        "pitch_angle": 45,
        "yaw_angle": 30,
        "mask_index": 1,
        "sub_location": "M1_segment",
    }


@pytest.fixture
def vessel_prediction_result() -> Dict[str, Any]:
    """Create valid vessel dilation prediction result dictionary.

    Returns:
        Dictionary containing basic fields for VesselDilatedDetectionItem.
    """
    return {
        "type": "dilation",
        "location": "basilar_artery",
    }


@pytest.fixture
def utc_timestamp() -> datetime:
    """Create UTC-aware timestamp for testing.

    Returns:
        Current datetime with UTC timezone.
    """
    return datetime.now(timezone.utc)


@pytest.fixture
def naive_timestamp() -> datetime:
    """Create timezone-naive timestamp for testing validation.

    Returns:
        Current datetime without timezone information.
    """
    return datetime.now()


@pytest.fixture
def non_utc_timestamp() -> datetime:
    """Create non-UTC timezone-aware timestamp for testing conversion.

    Returns:
        Current datetime with a non-UTC timezone (for validator testing).
    """
    # Import here to avoid dependency issues
    from datetime import timedelta, timezone as tz

    # Create timezone offset: UTC+8
    offset_tz = tz(timedelta(hours=8))
    return datetime.now(offset_tz)