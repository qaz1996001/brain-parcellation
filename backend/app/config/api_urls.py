"""API URL configuration helper for task parameter injection.

This module provides centralized configuration for API URLs used by task dispatchers.
Tasks receive the URL as a parameter rather than reading from environment directly,
enabling flexible routing in dual deployment scenarios.
"""

import os
from typing import Optional


def get_upload_data_api_url(override: Optional[str] = None) -> str:
    """Get the upload data API URL for task parameter injection.

    This function provides the API URL that task dispatchers should pass to workers
    as the `upload_data_api_url` parameter. In dual deployment scenarios, this allows
    Production and Testing backends to route tasks to different API endpoints while
    sharing the same worker pool.

    Args:
        override: Optional URL override. If provided, this URL is used instead of
                 reading from environment. Useful for testing or dynamic routing.

    Returns:
        str: The API URL to pass to task parameters.

    Raises:
        ValueError: If neither override nor UPLOAD_DATA_API_URL environment variable
                   is set, or if the URL format is invalid.

    Examples:
        >>> # Production backend
        >>> os.environ['UPLOAD_DATA_API_URL'] = 'http://production-api.example.com'
        >>> url = get_upload_data_api_url()
        >>> task_dict['upload_data_api_url'] = url
        >>> dicom_2_nii_series.push(task_dict)

        >>> # Testing backend with override
        >>> url = get_upload_data_api_url(override='http://test-api.example.com')
        >>> task_dict['upload_data_api_url'] = url
        >>> dicom_2_nii_series.push(task_dict)
    """
    # Use override if provided, otherwise read from environment
    api_url = override if override is not None else os.getenv("UPLOAD_DATA_API_URL")

    if api_url is None:
        raise ValueError(
            "upload_data_api_url must be provided via override parameter or "
            "UPLOAD_DATA_API_URL environment variable must be set"
        )

    # Validate URL format
    if not api_url.startswith(("http://", "https://")):
        raise ValueError(
            f"upload_data_api_url must be a valid HTTP/HTTPS URL, got: {api_url}"
        )

    return api_url
