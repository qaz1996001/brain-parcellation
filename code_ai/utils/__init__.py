from __future__ import annotations

import pathlib
import re

STUDY_ID_PATTERN = re.compile(
    r".*([0-9]{8,11}_[0-9]{8}_(MR|CT|PR|CR)_E?[0-9]{8,14})+.*",
    re.IGNORECASE,
)
study_id_pattern = STUDY_ID_PATTERN


def check_study_id(input_path: pathlib.Path) -> bool:
    """Return True when the folder name follows the standard study-id format."""
    if not input_path.is_dir():
        return False
    return STUDY_ID_PATTERN.match(input_path.name) is not None


def replace_suffix(
    filename: str, new_suffix: str, pattern: str = r"\.nii\.gz$|\.nii$"
) -> str:
    """Replace .nii / .nii.gz suffixes with the provided suffix."""
    return re.sub(pattern, new_suffix, filename)
