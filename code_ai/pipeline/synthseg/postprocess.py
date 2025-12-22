from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from code_ai.utils_parcellation import (
    CMBProcess,
    DWIProcess,
    run_wmh,
    run_with_WhiteMatterParcellation,
)


def compute_white_matter_masks(
    synthseg_array: np.ndarray,
    synthseg33_array: np.ndarray,
    depth_number: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the detailed segmentation (david) and the white matter mask.
    """
    return run_with_WhiteMatterParcellation(
        synthseg_array=synthseg_array,
        synthseg33=synthseg33_array,
        depth_number=depth_number,
    )


def compute_cmb(seg_array: np.ndarray) -> np.ndarray:
    return CMBProcess.run(seg_array)


def compute_dwi(seg_array: np.ndarray) -> np.ndarray:
    return DWIProcess.run(seg_array)


def compute_wmh(
    synthseg_array: np.ndarray,
    synthseg_array_wm: np.ndarray,
    depth_number: int,
) -> np.ndarray:
    return run_wmh(
        synthseg_array=synthseg_array,
        synthseg_array_wm=synthseg_array_wm,
        depth_number=depth_number,
    )


def save_volume(
    array: np.ndarray,
    reference: nib.Nifti1Image,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array, reference.affine, reference.header)
    nib.save(image, str(output))

