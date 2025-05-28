# app/routers/series/__main__.py
from functools import lru_cache
from code_ai.dicom2nii.convert import ConvertManager
from code_ai.dicom2nii.convert.base import ImageOrientationProcessingStrategy


@lru_cache
def get_rename_dicom_manager() -> ConvertManager:
    return ConvertManager(input_path='',output_path='')


@lru_cache
def get_dicom_orientation() -> ImageOrientationProcessingStrategy:
    return ImageOrientationProcessingStrategy()

