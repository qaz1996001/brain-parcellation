#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output directory management for model-organized structure

Author: Architecture Redesign Team
Created: 2025-10-20
"""
from pathlib import Path
from typing import Optional


class OutputManager:
    """Manages output directory structure per model specification

    Output structure:
        {output_dir}/{study_id}/
        ├── aneurysm_model/
        │   ├── {study_id}.json
        │   ├── {study_id}_A01.dcm
        │   └── {study_id}_A02.dcm
        ├── vessel_model/
        │   ├── {study_id}.json
        │   └── {study_id}.dcm
        └── cmb_model/
            ├── {study_id}.json
            ├── {study_id}_A01.dcm
            ├── ...
            └── {study_id}_A05.dcm

    Example:
        >>> manager = OutputManager(Path("/results"), "Study_12345")
        >>> aneurysm_dir = manager.get_model_dir("aneurysm")
        >>> # Returns: /results/Study_12345/aneurysm_model/
        >>>
        >>> json_path = manager.get_json_path("aneurysm")
        >>> # Returns: /results/Study_12345/aneurysm_model/Study_12345.json
        >>>
        >>> dicom_path = manager.get_dicom_path("aneurysm", slice_num=1)
        >>> # Returns: /results/Study_12345/aneurysm_model/Study_12345_A01.dcm
    """

    def __init__(self, base_output_dir: Path, study_id: str):
        """Initialize OutputManager

        Args:
            base_output_dir: Base output directory path
            study_id: Study/Patient ID for organizing outputs

        Raises:
            ValueError: If study_id is empty
        """
        if not study_id or not study_id.strip():
            raise ValueError("study_id cannot be empty")

        self.base_dir = Path(base_output_dir) / study_id
        self.study_id = study_id

    def get_model_dir(self, model_name: str) -> Path:
        """Get model-specific output directory

        Args:
            model_name: Model identifier (e.g., 'aneurysm', 'cmb', 'vessel')

        Returns:
            Path to model directory: {base_dir}/{model_name}_model/

        Example:
            >>> manager.get_model_dir("aneurysm")
            Path('/output/Study123/aneurysm_model/')

        Note:
            Directory is created automatically if it doesn't exist
        """
        model_dir = self.base_dir / f"{model_name}_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_output_path(self, model_name: str, filename: str) -> Path:
        """Get full path for output file in model directory

        Args:
            model_name: Model identifier (aneurysm, cmb, vessel)
            filename: Output filename (e.g., 'Study_12345.json', 'Pred_CMB.nii.gz')

        Returns:
            Full path to output file

        Example:
            >>> manager.get_output_path("cmb", "Pred_CMB.nii.gz")
            Path('/output/Study123/cmb_model/Pred_CMB.nii.gz')
        """
        return self.get_model_dir(model_name) / filename

    def get_json_path(self, model_name: str) -> Path:
        """Get path for model metadata JSON file

        Args:
            model_name: Model identifier

        Returns:
            Path to JSON file: {model_dir}/{study_id}.json

        Example:
            >>> manager.get_json_path("aneurysm")
            Path('/output/Study123/aneurysm_model/Study123.json')
        """
        return self.get_output_path(model_name, f"{self.study_id}.json")

    def get_dicom_path(self, model_name: str, slice_num: Optional[int] = None) -> Path:
        """Get path for DICOM-seg output file

        Args:
            model_name: Model identifier
            slice_num: Slice number for multi-slice outputs (1-based indexing)
                      If None, returns single DICOM file path

        Returns:
            Path to DICOM file:
                - With slice_num: {model_dir}/{study_id}_A{slice_num:02d}.dcm
                - Without slice_num: {model_dir}/{study_id}.dcm

        Examples:
            >>> manager.get_dicom_path("aneurysm", slice_num=1)
            Path('/output/Study123/aneurysm_model/Study123_A01.dcm')

            >>> manager.get_dicom_path("vessel")
            Path('/output/Study123/vessel_model/Study123.dcm')
        """
        if slice_num is not None:
            if slice_num < 1:
                raise ValueError(f"slice_num must be >= 1, got {slice_num}")
            filename = f"{self.study_id}_A{slice_num:02d}.dcm"
        else:
            filename = f"{self.study_id}.dcm"

        return self.get_output_path(model_name, filename)

    def get_nifti_path(self, model_name: str, suffix: str) -> Path:
        """Get path for NIfTI output file

        Args:
            model_name: Model identifier
            suffix: File suffix/name (e.g., 'Pred_Aneurysm', 'Prob_Aneurysm')

        Returns:
            Path to NIfTI file: {model_dir}/{suffix}.nii.gz

        Example:
            >>> manager.get_nifti_path("aneurysm", "Pred_Aneurysm")
            Path('/output/Study123/aneurysm_model/Pred_Aneurysm.nii.gz')
        """
        if not suffix.endswith('.nii.gz'):
            suffix = f"{suffix}.nii.gz"

        return self.get_output_path(model_name, suffix)

    def list_model_outputs(self, model_name: str) -> list[Path]:
        """List all output files in model directory

        Args:
            model_name: Model identifier

        Returns:
            List of paths to all files in model directory

        Example:
            >>> manager.list_model_outputs("aneurysm")
            [Path('.../Study123.json'), Path('.../Study123_A01.dcm'), ...]
        """
        model_dir = self.get_model_dir(model_name)
        if not model_dir.exists():
            return []

        return sorted([f for f in model_dir.iterdir() if f.is_file()])

    def get_legacy_output_dir(self) -> Path:
        """Get legacy output directory (flat structure)

        Returns:
            Base directory without model organization

        Note:
            Used for backward compatibility with --legacy-output flag
        """
        legacy_dir = self.base_dir / "legacy"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        return legacy_dir

    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"OutputManager(base_dir={self.base_dir}, study_id='{self.study_id}')"
