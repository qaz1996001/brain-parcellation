# -*- coding: utf-8 -*-
"""
DICOM-SEG Generator

This script creates multiple DICOM-SEG images from available mask data.
Due to a bug in Cornerstone (a JavaScript library for medical imaging)
that prevents reading mask IDs, each mask must be stored in a separate DICOM-SEG file.
The example cases provided include CMB (Cerebral Microbleeds).

This module handles:
1. Loading DICOM series and mask data (in NIfTI format)
2. Converting mask data to DICOM-SEG format
3. Creating required JSON metadata for each mask
4. Saving DICOM-SEG files and associated metadata

@author: sean
"""
import json
import os
import pathlib
from typing import Dict, List, Any, Union, Optional

import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk
import matplotlib.colors as mcolors
import pydicom_seg
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir

from code_ai.pipeline import pipeline_parser
from code_ai.pipeline.dicomseg import DCM_EXAMPLE
from code_ai.pipeline.dicomseg.schema import MaskRequest, MaskSeriesRequest, MaskInstanceRequest
from code_ai.pipeline.dicomseg.schema import StudyRequest, SortedRequest
from code_ai.pipeline.dicomseg.schema import AITeamRequest, GROUP_ID

from code_ai.utils.inference.config import CONFIG_DICT


def make_mask_instance_json(source_images: List[Union[FileDataset, DicomDir]],
                            dcm_seg: Union[FileDataset, DicomDir],
                            mask_index: int,
                            main_seg_slice: int,
                            *args, **kwargs) -> MaskInstanceRequest:
    """
    Create a JSON representation of a single mask instance.

    Args:
        source_images: List of DICOM image datasets
        dcm_seg: The DICOM segmentation dataset
        mask_index: Index of the mask
        main_seg_slice: Index of the slice containing the main segment
        *args, **kwargs: Additional parameters like diameter, type, location, etc.

    Returns:
        MaskInstanceRequest: Validated mask instance data
    """
    mask_instance_dict = dict()
    # Extract UIDs from the DICOM-SEG and source images
    seg_sop_instance_uid = dcm_seg.get((0x008, 0x0018)).value
    seg_series_instance_uid = dcm_seg.get((0x020, 0x000E)).value
    dicom_sop_instance_uid = source_images[main_seg_slice].get((0x008, 0x0018)).value
    # Extract mask name from DICOM-SEG Series Description
    # e.g. (0008,103E) Series Description: synthseg_SWAN_original_CMB_from_T1BRAVO_AXI_original_CMB
    mask_name = dcm_seg.get((0x0008, 0x103E)).value

    # Get additional mask parameters from kwargs or use defaults
    diameter = kwargs.get('diameter', '0.0')
    type = kwargs.get('type', 'saccular')
    location = kwargs.get('location', 'M')
    sub_location = kwargs.get('sub_location', '2')
    prob_max = kwargs.get('prob_max', '1.0')

    # Create mask instance dictionary with all required metadata
    mask_instance_dict.update({
        'mask_index': mask_index,
        'mask_name': mask_name,
        'diameter': diameter,
        'type': type,
        'location': location,
        'sub_location': sub_location,
        'prob_max': prob_max,
        'checked': "1",
        'is_ai': "1",
        'seg_sop_instance_uid': seg_sop_instance_uid,
        'seg_series_instance_uid': seg_series_instance_uid,
        'dicom_sop_instance_uid': dicom_sop_instance_uid,
        'main_seg_slice': main_seg_slice,
        'is_main_seg': "1"
    })
    # Validate and return the mask instance request
    return MaskInstanceRequest.model_validate(mask_instance_dict)


def make_mask_series_json(source_images: List[Union[FileDataset, DicomDir]],
                          sorted_dcms: List[Union[str, pathlib.Path]],
                          dcm_seg: Union[FileDataset, DicomDir],
                          mask_index: int,
                          main_seg_slice: int,
                          *args, **kwargs) -> MaskSeriesRequest:
    """
    Create a JSON representation of a mask series, which contains one or more mask instances.

    Args:
        source_images: List of DICOM image datasets
        sorted_dcms: List of sorted DICOM file paths
        dcm_seg: The DICOM segmentation dataset
        mask_index: Index of the mask
        main_seg_slice: Index of the slice containing the main segment
        *args, **kwargs: Additional parameters for mask instances

    Returns:
        MaskSeriesRequest: Validated mask series data
    """
    mask_series_dict = dict()
    mask_instance_list = []
    # Get the series UID and type from the source images
    series_instance_uid = source_images[0].get((0x0020, 0x000E)).value
    series_type = os.path.basename(os.path.dirname(sorted_dcms[0]))

    # Update the series dictionary with basic information
    mask_series_dict.update({
        'series_instance_uid': series_instance_uid,
        'series_type': series_type,
    })

    # Create a mask instance for this series
    mask_instance = make_mask_instance_json(source_images,
                                            dcm_seg,
                                            mask_index,
                                            main_seg_slice,
                                            *args, **kwargs)
    mask_instance_list.append(mask_instance)

    # Add the instances to the series dictionary
    mask_series_dict.update({'instances': mask_instance_list})

    # Validate and return the mask series request
    return MaskSeriesRequest.model_validate(mask_series_dict)


def make_mask_json(source_images: List[Union[FileDataset, DicomDir]],
                   sorted_dcms: List[Union[str, pathlib.Path]],
                   reslut_list: List[Dict[str, Any]],
                   group_id: int = GROUP_ID,
                   *args, **kwargs) -> MaskRequest:
    """
    Create a JSON representation of all masks for a study.

    Args:
        source_images: List of DICOM image datasets
        sorted_dcms: List of sorted DICOM file paths
        reslut_list: List of dictionaries containing mask results
        group_id: Group ID for the masks (default from GROUP_ID constant)
        *args, **kwargs: Additional parameters for mask instances

    Returns:
        MaskRequest: Validated mask request data containing all series
    """
    mask_dict = dict()
    mask_series_list = []

    # Process each result in the result list
    for index, reslut in enumerate(reslut_list):
        dcm_seg_path = reslut['dcm_seg_path']
        main_seg_slice = reslut['main_seg_slice']
        mask_index = reslut['mask_index']

        # Read the DICOM-SEG file
        with open(dcm_seg_path, 'rb') as f:
            dcm_seg = pydicom.read_file(f)

        # Create mask series JSON for this result
        mask_series = make_mask_series_json(source_images,
                                            sorted_dcms,
                                            dcm_seg,
                                            mask_index,
                                            main_seg_slice,
                                            *args, **kwargs)
        mask_series_list.append(mask_series)

        # If this is the first result, add study information to the mask dictionary
        if index == 0:
            study_instance_uid = source_images[0].get((0x0020, 0x000D)).value
            mask_dict.update({
                'study_instance_uid': study_instance_uid,
                'group_id': group_id
            })

    # Add the series list to the mask dictionary
    mask_dict.update({'series': mask_series_list})

    # Validate and return the mask request
    return MaskRequest.model_validate(mask_dict)


def make_sorted_json(source_images: List[Union[FileDataset, DicomDir]]) -> SortedRequest:
    """
    Create a JSON representation of sorted DICOM instances for a study.

    Args:
        source_images: List of DICOM image datasets

    Returns:
        SortedRequest: Validated sorted request data
    """
    instance_list = []
    series_list = []
    series_dict = dict()
    sorted_dict = dict()

    # Process each DICOM image
    for index, dicom_ds in enumerate(source_images):
        instance_dict = dict()
        # Extract SOP Instance UID and Image Position Patient
        sop_instance_uid = dicom_ds.get((0x0008, 0x0018)).value
        image_osition_patient = str(dicom_ds.get((0x0020, 0x0032)).value[-1])
        # Format: (0020,0032) Image Position Patient -14.7431\-143.248\98.5056

        # Update instance dictionary
        instance_dict.update(dict(
            sop_instance_uid=sop_instance_uid,
            projection=str(image_osition_patient)
        ))
        instance_list.append(instance_dict)

        # If this is the first image, add study and series information
        if index == 0:
            study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
            series_instance_uid = dicom_ds.get((0x0020, 0x000E)).value
            series_dict.update(dict(
                series_instance_uid=series_instance_uid,
                instance=instance_list
            ))
            series_list.append(series_dict)
            sorted_dict.update({'study_instance_uid': study_instance_uid})

    # Add the series list to the sorted dictionary
    sorted_dict.update({'series': series_list})

    # Validate and return the sorted request
    return SortedRequest.model_validate(sorted_dict)


def make_study_json(source_images: List[Union[FileDataset, DicomDir]],
                    group_id: int = GROUP_ID) -> StudyRequest:
    """
    Create a JSON representation of a study.

    Note: Previous version with commented-out code has been simplified.
    Only essential metadata is extracted from the first DICOM image.

    Args:
        source_images: List of DICOM image datasets
        group_id: Group ID for the study (default from GROUP_ID constant)

    Returns:
        StudyRequest: Validated study request data
    """
    study_dict = dict()
    dicom_ds = source_images[0]

    # Extract study metadata from the first DICOM image
    study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
    study_date = dicom_ds.get((0x0008, 0x0020)).value
    gender = dicom_ds.get((0x0010, 0x0040)).value
    age = dicom_ds.get((0x0010, 0x1010)).value
    study_name = dicom_ds.get((0x0008, 0x1030)).value
    patient_name = dicom_ds.get((0x0010, 0x0010)).value
    resolution_x = dicom_ds.get((0x0028, 0x0010)).value
    resolution_y = dicom_ds.get((0x0028, 0x0011)).value
    patient_id = dicom_ds.get((0x0010, 0x0020)).value

    # Update study dictionary with metadata
    study_dict.update(dict(
        group_id=group_id,
        study_instance_uid=study_instance_uid,
        study_date=study_date,
        gender=gender,
        age=age,
        study_name=study_name,
        patient_name=patient_name,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        patient_id=patient_id,
    ))

    # Validate and return the study request
    return StudyRequest.model_validate(study_dict)


def compute_orientation(init_axcodes, final_axcodes):
    """
    Calculate orientation transformation between initial and final axis codes.
    A thin wrapper around nibabel's ornt_transform function.

    Args:
        init_axcodes: Initial orientation codes (e.g., ('L', 'P', 'S'))
        final_axcodes: Target orientation codes (e.g., ('R', 'A', 'S'))

    Returns:
        tuple: Orientation transformation matrix, initial orientation, final orientation
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)
    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

    return ornt_transf, ornt_init, ornt_fin


def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    Reorient a 3D array from one orientation to another.
    Source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation

    Args:
        data_array: 3D array to reorient
        init_axcodes: Initial orientation codes
        final_axcodes: Target orientation codes

    Returns:
        numpy.ndarray: Reoriented data array
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)

    # If orientations are already the same, return the original data
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    # Apply the orientation transformation
    return nib.orientations.apply_orientation(data_array, ornt_transf)


def get_dicom_seg_template(model: str, label_dict: Dict) -> Dict:
    """
    Create a DICOM-SEG template with segment attributes.

    Args:
        model: Series description or model name
        label_dict: Dictionary of label IDs and attributes

    Returns:
        Dict: DICOM-SEG template dictionary
    """
    unique_labels = list(label_dict.keys())
    segment_attributes = []

    # Create segment attributes for each label
    for idx in unique_labels:
        name = label_dict[idx]["SegmentLabel"]
        # Convert color to RGB values (0-255)
        rgb_rate = mcolors.to_rgb(label_dict[idx]["color"])
        rgb = [int(y * 255) for y in rgb_rate]

        # Create segment attribute dictionary
        segment_attribute = {
            "labelID": int(idx),
            "SegmentLabel": name,
            "SegmentAlgorithmType": "MANUAL",
            "SegmentAlgorithmName": "SHH",
            "SegmentedPropertyCategoryCodeSequence": {
                "CodeValue": "M-01000",
                "CodingSchemeDesignator": "SRT",
                "CodeMeaning": "Morphologically Altered Structure",
            },
            "SegmentedPropertyTypeCodeSequence": {
                "CodeValue": "M-35300",
                "CodingSchemeDesignator": "SRT",
                "CodeMeaning": "Embolus",
            },
            "recommendedDisplayRGBValue": rgb,
        }
        segment_attributes.append(segment_attribute)

    # Create the template with all required metadata
    template = {
        "ContentCreatorName": "Reader1",
        "ClinicalTrialSeriesID": "Session1",
        "ClinicalTrialTimePointID": "1",
        "SeriesDescription": model,
        "SeriesNumber": "300",
        "InstanceNumber": "1",
        "segmentAttributes": [segment_attributes],
        "ContentLabel": "SEGMENTATION",
        "ContentDescription": "SHH",
        "ClinicalTrialCoordinatingCenterName": "SHH",
        "BodyPartExamined": "",
    }

    return template


def load_and_sort_dicom_files(path_dcms: str) -> tuple[
    List[Any], Any, FileDataset | DicomDir, list[FileDataset | DicomDir]]:
    """
    Load and sort DICOM files from a directory.
    This function only needs to be executed once per directory.

    Args:
        path_dcms: Path to the directory containing DICOM files

    Returns:
        tuple: (sorted_dcms, image, first_dcm, source_images)
            - sorted_dcms: List of sorted DICOM file paths
            - image: SimpleITK image object
            - first_dcm: First DICOM dataset
            - source_images: List of all DICOM datasets (without pixel data)
    """
    # Read DICOM file paths
    reader = sitk.ImageSeriesReader()
    dcms = sorted(reader.GetGDCMSeriesFileNames(path_dcms))

    # Read all slices
    slices = [pydicom.dcmread(dcm) for dcm in dcms]

    # Sort slices by position
    slice_dcm = []
    for (slice_data, dcm_slice) in zip(slices, dcms):
        # Get Image Orientation Patient (IOP)
        IOP = np.array(slice_data.get((0x0020, 0x0037)).value)
        # Get Image Position Patient (IPP)
        IPP = np.array(slice_data.get((0x0020, 0x0032)).value)
        # Calculate normal vector to the image plane
        normal = np.cross(IOP[0:3], IOP[3:])
        # Project IPP onto the normal vector
        projection = np.dot(IPP, normal)
        slice_dcm.append({"d": projection, "dcm": dcm_slice})

    # Sort slices by projection value
    slice_dcms = sorted(slice_dcm, key=lambda i: i['d'])
    sorted_dcms = [y['dcm'] for y in slice_dcms]

    # Read the image data
    reader.SetFileNames(sorted_dcms)
    image = reader.Execute()

    # Read the first DICOM image for metadata
    first_dcm = pydicom.dcmread(sorted_dcms[0], force=True)

    # Preload all DICOM files (without pixel data to save memory)
    source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in sorted_dcms]

    return sorted_dcms, image, first_dcm, source_images


def transform_mask_for_dicom_seg(mask: np.ndarray) -> np.ndarray:
    """
    Transform a mask array to the format required for DICOM-SEG.

    Args:
        mask: Input mask array (y, x, z)

    Returns:
        numpy.ndarray: Transformed mask array (z, y, x) with necessary flips
    """
    # Convert format: (y, x, z) -> (z, x, y)
    segmentation_data = mask.transpose(2, 0, 1).astype(np.uint8)

    # Convert format: (z, x, y) -> (z, y, x)
    segmentation_data = np.swapaxes(segmentation_data, 1, 2)

    # Flip y and x axes to match DICOM coordinate system
    segmentation_data = np.flip(segmentation_data, 1)
    segmentation_data = np.flip(segmentation_data, 2)

    return segmentation_data


def make_dicomseg_file(mask: np.ndarray,
                       image: sitk.Image,
                       first_dcm: pydicom.FileDataset,
                       source_images: List[pydicom.FileDataset],
                       template_json: Dict) -> pydicom.FileDataset:
    """
    Create a DICOM-SEG file from a mask array.

    Args:
        mask: Binary mask array
        image: SimpleITK image with spatial information
        first_dcm: First DICOM dataset for metadata
        source_images: List of source DICOM datasets
        template_json: DICOM-SEG template

    Returns:
        pydicom.FileDataset: DICOM-SEG dataset
    """
    # Create template from JSON
    template = pydicom_seg.template.from_dcmqi_metainfo(template_json)

    # Set up the writer
    writer = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=False,
        skip_empty_slices=True,
        skip_missing_segment=True,
    )

    # Use the mask as provided (already transformed)
    segmentation_data = mask

    # Create SimpleITK image from the segmentation data
    segmentation = sitk.GetImageFromArray(segmentation_data)
    segmentation.CopyInformation(image)

    # Generate DICOM-SEG file
    dcm_seg = writer.write(segmentation, source_images)

    # Copy relevant information from the first DICOM image
    dcm_seg[0x10, 0x0010].value = first_dcm[0x10, 0x0010].value  # Patient's Name
    dcm_seg[0x20, 0x0011].value = first_dcm[0x20, 0x0011].value  # Series Number

    # Copy more metadata from the example DICOM-SEG file
    dcm_seg[0x5200, 0x9229].value = DCM_EXAMPLE[0x5200, 0x9229].value
    dcm_seg[0x5200, 0x9229][0][0x20, 0x9116][0][0x20, 0x0037].value = first_dcm[0x20, 0x0037].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x18, 0x0050].value = first_dcm[0x18, 0x0050].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x18, 0x0088].value = first_dcm[0x18, 0x0088].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x28, 0x0030].value = first_dcm[0x28, 0x0030].value

    return dcm_seg


def create_dicom_seg_file(pred_data_unique: np.ndarray,
                          pred_data: np.ndarray,
                          series_name: str,
                          output_folder: pathlib.Path,
                          image: Any,
                          first_dcm: FileDataset | DicomDir,
                          source_images: List[FileDataset | DicomDir],
                          ) -> List[Dict[str, Any]]:
    """
    Create DICOM-SEG files for each unique value in the prediction data.

    Args:
        pred_data_unique: Array of unique values in prediction data
        pred_data: Full prediction data array
        series_name: Name of the series
        output_folder: Output directory for DICOM-SEG files
        image: SimpleITK image
        first_dcm: First DICOM dataset
        source_images: List of source DICOM datasets

    Returns:
        List[Dict[str, Any]]: List of results with mask index, file path, and main slice
    """
    pred_data_unique_len = len(pred_data_unique)
    reslut_list = []

    # Process each unique value in the prediction data
    for index, i in enumerate(pred_data_unique):
        # Create a binary mask for this specific region
        mask = np.zeros_like(pred_data)
        mask[pred_data == i] = 1

        # Only create DICOM-SEG if the mask contains positive values
        if np.sum(mask) > 0:
            # Create label dictionary for this mask
            label_dict = {1: {'SegmentLabel': f'A{i}', 'color': 'red'}}

            # Create template for DICOM-SEG
            template_json = get_dicom_seg_template(series_name, label_dict)

            # Generate DICOM-SEG file
            dcm_seg = make_dicomseg_file(
                mask.astype('uint8'),
                image,
                first_dcm,
                source_images,
                template_json
            )

            # Find the median slice containing the mask (main slice)
            main_seg_slice = int(np.median(np.where(mask)[0]))

            # Save DICOM-SEG file
            dcm_seg_filename = f'{series_name}_{label_dict[1]["SegmentLabel"]}.dcm'
            dcm_seg_path = output_folder.joinpath(dcm_seg_filename)
            dcm_seg.save_as(dcm_seg_path)

            # Clear console line and show progress
            print(f" " * 100, end='\r')
            print(f"{index + 1}/{pred_data_unique_len} Saved: {dcm_seg_path}", end='\r')

            # Add result to the list if file was created successfully
            if dcm_seg_path.exists():
                reslut_list.append({
                    'mask_index': i,
                    'dcm_seg_path': dcm_seg_path,
                    'main_seg_slice': main_seg_slice
                })

    return reslut_list


def process_prediction_mask(pred_data: np.ndarray,
                            path_dcms: str,
                            series_name: str,
                            output_folder: pathlib.Path,
                            ) -> Optional[AITeamRequest]:
    """
    Process prediction mask and generate DICOM-SEG files with metadata.

    Args:
        pred_data: Prediction data array
        path_dcms: Path to DICOM directory
        series_name: Name of the series
        output_folder: Output directory for DICOM-SEG files

    Returns:
        Optional[AITeamRequest]: AITeamRequest object with study, sorted, and mask data,
                               or None if no unique prediction values are found
    """
    # Load DICOM files (only once)
    sorted_dcms, image, first_dcm, source_images = load_and_sort_dicom_files(path_dcms)

    # Create study and sorted JSON data
    study_request = make_study_json(
        source_images=source_images,
        group_id=GROUP_ID
    )
    sorted_request = make_sorted_json(source_images=source_images)

    # Initialize AI team dictionary
    at_team_dict = {
        "study": study_request,
        "sorted": sorted_request
    }

    # Initialize mask request list
    mask_request_list = []

    # Get unique values in prediction data (excluding 0 background)
    pred_data_unique = np.unique(pred_data)
    if len(pred_data_unique) < 1:
        return None
    else:
        pred_data_unique = pred_data_unique[1:]  # Exclude background value (0)

    # Sample mask data (these would typically come from analysis of each mask)


    # Create DICOM-SEG files for each unique region
    result_list = create_dicom_seg_file(
        pred_data_unique,
        pred_data,
        series_name,
        output_folder,
        image,
        first_dcm,
        source_images
    )

    # Create mask JSON data
    mask_request = make_mask_json(
        source_images=source_images,
        sorted_dcms=sorted_dcms,
        reslut_list=result_list,

    )

    # Create and update AI team request
    at_team_request = AITeamRequest.model_validate(at_team_dict)
    at_team_request.mask = mask_request
    at_team_request.study.aneurysm_lession = pred_data_unique.shape[0]

    return at_team_request


def main():
    """
    Main function to process command line arguments and execute the pipeline.

    This function:
    1. Parses command line arguments
    2. Loads DICOM and NIfTI data
    3. Processes the prediction mask
    4. Creates DICOM-SEG files
    5. Generates JSON metadata
    6. Saves all results to the specified output folder
    """
    # Parse command line arguments
    parser = pipeline_parser()
    args = parser.parse_args()

    # Extract arguments
    ID = args.ID
    path_dcms = pathlib.Path(args.InputsDicomDir)
    path_nii = pathlib.Path(args.Inputs[0])
    path_dcmseg = pathlib.Path(args.Output_folder)

    # Get series name from NIfTI file
    series = path_nii.name.split('.')[0]

    # Load prediction data from NIfTI file
    pred_nii = nib.load(path_nii)
    pred_data = np.array(pred_nii.dataobj)

    # Reorient prediction data to standard orientation
    pred_nii_obj_axcodes = tuple(nib.aff2axcodes(pred_nii.affine))
    new_nifti_array = do_reorientation(pred_data, pred_nii_obj_axcodes, ('S', 'P', 'L'))

    # Create output directory
    series_folder = path_dcmseg.joinpath(f'{ID}')
    if series_folder.is_dir():
        series_folder.mkdir(exist_ok=True, parents=True)
    else:
        series_folder.parent.mkdir(parents=True, exist_ok=True)

    # Process prediction mask and create DICOM-SEG files
    at_team_request: AITeamRequest = process_prediction_mask(new_nifti_array,
                                                             str(path_dcms),
                                                             series,
                                                             series_folder,)

    # Sort mask series by main_seg_slice for better organization
    at_team_request.mask.series = sorted(at_team_request.mask.series,
                                         key=lambda x: x.instances[0].main_seg_slice)
    # Save platform JSON metadata

    platform_json_path = series_folder.joinpath(path_nii.name.replace('.nii.gz', '_platform_json.json'))
    print('platform_json_path', platform_json_path)
    with open(platform_json_path, 'w') as f:
        f.write(at_team_request.model_dump_json())
    print("Processing complete!")


if __name__ == '__main__':
    main()
