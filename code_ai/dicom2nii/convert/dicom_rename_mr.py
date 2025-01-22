import os
import pathlib
import re
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Union, List, Callable

import pydicom.errors
from pydicom import dcmread, FileDataset
from tqdm.auto import tqdm

from .base import MRRenameSeriesProcessingStrategy, ImageOrientationProcessingStrategy, ContrastProcessingStrategy, \
    ModalityProcessingStrategy, MRAcquisitionTypeProcessingStrategy
from .config import BaseEnum, NullEnum, MRSeriesRenameEnum, MRAcquisitionTypeEnum, SeriesEnum, T1SeriesRenameEnum, \
    ImageOrientationEnum, ContrastEnum, T2SeriesRenameEnum, ASLSEQSeriesRenameEnum, DSCSeriesRenameEnum, DTISeriesEnum, \
    RepetitionTimeEnum, BodyPartEnum


class DwiProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for DWI series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of b-values to corresponding MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types, in this case, only 2D.
    pattern : re.Pattern
        Regular expression pattern for series description matching DWI or AUTODIFF, case insensitive.
    """

    type_2D_series_rename_mapping = {
        MRSeriesRenameEnum.DWI: re.compile('.*(DWI|AUTODIFF).*', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,)
    type_2D_series_rename_dict = {
        MRSeriesRenameEnum.DWI0: {MRSeriesRenameEnum.DWI, MRSeriesRenameEnum.B_VALUES_0,
                                  ImageOrientationEnum.AXI, },
        MRSeriesRenameEnum.DWI1000: {MRSeriesRenameEnum.DWI, MRSeriesRenameEnum.B_VALUES_1000,
                                     ImageOrientationEnum.AXI},
    }
    image_orientation_processing_strategy = ImageOrientationProcessingStrategy()
    series_group_fn_list = []

    @classmethod
    def get_series_group_fn_list(cls):
        """Get the list of series group functions.

        Returns:
        list: List of functions for extracting series group information.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_image_orientation)
            cls.series_group_fn_list.append(cls.get_b_values)

        return cls.series_group_fn_list

    @classmethod
    def get_image_orientation(cls, dicom_ds: FileDataset):
        """Get the image orientation information from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, ImageOrientationEnum]: Image orientation information.
        """
        # (0008,0008)	Image Type	DERIVED\SECONDARY\REFORMATTED
        image_orientation = cls.image_orientation_processing_strategy.process(dicom_ds=dicom_ds)
        image_type = dicom_ds.get((0x08, 0x08))
        if image_type[2] == 'REFORMATTED':
            if image_orientation == ImageOrientationEnum.AXI:
                return ImageOrientationEnum.AXIr
            elif image_orientation == ImageOrientationEnum.SAG:
                return ImageOrientationEnum.SAGr
            elif image_orientation == ImageOrientationEnum.COR:
                return ImageOrientationEnum.CORr
            else:
                return image_orientation
        else:
            return image_orientation

    @classmethod
    def get_b_values(cls, dicom_ds: FileDataset):
        dicom_tag = dicom_ds.get((0x43, 0x1039))
        if dicom_tag:
            b_values = dicom_tag[0]
            if int(b_values) == int(MRSeriesRenameEnum.B_VALUES_0.value):
                return MRSeriesRenameEnum.B_VALUES_0
            elif int(b_values) == int(MRSeriesRenameEnum.B_VALUES_1000.value):
                return MRSeriesRenameEnum.B_VALUES_1000

        return NullEnum.NULL

    def type_process(self, dicom_ds: FileDataset, type_series_rename_mapping, type_series_rename_dict) -> \
            Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for T1 series renaming based on acquisition type.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.
        type_series_rename_mapping (dict): Mapping of series descriptions to corresponding regex patterns for the acquisition type.
        type_series_rename_dict (dict): Mapping of T1SeriesRenameEnum values to sets of attributes for the acquisition type.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in type_series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                series_group_set.add(MRSeriesRenameEnum.DWI)
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                for series_rename_enum, series_rename_group_set in type_series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum
        return NullEnum.NULL

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for T1 series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
            return self.type_process(dicom_ds, self.type_2D_series_rename_mapping, self.type_2D_series_rename_dict)
        return NullEnum.NULL


class ADCProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for ADC series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 2D and NullEnum.NULL.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.ADC: re.compile('.*(?<!e)(ADC|Apparent Diffusion Coefficient).*', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,
                                                                          NullEnum.NULL)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process a DICOM dataset for ADC series renaming.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]
            The renamed series enumeration or NullEnum.NULL if no match is found.
        """

        # Extract relevant DICOM tags
        image_type_tag = dicom_ds.get((0x08, 0x08))
        instance_creation_time = dicom_ds.get((0x08, 0x13))
        series_description = dicom_ds.get((0x08, 0x103E))

        # Check if image type is 'DERIVED' and instance creation time is available
        if image_type_tag[0] == 'DERIVED' and instance_creation_time is not None:
            # Iterate through the mapping and check for a match with the series description
            for key, pattern in self.series_rename_mapping.items():
                match_result = pattern.match(series_description.value)
                if match_result:
                    return key

        # If no match is found, return NullEnum.NULL
        return NullEnum.NULL


class EADCProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for eADC series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 3D and NullEnum.NULL.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.eADC: re.compile('.*(eADC).*', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,NullEnum.NULL)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process a DICOM dataset for eADC series renaming.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]
            The renamed series enumeration or NullEnum.NULL if no match is found.
        """

        # Extract series description from DICOM dataset
        series_description = dicom_ds.get((0x08, 0x103E))

        # Iterate through the mapping and check for a match with the series description
        for key, pattern in self.series_rename_mapping.items():
            match_result = pattern.match(series_description.value)
            if match_result:
                return key

        # If no match is found, return NullEnum.NULL
        return NullEnum.NULL


class SWANProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for SWAN series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 3D.
    series_group_fn_list : list
        List of functions to extract series groups for additional processing.
    series_rename_dict : dict
        Mapping of MRSeriesRenameEnum values to sets of SeriesEnum values for grouping.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.SWAN: re.compile('.*(?<!e)(SWAN).*', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)

    series_group_fn_list = []

    series_rename_dict = {
        MRSeriesRenameEnum.SWAN: {SeriesEnum.SWAN},
        MRSeriesRenameEnum.SWANmIP: {SeriesEnum.SWAN, SeriesEnum.mIP},
        MRSeriesRenameEnum.SWANPHASE: {SeriesEnum.SWAN, SeriesEnum.SWANPHASE},
    }

    @classmethod
    def get_mIP(cls, dicom_ds: FileDataset) -> Union[SeriesEnum, NullEnum]:
        """Extract mIP series group.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[SeriesEnum, NullEnum]
            The mIP series group or NullEnum.NULL if not found.
        """
        image_type = dicom_ds.get((0x08, 0x08))
        instance_creation_time = dicom_ds.get((0x08, 0x13))
        if image_type and instance_creation_time is not None:
            if image_type.value[-1] == 'MIN IP':
                return SeriesEnum.mIP
        return NullEnum.NULL

    @classmethod
    def get_mag(cls, dicom_ds: FileDataset) -> Union[SeriesEnum, NullEnum]:
        """Extract SWANPHASE series group.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[SeriesEnum, NullEnum]
            The SWANPHASE series group or NullEnum.NULL if not found.
        """
        image_type = dicom_ds.get((0x43, 0x102F))
        if image_type:
            if image_type.value == SeriesEnum.SWANPHASE.value:
                return SeriesEnum.SWANPHASE

        return NullEnum.NULL

    @classmethod
    def get_swan(cls, dicom_ds: FileDataset) -> Union[SeriesEnum, NullEnum]:
        """Extract SWAN series group.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[SeriesEnum, NullEnum]
            The SWAN series group or NullEnum.NULL if not found.
        """
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            if str(pulse_sequence_name.value).lower() == SeriesEnum.SWAN.value.lower():
                return SeriesEnum.SWAN
        return NullEnum.NULL

    @classmethod
    def get_series_group_fn_list(cls) -> List[Callable]:
        """Get the list of series group extraction functions.

        Returns:
        List[Callable]
            The list of series group extraction functions.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_swan)
            cls.series_group_fn_list.append(cls.get_mIP)
            cls.series_group_fn_list.append(cls.get_mag)
        return cls.series_group_fn_list

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process a DICOM dataset for SWAN series renaming.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]
            The renamed series enumeration or NullEnum.NULL if no match is found.
        """

        # Extract series description from DICOM dataset
        series_description = dicom_ds.get((0x08, 0x103E))

        # Iterate through the mapping and check for a match with the series description
        for series_rename_enum, series_pattern in self.series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                # Extract series groups using additional processing functions
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                # Check if the extracted series groups match any defined in series_rename_dict
                for series_rename_enum, series_rename_group_set in self.series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum

        # If no match is found, return NullEnum.NULL
        return NullEnum.NULL


class ESWANProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for eSWAN series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 3D.
    series_group_fn_list : list
        List of functions to extract series groups for additional processing.
    series_rename_dict : dict
        Mapping of MRSeriesRenameEnum values to sets of SeriesEnum values for grouping.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.eSWAN: re.compile('.*(SWAN).*', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)
    series_group_fn_list = []

    series_rename_dict = {
        MRSeriesRenameEnum.eSWAN: {SeriesEnum.eSWAN,SeriesEnum.ORIGINAL},
        MRSeriesRenameEnum.eSWANmIP: {SeriesEnum.eSWAN, SeriesEnum.mIP},
    }

    @classmethod
    def get_mIP(cls, dicom_ds: FileDataset) -> Union[SeriesEnum, NullEnum]:
        """Extract mIP series group.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[SeriesEnum, NullEnum]
            The mIP series group or NullEnum.NULL if not found.
        """
        image_type = dicom_ds.get((0x08, 0x08))
        instance_creation_time = dicom_ds.get((0x08, 0x13))
        if image_type and instance_creation_time is not None:
            if (image_type.value[-1] == 'MIN IP' or image_type.value[-1] == 'REFORMATTED') and \
                    instance_creation_time is not None:
                return SeriesEnum.mIP

        return NullEnum.NULL

    @classmethod
    def get_original(cls, dicom_ds: FileDataset) -> Union[SeriesEnum, NullEnum]:
        """Extract mIP series group.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[SeriesEnum, NullEnum]
            The mIP series group or NullEnum.NULL if not found.
        """
        image_type = dicom_ds.get((0x08, 0x08))
        if image_type is not None:
            if image_type.value[0] == 'ORIGINAL':
                return SeriesEnum.ORIGINAL
        return NullEnum.NULL

    @classmethod
    def get_eswan(cls, dicom_ds: FileDataset) -> Union[SeriesEnum, NullEnum]:
        """Extract eSWAN series group.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[SeriesEnum, NullEnum]
            The eSWAN series group or NullEnum.NULL if not found.
        """
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            if str(pulse_sequence_name.value).lower() == SeriesEnum.eSWAN.value.lower():
                return SeriesEnum.eSWAN
        return NullEnum.NULL

    @classmethod
    def get_series_group_fn_list(cls) -> List[Callable]:
        """Get the list of series group extraction functions.

        Returns:
        List[Callable]
            The list of series group extraction functions.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_eswan)
            cls.series_group_fn_list.append(cls.get_mIP)
            cls.series_group_fn_list.append(cls.get_original)
        return cls.series_group_fn_list

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process a DICOM dataset for eSWAN series renaming.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]
            The renamed series enumeration or NullEnum.NULL if no match is found.
        """

        # Extract series description from DICOM dataset
        series_description = dicom_ds.get((0x08, 0x103E))

        # Iterate through the mapping and check for a match with the series description
        for series_rename_enum, series_pattern in self.series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                # Extract series groups using additional processing functions
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                # Check if the extracted series groups match any defined in series_rename_dict
                for series_rename_enum, series_rename_group_set in self.series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum

        # If no match is found, return NullEnum.NULL
        return NullEnum.NULL


class MRABrainProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for MRA Brain series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 3D.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.MRA_BRAIN: re.compile('.+(TOF)(((?!Neck).)*)$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process a DICOM dataset for MRA Brain series renaming.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]
            The renamed series enumeration or NullEnum.NULL if no match is found.
        """

        # Check if modality is MR and acquisition type is 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        pattern = self.series_rename_mapping[MRSeriesRenameEnum.MRA_BRAIN]
        match_result = pattern.match(series_description.value)

        if match_result:
            image_type_tag = dicom_ds.get((0x08, 0x08))
            # Check if image type is 'ORIGINAL'
            if image_type_tag[0] == 'ORIGINAL':
                return MRSeriesRenameEnum.MRA_BRAIN

        # If no match or image type is not 'ORIGINAL', return NullEnum.NULL
        return NullEnum.NULL


class MRANeckProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for MRA Neck series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 3D.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.MRA_NECK: re.compile('.*(TOF).*((Neck+).*)$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process a DICOM dataset for MRA Neck series renaming.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]
            The renamed series enumeration or NullEnum.NULL if no match is found.
        """

        # Check if modality is MR and acquisition type is 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        pattern = self.series_rename_mapping[MRSeriesRenameEnum.MRA_NECK]
        match_result = pattern.match(series_description.value)

        if match_result:
            image_type_tag = dicom_ds.get((0x08, 0x08))
            # Check if image type is 'ORIGINAL'
            if image_type_tag[0] == 'ORIGINAL':
                return MRSeriesRenameEnum.MRA_NECK

        # If no match or image type is not 'ORIGINAL', return NullEnum.NULL
        return NullEnum.NULL


class MRAVRBrainProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for MRAVR Brain series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 3D.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.MRAVR_BRAIN: re.compile('((?!TOF|Neck).)*(MRA)((?!Neck).)*$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process a DICOM dataset for MRAVR Brain series renaming.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]
            The renamed series enumeration or NullEnum.NULL if no match is found.
        """

        # Check if modality is MR and acquisition type is 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        pattern = self.series_rename_mapping[MRSeriesRenameEnum.MRAVR_BRAIN]
        match_result = pattern.match(series_description.value)

        if match_result:
            return MRSeriesRenameEnum.MRAVR_BRAIN

        # If no match, return NullEnum.NULL
        return NullEnum.NULL


class MRAVRNeckProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for MRAVR Neck series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 3D.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.MRAVR_NECK: re.compile('((?!TOF).)*(Neck.*MRA)|(MRA.*Neck).*$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process a DICOM dataset for MRAVR Neck series renaming.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]
            The renamed series enumeration or NullEnum.NULL if no match is found.
        """

        # Check if modality is MR and acquisition type is 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        pattern = self.series_rename_mapping[MRSeriesRenameEnum.MRAVR_NECK]
        match_result = pattern.match(series_description.value)

        if match_result:
            return MRSeriesRenameEnum.MRAVR_NECK

        # If no match, return NullEnum.NULL
        return NullEnum.NULL


class T1ProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for T1 series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and T1SeriesRenameEnum or SeriesEnum values.
    type_3D_series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns for 3D acquisitions.
    type_2D_series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns for 2D acquisitions.
    type_2D_series_rename_dict : dict
        Mapping of T1SeriesRenameEnum values to sets of attributes for 2D acquisitions.
    type_3D_series_rename_dict : dict
        Mapping of T1SeriesRenameEnum values to sets of attributes for 3D acquisitions.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 3D and 2D.
    image_orientation_processing_strategy : ImageOrientationProcessingStrategy
        Strategy for extracting image orientation information.
    contrast_processing_strategy : ContrastProcessingStrategy
        Strategy for extracting contrast information.
    series_group_fn_list : list
        List containing functions for extracting series group information.
    """

    series_rename_mapping = {
        T1SeriesRenameEnum.T1: re.compile('.*(T1|AX|COR|SAG).*', re.IGNORECASE),
        SeriesEnum.FLAIR: re.compile('(FLAIR)', re.IGNORECASE),
        SeriesEnum.CUBE: re.compile('.*(CUBE).*', re.IGNORECASE),
        SeriesEnum.BRAVO: re.compile('.*(BRAVO|FSPGR).*', re.IGNORECASE),
    }
    type_3D_series_rename_mapping = series_rename_mapping

    type_2D_series_rename_mapping = {
        T1SeriesRenameEnum.T1: re.compile('.*(T1).*', re.IGNORECASE),
        SeriesEnum.FLAIR: re.compile('(FLAIR)', re.IGNORECASE),
    }

    type_2D_series_rename_dict = {
        T1SeriesRenameEnum.T1_AXI: {T1SeriesRenameEnum.T1, ImageOrientationEnum.AXI, ContrastEnum.NE},
        T1SeriesRenameEnum.T1_SAG: {T1SeriesRenameEnum.T1, ImageOrientationEnum.SAG, ContrastEnum.NE},
        T1SeriesRenameEnum.T1_COR: {T1SeriesRenameEnum.T1, ImageOrientationEnum.COR, ContrastEnum.NE},
        T1SeriesRenameEnum.T1CE_AXI: {T1SeriesRenameEnum.T1, ImageOrientationEnum.AXI, ContrastEnum.CE},
        T1SeriesRenameEnum.T1CE_SAG: {T1SeriesRenameEnum.T1, ImageOrientationEnum.SAG, ContrastEnum.CE},
        T1SeriesRenameEnum.T1CE_COR: {T1SeriesRenameEnum.T1, ImageOrientationEnum.COR, ContrastEnum.CE},

        T1SeriesRenameEnum.T1FLAIR_AXI: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, ImageOrientationEnum.AXI,
                                         ContrastEnum.NE},
        T1SeriesRenameEnum.T1FLAIR_SAG: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, ImageOrientationEnum.SAG,
                                         ContrastEnum.NE},
        T1SeriesRenameEnum.T1FLAIR_COR: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, ImageOrientationEnum.COR,
                                         ContrastEnum.NE},
        T1SeriesRenameEnum.T1FLAIRCE_AXI: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, ImageOrientationEnum.AXI,
                                           ContrastEnum.CE},
        T1SeriesRenameEnum.T1FLAIRCE_SAG: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, ImageOrientationEnum.SAG,
                                           ContrastEnum.CE},
        T1SeriesRenameEnum.T1FLAIRCE_COR: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, ImageOrientationEnum.COR,
                                           ContrastEnum.CE},
    }

    type_3D_series_rename_dict = {
        T1SeriesRenameEnum.T1CUBE_AXI: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.AXI,
                                        ContrastEnum.NE},
        T1SeriesRenameEnum.T1CUBE_SAG: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.SAG,
                                        ContrastEnum.NE},
        T1SeriesRenameEnum.T1CUBE_COR: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.COR,
                                        ContrastEnum.NE},
        T1SeriesRenameEnum.T1CUBECE_AXI: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.AXI,
                                          ContrastEnum.CE},
        T1SeriesRenameEnum.T1CUBECE_SAG: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.SAG,
                                          ContrastEnum.CE},
        T1SeriesRenameEnum.T1CUBECE_COR: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.COR,
                                          ContrastEnum.CE},
        T1SeriesRenameEnum.T1FLAIRCUBE_AXI: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                             ImageOrientationEnum.AXI,
                                             ContrastEnum.NE},
        T1SeriesRenameEnum.T1FLAIRCUBE_SAG: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                             ImageOrientationEnum.SAG,
                                             ContrastEnum.NE},
        T1SeriesRenameEnum.T1FLAIRCUBE_COR: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                             ImageOrientationEnum.COR,
                                             ContrastEnum.NE},
        T1SeriesRenameEnum.T1FLAIRCUBECE_AXI: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                               ImageOrientationEnum.AXI,
                                               ContrastEnum.CE},
        T1SeriesRenameEnum.T1FLAIRCUBECE_SAG: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                               ImageOrientationEnum.SAG,
                                               ContrastEnum.CE},
        T1SeriesRenameEnum.T1FLAIRCUBECE_COR: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                               ImageOrientationEnum.COR,
                                               ContrastEnum.CE},
        T1SeriesRenameEnum.T1BRAVO_AXI: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.AXI,
                                         ContrastEnum.NE},
        T1SeriesRenameEnum.T1BRAVO_SAG: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.SAG,
                                         ContrastEnum.NE},
        T1SeriesRenameEnum.T1BRAVO_COR: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.COR,
                                         ContrastEnum.NE},
        T1SeriesRenameEnum.T1BRAVOCE_AXI: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.AXI,
                                           ContrastEnum.CE},
        T1SeriesRenameEnum.T1BRAVOCE_SAG: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.SAG,
                                           ContrastEnum.CE},
        T1SeriesRenameEnum.T1BRAVOCE_COR: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.COR,
                                           ContrastEnum.CE},
        #
        T1SeriesRenameEnum.T1CUBE_AXIr: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.AXIr,
                                         ContrastEnum.NE},
        T1SeriesRenameEnum.T1CUBE_SAGr: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.SAGr,
                                         ContrastEnum.NE},
        T1SeriesRenameEnum.T1CUBE_CORr: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.CORr,
                                         ContrastEnum.NE},
        T1SeriesRenameEnum.T1CUBECE_AXIr: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.AXIr,
                                           ContrastEnum.CE},
        T1SeriesRenameEnum.T1CUBECE_SAGr: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.SAGr,
                                           ContrastEnum.CE},
        T1SeriesRenameEnum.T1CUBECE_CORr: {T1SeriesRenameEnum.T1, SeriesEnum.CUBE, ImageOrientationEnum.CORr,
                                           ContrastEnum.CE},
        T1SeriesRenameEnum.T1FLAIRCUBE_AXIr: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                              ImageOrientationEnum.AXIr,
                                              ContrastEnum.NE},
        T1SeriesRenameEnum.T1FLAIRCUBE_SAGr: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                              ImageOrientationEnum.SAGr,
                                              ContrastEnum.NE},
        T1SeriesRenameEnum.T1FLAIRCUBE_CORr: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                              ImageOrientationEnum.CORr,
                                              ContrastEnum.NE},
        T1SeriesRenameEnum.T1FLAIRCUBECE_AXIr: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                                ImageOrientationEnum.AXIr,
                                                ContrastEnum.CE},
        T1SeriesRenameEnum.T1FLAIRCUBECE_SAGr: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                                ImageOrientationEnum.SAGr,
                                                ContrastEnum.CE},
        T1SeriesRenameEnum.T1FLAIRCUBECE_CORr: {T1SeriesRenameEnum.T1, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                                ImageOrientationEnum.CORr,
                                                ContrastEnum.CE},
        T1SeriesRenameEnum.T1BRAVO_AXIr: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.AXIr,
                                          ContrastEnum.NE},
        T1SeriesRenameEnum.T1BRAVO_SAGr: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.SAGr,
                                          ContrastEnum.NE},
        T1SeriesRenameEnum.T1BRAVO_CORr: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.CORr,
                                          ContrastEnum.NE},
        T1SeriesRenameEnum.T1BRAVOCE_AXIr: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.AXIr,
                                            ContrastEnum.CE},
        T1SeriesRenameEnum.T1BRAVOCE_SAGr: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.SAGr,
                                            ContrastEnum.CE},

        T1SeriesRenameEnum.T1BRAVOCE_CORr: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.CORr,
                                            ContrastEnum.CE},
    }

    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,
                                                                          MRAcquisitionTypeEnum.TYPE_2D,
                                                                          )
    image_orientation_processing_strategy = ImageOrientationProcessingStrategy()
    contrast_processing_strategy = ContrastProcessingStrategy()
    series_group_fn_list = []

    @classmethod
    def get_series_group_fn_list(cls):
        """Get the list of series group functions.

        Returns:
        list: List of functions for extracting series group information.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_image_orientation)
            cls.series_group_fn_list.append(cls.get_contrast)
            cls.series_group_fn_list.append(cls.get_flair)
            cls.series_group_fn_list.append(cls.get_cube)
            cls.series_group_fn_list.append(cls.get_bravo)
        return cls.series_group_fn_list

    @classmethod
    def get_image_orientation(cls, dicom_ds: FileDataset):
        """Get the image orientation information from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, ImageOrientationEnum]: Image orientation information.
        """
        # (0008,0008)	Image Type	DERIVED\SECONDARY\REFORMATTED
        image_orientation = cls.image_orientation_processing_strategy.process(dicom_ds=dicom_ds)
        image_type = dicom_ds.get((0x08, 0x08))
        if image_type[2] == 'REFORMATTED':
            if image_orientation == ImageOrientationEnum.AXI:
                return ImageOrientationEnum.AXIr
            elif image_orientation == ImageOrientationEnum.SAG:
                return ImageOrientationEnum.SAGr
            elif image_orientation == ImageOrientationEnum.COR:
                return ImageOrientationEnum.CORr
            else:
                return image_orientation
        else:
            return image_orientation

    @classmethod
    def get_contrast(cls, dicom_ds: FileDataset):
        """Get the contrast information from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, ContrastEnum]: Contrast information.
        """
        contrast = cls.contrast_processing_strategy.process(dicom_ds=dicom_ds)
        return contrast

    @classmethod
    def get_flair(cls, dicom_ds: FileDataset, ):
        """Get series information for FLAIR sequences from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, T1SeriesRenameEnum, SeriesEnum]: FLAIR series information.
        """
        tr = float(dicom_ds[0x18, 0x80].value)
        te = float(dicom_ds[0x18, 0x81].value)
        if 800 <= tr <= 3000 and te <= 30:
            return T1SeriesRenameEnum.T1, SeriesEnum.FLAIR
        if tr <= 800 and te <= 30:
            return T1SeriesRenameEnum.T1
        return NullEnum.NULL

    @classmethod
    def get_cube(cls, dicom_ds: FileDataset, ):
        """Get series information for CUBE sequences from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, SeriesEnum]: CUBE series information.
        """
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            if str(pulse_sequence_name.value).lower() == SeriesEnum.CUBE.value.lower():
                return SeriesEnum.CUBE
        return NullEnum.NULL

    @classmethod
    def get_bravo(cls, dicom_ds: FileDataset, ):
        """Get series information for BRAVO sequences from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, T1SeriesRenameEnum, SeriesEnum]: BRAVO series information.
        """
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            pulse_sequence_name_str = str(pulse_sequence_name.value).lower()
            if (
                    pulse_sequence_name_str == SeriesEnum.BRAVO.value.lower()) or pulse_sequence_name_str == SeriesEnum.FSPGR.value.lower():
                return T1SeriesRenameEnum.T1, SeriesEnum.BRAVO
        return NullEnum.NULL

    def type_process(self, dicom_ds: FileDataset, type_series_rename_mapping, type_series_rename_dict) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for T1 series renaming based on acquisition type.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.
        type_series_rename_mapping (dict): Mapping of series descriptions to corresponding regex patterns for the acquisition type.
        type_series_rename_dict (dict): Mapping of T1SeriesRenameEnum values to sets of attributes for the acquisition type.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in type_series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                for series_rename_enum, series_rename_group_set in type_series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum

        return NullEnum.NULL

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for T1 series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
            return self.type_process(dicom_ds, self.type_2D_series_rename_mapping, self.type_2D_series_rename_dict)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_3D:
            return self.type_process(dicom_ds, self.type_3D_series_rename_mapping, self.type_3D_series_rename_dict)
        return NullEnum.NULL


class T2ProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for T2 series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and T2SeriesRenameEnum or SeriesEnum values.
    type_3D_series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns for 3D acquisitions.
    type_2D_series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns for 2D acquisitions.
    type_2D_series_rename_dict : dict
        Mapping of T2SeriesRenameEnum values to sets of attributes for 2D acquisitions.
    type_3D_series_rename_dict : dict
        Mapping of T2SeriesRenameEnum values to sets of attributes for 3D acquisitions.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 3D and 2D.
    image_orientation_processing_strategy : ImageOrientationProcessingStrategy
        Strategy for extracting image orientation information.
    contrast_processing_strategy : ContrastProcessingStrategy
        Strategy for extracting contrast information.
    series_group_fn_list : list
        List containing functions for extracting series group information.
    """

    series_rename_mapping = {
        T2SeriesRenameEnum.T2: re.compile('.*(T2).*', re.IGNORECASE),
        SeriesEnum.FLAIR: re.compile('(FLAIR)', re.IGNORECASE),
        SeriesEnum.CUBE: re.compile('.*(CUBE).*', re.IGNORECASE),
    }
    type_3D_series_rename_mapping = series_rename_mapping

    type_2D_series_rename_mapping = {
        T2SeriesRenameEnum.T2: re.compile('.*(T2).*', re.IGNORECASE),
        SeriesEnum.FLAIR: re.compile('(FLAIR)', re.IGNORECASE),
    }

    type_2D_series_rename_dict = {
        T2SeriesRenameEnum.T2_AXI: {T2SeriesRenameEnum.T2, ImageOrientationEnum.AXI, ContrastEnum.NE},
        T2SeriesRenameEnum.T2_SAG: {T2SeriesRenameEnum.T2, ImageOrientationEnum.SAG, ContrastEnum.NE},
        T2SeriesRenameEnum.T2_COR: {T2SeriesRenameEnum.T2, ImageOrientationEnum.COR, ContrastEnum.NE},
        T2SeriesRenameEnum.T2CE_AXI: {T2SeriesRenameEnum.T2, ImageOrientationEnum.AXI, ContrastEnum.CE},
        T2SeriesRenameEnum.T2CE_SAG: {T2SeriesRenameEnum.T2, ImageOrientationEnum.SAG, ContrastEnum.CE},
        T2SeriesRenameEnum.T2CE_COR: {T2SeriesRenameEnum.T2, ImageOrientationEnum.COR, ContrastEnum.CE},

        T2SeriesRenameEnum.T2FLAIR_AXI: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, ImageOrientationEnum.AXI,
                                         ContrastEnum.NE},
        T2SeriesRenameEnum.T2FLAIR_SAG: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, ImageOrientationEnum.SAG,
                                         ContrastEnum.NE},
        T2SeriesRenameEnum.T2FLAIR_COR: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, ImageOrientationEnum.COR,
                                         ContrastEnum.NE},
        T2SeriesRenameEnum.T2FLAIRCE_AXI: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, ImageOrientationEnum.AXI,
                                           ContrastEnum.CE},
        T2SeriesRenameEnum.T2FLAIRCE_SAG: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, ImageOrientationEnum.SAG,
                                           ContrastEnum.CE},
        T2SeriesRenameEnum.T2FLAIRCE_COR: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, ImageOrientationEnum.COR,
                                           ContrastEnum.CE},
    }

    type_3D_series_rename_dict = {
        T2SeriesRenameEnum.T2CUBE_AXI: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.AXI,
                                        ContrastEnum.NE},
        T2SeriesRenameEnum.T2CUBE_SAG: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.SAG,
                                        ContrastEnum.NE},
        T2SeriesRenameEnum.T2CUBE_COR: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.COR,
                                        ContrastEnum.NE},
        T2SeriesRenameEnum.T2CUBECE_AXI: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.AXI,
                                          ContrastEnum.CE},
        T2SeriesRenameEnum.T2CUBECE_SAG: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.SAG,
                                          ContrastEnum.CE},
        T2SeriesRenameEnum.T2CUBECE_COR: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.COR,
                                          ContrastEnum.CE},
        T2SeriesRenameEnum.T2FLAIRCUBE_AXI: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                             ImageOrientationEnum.AXI,
                                             ContrastEnum.NE},
        T2SeriesRenameEnum.T2FLAIRCUBE_SAG: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                             ImageOrientationEnum.SAG,
                                             ContrastEnum.NE},
        T2SeriesRenameEnum.T2FLAIRCUBE_COR: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                             ImageOrientationEnum.COR,
                                             ContrastEnum.NE},
        T2SeriesRenameEnum.T2FLAIRCUBECE_AXI: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                               ImageOrientationEnum.AXI,
                                               ContrastEnum.CE},
        T2SeriesRenameEnum.T2FLAIRCUBECE_SAG: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                               ImageOrientationEnum.SAG,
                                               ContrastEnum.CE},
        T2SeriesRenameEnum.T2FLAIRCUBECE_COR: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                               ImageOrientationEnum.COR,
                                               ContrastEnum.CE},

        T2SeriesRenameEnum.T2CUBE_AXIr: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.AXIr,
                                         ContrastEnum.NE},
        T2SeriesRenameEnum.T2CUBE_SAGr: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.SAGr,
                                         ContrastEnum.NE},
        T2SeriesRenameEnum.T2CUBE_CORr: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.CORr,
                                         ContrastEnum.NE},
        T2SeriesRenameEnum.T2CUBECE_AXIr: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.AXIr,
                                           ContrastEnum.CE},
        T2SeriesRenameEnum.T2CUBECE_SAGr: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.SAGr,
                                           ContrastEnum.CE},
        T2SeriesRenameEnum.T2CUBECE_CORr: {T2SeriesRenameEnum.T2, SeriesEnum.CUBE, ImageOrientationEnum.CORr,
                                           ContrastEnum.CE},
        T2SeriesRenameEnum.T2FLAIRCUBE_AXIr: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                              ImageOrientationEnum.AXIr,
                                              ContrastEnum.NE},
        T2SeriesRenameEnum.T2FLAIRCUBE_SAGr: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                              ImageOrientationEnum.SAGr,
                                              ContrastEnum.NE},
        T2SeriesRenameEnum.T2FLAIRCUBE_CORr: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                              ImageOrientationEnum.CORr,
                                              ContrastEnum.NE},
        T2SeriesRenameEnum.T2FLAIRCUBECE_AXIr: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                                ImageOrientationEnum.AXIr,
                                                ContrastEnum.CE},
        T2SeriesRenameEnum.T2FLAIRCUBECE_SAGr: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                                ImageOrientationEnum.SAGr,
                                                ContrastEnum.CE},
        T2SeriesRenameEnum.T2FLAIRCUBECE_CORr: {T2SeriesRenameEnum.T2, SeriesEnum.FLAIR, SeriesEnum.CUBE,
                                                ImageOrientationEnum.CORr,
                                                ContrastEnum.CE},
    }
    mr_acquisition_typemr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (
        MRAcquisitionTypeEnum.TYPE_3D,
        MRAcquisitionTypeEnum.TYPE_2D,
    )
    image_orientation_processing_strategy = ImageOrientationProcessingStrategy()
    contrast_processing_strategy = ContrastProcessingStrategy()
    series_group_fn_list = []

    @classmethod
    def get_series_group_fn_list(cls):
        """Get the list of series group functions.

        Returns:
        list: List of functions for extracting series group information.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_image_orientation)
            cls.series_group_fn_list.append(cls.get_contrast)
            cls.series_group_fn_list.append(cls.get_flair)
            cls.series_group_fn_list.append(cls.get_cube)
        return cls.series_group_fn_list

    @classmethod
    def get_image_orientation(cls, dicom_ds: FileDataset):
        """Get the image orientation information from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, ImageOrientationEnum]: Image orientation information.
        """
        # (0008,0008)	Image Type	DERIVED\SECONDARY\REFORMATTED
        image_orientation = cls.image_orientation_processing_strategy.process(dicom_ds=dicom_ds)
        image_type = dicom_ds.get((0x08, 0x08))
        if image_type[2] == 'REFORMATTED':
            if image_orientation == ImageOrientationEnum.AXI:
                return ImageOrientationEnum.AXIr
            elif image_orientation == ImageOrientationEnum.SAG:
                return ImageOrientationEnum.SAGr
            elif image_orientation == ImageOrientationEnum.COR:
                return ImageOrientationEnum.CORr
            else:
                return image_orientation
        else:
            return image_orientation

    @classmethod
    def get_contrast(cls, dicom_ds: FileDataset):
        """Get the contrast information from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, ContrastEnum]: Contrast information.
        """
        contrast = cls.contrast_processing_strategy.process(dicom_ds=dicom_ds)
        return contrast

    @classmethod
    def get_flair(cls, dicom_ds: FileDataset, ):
        """Get series information for FLAIR sequences from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, T2SeriesRenameEnum, SeriesEnum]: FLAIR series information.
        """
        # tr = float(dicom_ds[0x18, 0x80].value)
        te = float(dicom_ds[0x18, 0x81].value)
        # (0018,0082)	Inversion Time	2500
        ti = dicom_ds.get((0x18, 0x82))
        if te >= 80 and ti is not None:
            return T2SeriesRenameEnum.T2, SeriesEnum.FLAIR
        if te >= 80 and ti is None:
            return T2SeriesRenameEnum.T2
        # if 1000 <= tr < 6250 and te >= 80:
        #     return T2SeriesRenameEnum.T2
        # if 6250 <= tr <= 10000 and te >= 80:
        #     return T2SeriesRenameEnum.T2, SeriesEnum.FLAIR
        # if 1000 <= tr < 6250 and te >= 80:
        #     return T2SeriesRenameEnum.T2
        return NullEnum.NULL

    @classmethod
    def get_cube(cls, dicom_ds: FileDataset, ):
        """Get series information for CUBE sequences from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, SeriesEnum]: CUBE series information.
        """
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            if SeriesEnum.CUBE.value.lower() in str(pulse_sequence_name.value).lower():
                return SeriesEnum.CUBE
        return NullEnum.NULL

    def type_process(self, dicom_ds: FileDataset, type_series_rename_mapping, type_series_rename_dict) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for T2 series renaming based on acquisition type.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.
        type_series_rename_mapping (dict): Mapping of series descriptions to corresponding regex patterns for the acquisition type.
        type_series_rename_dict (dict): Mapping of T2SeriesRenameEnum values to sets of attributes for the acquisition type.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in type_series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                for series_rename_enum, series_rename_group_set in type_series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum

        return NullEnum.NULL

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for T2 series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
            return self.type_process(dicom_ds, self.type_2D_series_rename_mapping, self.type_2D_series_rename_dict)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_3D:
            return self.type_process(dicom_ds, self.type_3D_series_rename_mapping, self.type_3D_series_rename_dict)
        return NullEnum.NULL


class ASLProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for ASL (Arterial Spin Labeling) series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and ASLSEQSeriesRenameEnum values.
    type_3D_series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns for 3D acquisitions.
    type_3D_series_rename_dict : dict
        Mapping of ASLSEQSeriesRenameEnum values to sets of attributes for 3D acquisitions.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, only 3D.
    series_group_fn_list : list
        List containing functions for extracting series group information.
    """

    # multi-Delay ASL SEQ  TDelay_SEQ
    # ASLSEQSeriesRenameEnum.ASLSEQATT: re.compile('([(]Transit delay[)] multi-Delay ASL SEQ)', re.IGNORECASE),
    # ASLSEQSeriesRenameEnum.ASLSEQATT: re.compile('([(]Transit delay[)]) (multi-Delay ASL SEQ|TDelay_SEQ)',
    #                                              re.IGNORECASE),
    # ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR: re.compile('([(]Color Transit delay[)] multi-Delay ASL SEQ)',
    #                                                    re.IGNORECASE),
    # ASLSEQSeriesRenameEnum.ASLSEQCBF: re.compile('([(]Transit corrected CBF[)] multi-Delay ASL SEQ)',
    #                                              re.IGNORECASE),
    # ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR: re.compile('([(]Color Transit corrected CBF[)] multi-Delay ASL SEQ)',
    #                                                    re.IGNORECASE),
    # ASLSEQSeriesRenameEnum.ASLSEQPW: re.compile('([(]per del, mean PW, REF[)] multi-Delay ASL SEQ)', re.IGNORECASE),

    series_rename_mapping = {
        # ASLSEQSeriesRenameEnum.ASLSEQ: re.compile('(multi-Delay ASL SEQ)',re.IGNORECASE),
        # ASLSEQSeriesRenameEnum.ASLSEQATT: re.compile('([(]Transit delay[)])',re.IGNORECASE),
        # ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR: re.compile('([(]Color Transit delay[)])',re.IGNORECASE),
        # ASLSEQSeriesRenameEnum.ASLSEQCBF: re.compile('([(]Transit corrected CBF[)])',re.IGNORECASE),
        # ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR: re.compile('([(]Color Transit corrected CBF[)])',re.IGNORECASE),
        # ASLSEQSeriesRenameEnum.ASLSEQPW: re.compile('([(]per del, mean PW, REF[)])',re.IGNORECASE),
        # ASLSEQSeriesRenameEnum.ASLPROD: re.compile('.*(ASL).*',re.IGNORECASE),
        # ASLSEQSeriesRenameEnum.ASLPRODCBF: re.compile(r'.*((?<!r)CBF|Cerebral Blood Flow).*', re.IGNORECASE),
    }

    type_3D_series_rename_mapping = {
        ASLSEQSeriesRenameEnum.ASLSEQ: re.compile('(multi-Delay ASL SEQ)',re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLSEQATT: re.compile('([(]Transit delay[)])',re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR: re.compile('([(]Color Transit delay[)])',re.IGNORECASE),

        ASLSEQSeriesRenameEnum.ASLSEQCBF: re.compile('([(]Transit corrected CBF[)])',re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR: re.compile('([(]Color Transit corrected CBF[)])',
                                                           re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLSEQPW: re.compile('([(]per del, mean PW, REF[)])',re.IGNORECASE),

        ASLSEQSeriesRenameEnum.ASLPROD: re.compile('.*(ASL).*',re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLPRODCBF: re.compile(r'.*((?<!r)CBF|Cerebral Blood Flow).*', re.IGNORECASE),
    }
    type_3D_series_rename_dict = {
        ASLSEQSeriesRenameEnum.ASLSEQ: {ASLSEQSeriesRenameEnum.ASLSEQ},

        ASLSEQSeriesRenameEnum.ASLSEQATT: {ASLSEQSeriesRenameEnum.ASLSEQATT},
        ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR: {ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR},

        ASLSEQSeriesRenameEnum.ASLSEQCBF: {ASLSEQSeriesRenameEnum.ASLSEQCBF},
        ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR: {ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR},

        ASLSEQSeriesRenameEnum.ASLSEQPW: {ASLSEQSeriesRenameEnum.ASLSEQPW},

        ASLSEQSeriesRenameEnum.ASLPROD: {ASLSEQSeriesRenameEnum.ASLPROD,ASLSEQSeriesRenameEnum.ASL},
        ASLSEQSeriesRenameEnum.ASLPRODCBF: {ASLSEQSeriesRenameEnum.ASLPRODCBF,
                                            ASLSEQSeriesRenameEnum.ASL,
                                            ASLSEQSeriesRenameEnum.CBF},
    }
    type_null_series_rename_mapping = {
        ASLSEQSeriesRenameEnum.ASLPRODCBF_COLOR: re.compile(r'.*((?<!r)CBF|SCREENSAVE).*', re.IGNORECASE),
    }
    type_null_series_rename_dict = {
        ASLSEQSeriesRenameEnum.ASLPRODCBF_COLOR: {ASLSEQSeriesRenameEnum.ASLPRODCBF_COLOR,
                                                  ASLSEQSeriesRenameEnum.ASL,
                                                  ASLSEQSeriesRenameEnum.CBF,
                                                  ASLSEQSeriesRenameEnum.COLOR},
    }

    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,NullEnum.NULL)
    series_group_fn_list = []

    @classmethod
    def get_series_group_fn_list(cls):
        """Get the list of series group functions.

        Returns:
        list: List of functions for extracting series group information.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_conversion_type)
            cls.series_group_fn_list.append(cls.get_asl)
            cls.series_group_fn_list.append(cls.get_cbf)
        return cls.series_group_fn_list


    @classmethod
    def get_asl(cls, dicom_ds: FileDataset, ):
        # (0019,109C)	Unknown  Tag &  Data	ASL
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            # SIGNA Voyager (0019,109C)	Unknown  Tag &  Data	ASL
            if str(pulse_sequence_name.value).lower() == ASLSEQSeriesRenameEnum.ASL.value.lower():
                return ASLSEQSeriesRenameEnum.ASL
        else:
            # MR360 (0008,103E)	Series Description	SCREENSAVE
            # (0043,10A4)	Unknown  Tag &  Data	3D pulsed continuous ASL technique
            ASL_technique = dicom_ds.get((0x43, 0x10A4))
            if ASL_technique:
                if ASLSEQSeriesRenameEnum.ASL.value.lower() in str(ASL_technique.value).lower():
                    return ASLSEQSeriesRenameEnum.ASL
        return NullEnum.NULL

    @classmethod
    def get_cbf(cls, dicom_ds: FileDataset, ):
        functional_processing_name = dicom_ds.get((0x51, 0x1002))
        if functional_processing_name:
            if str(functional_processing_name.value).lower() == ASLSEQSeriesRenameEnum.CBF.value.lower():
                return ASLSEQSeriesRenameEnum.CBF
            elif str(functional_processing_name.value).lower() == ASLSEQSeriesRenameEnum.Cerebral_Blood_Flow.value.lower():
                return ASLSEQSeriesRenameEnum.CBF
        else:
            # (0008,1090)	Manufacturer Model Name	MR360
            manufacturer_model_name = dicom_ds.get((0x08, 0x1090))
            image_type_tag = dicom_ds.get((0x08, 0x08))
            # (0008,0008)	Image Type	DERIVED\SECONDARY\SCREEN SAVE
            # ASLSEQSeriesRenameEnum.ASLPRODCBF_COLOR get CBF
            if 'MR360' in manufacturer_model_name.value and image_type_tag[-1] == 'SCREEN SAVE':
                    return ASLSEQSeriesRenameEnum.CBF
        return NullEnum.NULL

    @classmethod
    def get_conversion_type(cls, dicom_ds: FileDataset, ):
        # (0008,0064)	Conversion Type	WSD
        conversion_type = dicom_ds.get((0x08, 0x64))
        if conversion_type:
            return ASLSEQSeriesRenameEnum.COLOR
        return NullEnum.NULL

    def type_process(self, dicom_ds: FileDataset, type_series_rename_mapping, type_series_rename_dict) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for ASL series renaming based on acquisition type.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.
        type_series_rename_mapping (dict): Mapping of series descriptions to corresponding regex patterns for the acquisition type.
        type_series_rename_dict (dict): Mapping of ASLSEQSeriesRenameEnum values to sets of attributes for the acquisition type.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in type_series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                series_group_set.add(series_rename_enum)
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                for series_rename_enum, series_rename_group_set in type_series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum
        return NullEnum.NULL

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for ASL series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_3D:
            return self.type_process(dicom_ds, self.type_3D_series_rename_mapping, self.type_3D_series_rename_dict)
        if mr_acquisition_type_enum == NullEnum.NULL:
            return self.type_process(dicom_ds, self.type_null_series_rename_mapping,self.type_null_series_rename_dict)
        return NullEnum.NULL


class DSCProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for DSC (Dynamic Susceptibility Contrast) series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and DSCSeriesRenameEnum values.
    type_2D_series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns for 2D acquisitions.
    type_null_series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns for null acquisitions.
    type_null_series_rename_dict : dict
        Mapping of DSCSeriesRenameEnum values to sets of attributes for null acquisitions.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition types and NullEnum, in this case, 2D and null.
    series_group_fn_list : list
        List containing functions for extracting series group information.
    """

    series_rename_mapping = {
        DSCSeriesRenameEnum.DSC: re.compile('.*(AUTOPWI|Perfusion).*', re.IGNORECASE),
        DSCSeriesRenameEnum.rCBF: re.compile('.*(CBF).*', re.IGNORECASE),
        DSCSeriesRenameEnum.rCBV: re.compile('.*(CBV).*', re.IGNORECASE),
        DSCSeriesRenameEnum.MTT: re.compile('.*(MTT).*', re.IGNORECASE),
    }
    type_2D_series_rename_mapping = {
        DSCSeriesRenameEnum.DSC: re.compile('.*(AUTOPWI|Perfusion).*', re.IGNORECASE),
    }
    type_null_series_rename_mapping = {
        DSCSeriesRenameEnum.rCBF: re.compile('.*(CBF).*', re.IGNORECASE),
        DSCSeriesRenameEnum.rCBV: re.compile('.*(CBV).*', re.IGNORECASE),
        DSCSeriesRenameEnum.MTT: re.compile('.*(MTT).*', re.IGNORECASE),
    }
    type_null_series_rename_dict = {
        DSCSeriesRenameEnum.rCBF: {DSCSeriesRenameEnum.rCBF},
        DSCSeriesRenameEnum.rCBV: {DSCSeriesRenameEnum.rCBV},
        DSCSeriesRenameEnum.MTT: {DSCSeriesRenameEnum.MTT},
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_2D, NullEnum.NULL,)
    series_group_fn_list = []

    @classmethod
    def get_functional_processing_name(cls, dicom_ds: FileDataset):
        """Get the functional processing name from the DICOM dataset.
           DSCSeriesRenameEnum.rCBF、DSCSeriesRenameEnum.rCBV、DSCSeriesRenameEnum.MTT
        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The functional processing name enumeration or NullEnum.NULL if not found.
        """
        functional_processing_name = dicom_ds.get((0x51, 0x1002))
        if functional_processing_name:
            for dsc_series_rename_enum in DSCSeriesRenameEnum.to_list():
                if dsc_series_rename_enum.name == functional_processing_name.value:
                    return dsc_series_rename_enum
        return NullEnum.NULL

    @classmethod
    def get_series_group_fn_list(cls):
        """Get the list of series group functions.

        Returns:
        list: List of functions for extracting series group information.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_functional_processing_name)
        return cls.series_group_fn_list

    def type_2D_process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for 2D DSC series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in self.type_2D_series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                return series_rename_enum
        return NullEnum.NULL

    def type_null_process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for rCBF rCBV MTT series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in self.type_null_series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                for series_rename_enum, series_rename_group_set in self.type_null_series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum
        return NullEnum.NULL

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for DSC series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
            return self.type_2D_process(dicom_ds)
        if mr_acquisition_type_enum == NullEnum.NULL:
            return self.type_null_process(dicom_ds)
        return NullEnum.NULL


class CVRProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for CVR (Cerebrovascular Reactivity) series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition type 2D and NullEnum.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.CVR: re.compile('.*(CVR).*$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,)
    type_2D_series_rename_dict = {
        MRSeriesRenameEnum.CVR2000_EAR: {MRSeriesRenameEnum.CVR, RepetitionTimeEnum.TR2000, BodyPartEnum.EAR},
        MRSeriesRenameEnum.CVR2000_EYE: {MRSeriesRenameEnum.CVR, RepetitionTimeEnum.TR2000, BodyPartEnum.EYE},
        MRSeriesRenameEnum.CVR2000: {MRSeriesRenameEnum.CVR, RepetitionTimeEnum.TR2000},
        MRSeriesRenameEnum.CVR1000: {MRSeriesRenameEnum.CVR, RepetitionTimeEnum.TR1000},
        MRSeriesRenameEnum.CVR: {MRSeriesRenameEnum.CVR},
    }
    series_group_fn_list = []

    # MRSeriesRenameEnum.CVR2000_EAR: re.compile('.*(CVR).*(2000).*(ear).*$', re.IGNORECASE),
    # MRSeriesRenameEnum.CVR2000_EYE: re.compile('.*(CVR).*(2000).*(eye).*$', re.IGNORECASE),
    # MRSeriesRenameEnum.CVR2000: re.compile('.*(CVR).*(2000).*$', re.IGNORECASE),
    # MRSeriesRenameEnum.CVR1000: re.compile('.*(CVR).*(1000).*$', re.IGNORECASE),
    @classmethod
    def get_description_eye(cls, dicom_ds: FileDataset, ):
        series_description = dicom_ds.get((0x08, 0x103E))
        if series_description:
            if BodyPartEnum.EYE.value.lower() in str(series_description.value).lower():
                return BodyPartEnum.EYE
        return NullEnum.NULL

    @classmethod
    def get_description_ear(cls, dicom_ds: FileDataset, ):
        series_description = dicom_ds.get((0x08, 0x103E))
        if series_description:
            if BodyPartEnum.EAR.value.lower() in str(series_description.value).lower():
                return BodyPartEnum.EAR
        return NullEnum.NULL

    @classmethod
    def get_repetition_time(cls, dicom_ds: FileDataset, ):
        """Get series information for repetition time sequences from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, T2SeriesRenameEnum, SeriesEnum]: repetition time series information.
        """
        tr = str(dicom_ds[0x18, 0x80].value)
        if tr == RepetitionTimeEnum.TR2000.value:
            return RepetitionTimeEnum.TR2000
        if tr == RepetitionTimeEnum.TR1000.value:
            return RepetitionTimeEnum.TR1000
        return NullEnum.NULL

    @classmethod
    def get_series_group_fn_list(cls):
        """Get the list of series group functions.

        Returns:
        list: List of functions for extracting series group information.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_repetition_time)
            cls.series_group_fn_list.append(cls.get_description_ear)
            cls.series_group_fn_list.append(cls.get_description_eye)
        return cls.series_group_fn_list

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for CVR series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in self.series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                series_group_set.add(series_rename_enum)
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                for series_rename_enum, series_rename_group_set in self.type_2D_series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum
                return series_rename_enum
        return NullEnum.NULL
        # series_description = dicom_ds.get((0x08, 0x103E))
        # for series_rename_enum, series_pattern in self.series_rename_mapping.items():
        #     match_result = series_pattern.match(series_description.value)
        #     if match_result:
        #         return series_rename_enum
        # return NullEnum.NULL


class RestingProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for renaming resting series based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition type 2D and NullEnum.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.RESTING: re.compile('.*(resting).*$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,)

    type_2D_series_rename_dict = {
        MRSeriesRenameEnum.RESTING2000: {MRSeriesRenameEnum.RESTING, RepetitionTimeEnum.TR2000},
        MRSeriesRenameEnum.RESTING: {MRSeriesRenameEnum.RESTING},
    }
    series_group_fn_list = []

    @classmethod
    def get_repetition_time(cls, dicom_ds: FileDataset, ):
        """Get series information for repetition time sequences from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, T2SeriesRenameEnum, SeriesEnum]: repetition time series information.
        """
        tr = str(dicom_ds[0x18, 0x80].value)
        if tr == RepetitionTimeEnum.TR2000.value:
            return RepetitionTimeEnum.TR2000
        return NullEnum.NULL

    @classmethod
    def get_series_group_fn_list(cls):
        """Get the list of series group functions.

        Returns:
        list: List of functions for extracting series group information.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_repetition_time)
        return cls.series_group_fn_list

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for resting series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in self.series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                series_group_set.add(series_rename_enum)
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                for series_rename_enum, series_rename_group_set in self.type_2D_series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum
                return series_rename_enum
        return NullEnum.NULL


class DTIProcessingStrategy(MRRenameSeriesProcessingStrategy):
    """A processing strategy for DTI (Diffusion Tensor Imaging) series renaming based on DICOM attributes.

    Attributes:
    series_rename_mapping : dict
        Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
    series_rename_dict : dict
        Mapping of series enumeration to corresponding DTISeriesEnum values.
    mr_acquisition_type : tuple
        Tuple containing MR acquisition type 2D and NullEnum.
    series_group_fn_list : list
        List containing functions for extracting series group information.
    """

    series_rename_mapping = {
        MRSeriesRenameEnum.DTI32D: re.compile('.*(DTI).*', re.IGNORECASE),
        MRSeriesRenameEnum.DTI64D: re.compile('.*(DTI).*', re.IGNORECASE),
    }
    series_rename_dict = {
        MRSeriesRenameEnum.DTI32D: {DTISeriesEnum.DTI32D},
        MRSeriesRenameEnum.DTI64D: {DTISeriesEnum.DTI64D}
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,)
    series_group_fn_list = []

    @classmethod
    def get_series_group_fn_list(cls):
        """Get the list of series group functions.

        Returns:
        list: List of functions for extracting series group information.
        """
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_dti_diffusion)
        return cls.series_group_fn_list

    @classmethod
    def get_dti_diffusion(cls, dicom_ds: FileDataset):
        """Get the DTI diffusion information from the DICOM dataset.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, DTISeriesEnum]: The DTI diffusion enumeration or NullEnum.NULL if not found.
        """
        dti_diffusion = dicom_ds.get((0x19, 0x10E0))
        if dti_diffusion:
            for dti_diffusion_rename_enum in DTISeriesEnum.to_list():
                if dti_diffusion_rename_enum.value == dti_diffusion.value:
                    return dti_diffusion_rename_enum
        return NullEnum.NULL

    def type_process(self, dicom_ds: FileDataset, type_series_rename_mapping, type_series_rename_dict) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for DTI series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.
        type_series_rename_mapping : dict
            Mapping of series descriptions to corresponding regex patterns and MRSeriesRenameEnum values.
        type_series_rename_dict : dict
            Mapping of series enumeration to corresponding DTISeriesEnum values.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in type_series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                series_group_set = set()
                for series_group_fn in self.get_series_group_fn_list():
                    item_enum = series_group_fn(dicom_ds=dicom_ds)
                    if item_enum is not NullEnum.NULL:
                        if isinstance(item_enum, tuple):
                            series_group_set.update(item_enum)
                        else:
                            series_group_set.add(item_enum)
                for series_rename_enum, series_rename_group_set in type_series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum

        return NullEnum.NULL

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process the DICOM dataset for DTI series renaming.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]: The renamed series enumeration or NullEnum.NULL if no match is found.
        """
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
            return self.type_process(dicom_ds, self.series_rename_mapping, self.series_rename_dict)

        return NullEnum.NULL


class ConvertManager:
    """A manager class for DICOM file conversion and renaming.

    Attributes:
    modality_processing_strategy : ModalityProcessingStrategy
        An instance of the ModalityProcessingStrategy for determining the modality.
    mr_acquisition_type_processing_strategy : MRAcquisitionTypeProcessingStrategy
        An instance of the MRAcquisitionTypeProcessingStrategy for determining MR acquisition type.
    processing_strategy_list : list
        List of MRRenameSeriesProcessingStrategy instances for various series renaming strategies.
    _input_path : pathlib.Path
        The input path containing the DICOM files.
    output_path : pathlib.Path
        The output path where the renamed DICOM files will be saved.
    """

    modality_processing_strategy: ModalityProcessingStrategy = ModalityProcessingStrategy()
    mr_acquisition_type_processing_strategy: MRAcquisitionTypeProcessingStrategy = MRAcquisitionTypeProcessingStrategy()
    processing_strategy_list: List[MRRenameSeriesProcessingStrategy] = [DwiProcessingStrategy(),
                                                                        ADCProcessingStrategy(),
                                                                        EADCProcessingStrategy(),
                                                                        SWANProcessingStrategy(),
                                                                        ESWANProcessingStrategy(),
                                                                        MRABrainProcessingStrategy(),
                                                                        MRANeckProcessingStrategy(),
                                                                        MRAVRBrainProcessingStrategy(),
                                                                        MRAVRNeckProcessingStrategy(),
                                                                        T1ProcessingStrategy(),
                                                                        T2ProcessingStrategy(),
                                                                        ASLProcessingStrategy(),
                                                                        DSCProcessingStrategy(),
                                                                        RestingProcessingStrategy(),
                                                                        CVRProcessingStrategy(),
                                                                        DTIProcessingStrategy()]

    def __init__(self, input_path: Union[str, pathlib.Path], output_path: Union[str, pathlib.Path], *args, **kwargs):
        """Initialize the ConvertManager.

        Parameters:
        input_path (Union[str, pathlib.Path]): The input path containing the DICOM files.
        output_path (Union[str, pathlib.Path]): The output path where the renamed DICOM files will be saved.
        """

        self._input_path = pathlib.Path(input_path)
        self.output_path = pathlib.Path(output_path)

    @staticmethod
    def get_output_study(dicom_ds: FileDataset, output_path: pathlib.Path):
        """Get the output study folder based on DICOM attributes.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.
        output_path (pathlib.Path): The output path.

        Returns:
        pathlib.Path: The output study folder path.
        """
        study_folder_name = ConvertManager.get_study_folder_name(dicom_ds=dicom_ds)
        if output_path.name == study_folder_name:
            return output_path
        if study_folder_name:
            output_study = output_path.joinpath(study_folder_name)
            return output_study
        return None

    @staticmethod
    def get_study_folder_name(dicom_ds: FileDataset):
        """Generate the study folder name based on DICOM attributes.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        str: The generated study folder name.
        """
        modality = dicom_ds[0x08, 0x60].value
        patient_id = dicom_ds[0x10, 0x20].value
        accession_number = dicom_ds[0x08, 0x50].value
        # (0008,0020)	Study Date	20160722
        study_date = dicom_ds.get((0x08, 0x20), None)
        if study_date is None:
            return None
        else:
            study_date = study_date.value
        return f'{patient_id}_{study_date}_{modality}_{accession_number}'

    def rename_dicom_path(self, dicom_ds: FileDataset):
        """Rename the DICOM series based on processing strategies.

        Parameters:
        dicom_ds (FileDataset): The DICOM dataset.

        Returns:
        str: The renamed series name.
        """
        modality_enum = self.modality_processing_strategy.process(dicom_ds=dicom_ds)
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        for processing_strategy in self.processing_strategy_list:
            if modality_enum == processing_strategy.modality:
                for mr_acquisition_type in processing_strategy.mr_acquisition_type:
                    if mr_acquisition_type_enum == mr_acquisition_type:
                        series_enum = processing_strategy.process(dicom_ds=dicom_ds)
                        if series_enum is not NullEnum.NULL:
                            return series_enum.value
        return ''

    def rename_process(self, instances, *args, **kwargs):
        """Process the renaming of DICOM files in the specified instances list.

        Parameters:
        instances_list (list): List of DICOM file instances.
        """
        try:
            dicom_ds = dcmread(str(instances), stop_before_pixels=True)
            output_study = self.get_output_study(dicom_ds=dicom_ds, output_path=self.output_path)
            if output_study:
                rename_series = self.rename_dicom_path(dicom_ds=dicom_ds)
                if len(rename_series) > 0:
                    os.makedirs(output_study, exist_ok=True)
                    output_study_series = output_study.joinpath(rename_series)
                    if output_study_series.exists():
                        pass
                    else:
                        output_study_series.mkdir(exist_ok=True)
                    output_study_instances = output_study_series.joinpath(instances.name)
                    if output_study_instances.exists():
                        pass
                    else:
                        # print(output_study_instances)
                        shutil.copyfile(instances, output_study_instances)
        except (pydicom.errors.InvalidDicomError, pydicom.errors.BytesLengthException):
            print(f'except {instances}')
        except:
            print(traceback.format_exc())
            print('Unknown except')

    def run(self, executor: Union[ThreadPoolExecutor, None] = None):
        """Run the DICOM file conversion and renaming process.

        Parameters:
        executor (Union[ThreadPoolExecutor, None]): Thread pool executor for parallel processing.
        """
        is_dir_flag = all(list(map(lambda x: x.is_dir(), self.input_path.iterdir())))
        if is_dir_flag:
            # for sub_dir in tqdm(list(self.input_path.iterdir()),desc=f'sub dir'):
            for sub_dir in self.input_path.iterdir():
                instances_list = list(sub_dir.rglob('*.dcm'))
                if executor:
                    # executor.map(self.rename_process, (instances_list,))
                    results = list(tqdm(executor.map(self.rename_process, instances_list), total=len(instances_list),
                                        desc=f'dir:{sub_dir.name}', ))
                else:
                    for instances in tqdm(instances_list, total=len(instances_list),
                                          desc=f'dir:{sub_dir.name}', ):
                        self.rename_process(instances=instances)
        else:
            instances_list = list(self.input_path.rglob('*.dcm'))
            if executor:
                # executor.map(self.rename_process, (instances_list,))
                results = list(tqdm(executor.map(self.rename_process, instances_list),
                                    total=len(instances_list), desc=f'dir:{self.input_path.name}'),
                               )
            else:
                for instances in tqdm(instances_list):
                    self.rename_process(instances=instances)

    @property
    def input_path(self):
        """Getter for the input path.

        Returns:
        pathlib.Path: The input path.
        """
        return self._input_path

    @input_path.setter
    def input_path(self, value: str):
        """Setter for the input path.

        Parameters:
        value (str): The input path.
        """
        self._input_path = pathlib.Path(value)
