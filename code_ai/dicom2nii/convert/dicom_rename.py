import argparse
import os
import pathlib
import re
import shutil

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

from pydicom import dcmread, FileDataset

from base import *
from config import *


# 1. Raw Dicom Folder -> Rename Dicom Folder
# 2. Rename Dicom Folder -> Nifti
# 3. DateBase


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

    series_rename_mapping = {
        0: MRSeriesRenameEnum.DWI0,
        1000: MRSeriesRenameEnum.DWI1000,
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,)
    pattern = re.compile('.*(DWI|AUTODIFF).*', re.IGNORECASE)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        """Process a DICOM dataset for DWI series renaming.

        Parameters:
        dicom_ds : FileDataset
            The DICOM dataset to process.

        Returns:
        Union[BaseEnum, MRSeriesRenameEnum]
            The renamed series enumeration or NullEnum.NULL if no match is found.
        """

        # Check if modality is MR and acquisition type is 2D
        series_description = dicom_ds.get((0x08, 0x103E))
        match_result = self.pattern.match(series_description.value)

        if match_result:
            dicom_tag = dicom_ds.get((0x43, 0x1039))
            b_values = dicom_tag[0]

            # Check if DICOM tag exists
            if dicom_tag:
                # Iterate through the mapping and return corresponding MRSeriesRenameEnum
                for key, series_rename_enum in self.series_rename_mapping.items():
                    if b_values == key:
                        return series_rename_enum

        # If no match or mapping is found, return NullEnum.NULL
        return NullEnum.NULL


class DwiProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        0: MRSeriesRenameEnum.DWI0,
        1000: MRSeriesRenameEnum.DWI1000,
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,)
    pattern = re.compile('.*(DWI|AUTODIFF).*', re.IGNORECASE)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        #  modality == MR
        # # mr_acquisition_type == 2D
        series_description = dicom_ds.get((0x08, 0x103E))
        match_result = self.pattern.match(series_description.value)
        if match_result:
            dicom_tag = dicom_ds.get((0x43, 0x1039))
            b_values = dicom_tag[0]
            # DWI0、DWI1000
            if dicom_tag:
                for key, series_rename_enum in self.series_rename_mapping.items():
                    if b_values == key:
                        return series_rename_enum
        return NullEnum.NULL


class ADCProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        MRSeriesRenameEnum.ADC: re.compile('.*(?<!e)(ADC|Apparent Diffusion Coefficient).*', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,
                                                                          NullEnum.NULL)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        # (0008,0008)	Image Type	DERIVED\SECONDARY\COMBINED
        image_type_tag = dicom_ds.get((0x08, 0x08))
        # (0008,0013)	Instance Creation Time	165409
        instance_creation_time = dicom_ds.get((0x08, 0x13))
        # (0008,103E)	Series Description	ADC
        series_description = dicom_ds.get((0x08, 0x103E))
        # 正則表達式中，(?<!e) 表示負向回顧斷言，確保匹配的字串前面不是 "e"。
        # if image_type_tag[0] == 'DERIVED' and \
        #         image_type_tag[1] == 'SECONDARY' and \
        #         image_type_tag[2] == 'COMBINED' and \
        #         instance_creation_time is not None:
        if image_type_tag[0] == 'DERIVED' and instance_creation_time is not None:
            for key, pattern in self.series_rename_mapping.items():
                match_result = pattern.match(series_description.value)
                if match_result:
                    return key
        return NullEnum.NULL


class EADCProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        MRSeriesRenameEnum.eADC: re.compile('.*(eADC).*', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,
                                                                          NullEnum.NULL)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        series_description = dicom_ds.get((0x08, 0x103E))
        for key, pattern in self.series_rename_mapping.items():
            match_result = pattern.match(series_description.value)
            if match_result:
                return key
        return NullEnum.NULL


class SWANProcessingStrategy(MRRenameSeriesProcessingStrategy):
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
    def get_mIP(cls, dicom_ds: FileDataset):
        image_type = dicom_ds.get((0x08, 0x08))
        instance_creation_time = dicom_ds.get((0x08, 0x13))
        if image_type and instance_creation_time is not None:
            if image_type.value[-1] == 'MIN IP':
                return SeriesEnum.mIP
        return NullEnum.NULL

    @classmethod
    def get_mag(cls, dicom_ds: FileDataset):
        image_type = dicom_ds.get((0x43, 0x102F))
        if image_type:
            if image_type.value == SeriesEnum.SWANPHASE.value:
                return SeriesEnum.SWANPHASE

        return NullEnum.NULL

    @classmethod
    def get_swan(cls, dicom_ds: FileDataset):
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            if str(pulse_sequence_name.value).lower() == SeriesEnum.SWAN.value.lower():
                return SeriesEnum.SWAN
        return NullEnum.NULL

    @classmethod
    def get_series_group_fn_list(cls):
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_swan)
            cls.series_group_fn_list.append(cls.get_mIP)
            cls.series_group_fn_list.append(cls.get_mag)
        return cls.series_group_fn_list

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in self.series_rename_mapping.items():
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
                for series_rename_enum, series_rename_group_set in self.series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum
        return NullEnum.NULL


class ESWANProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        MRSeriesRenameEnum.eSWAN: re.compile('.*(SWAN).*', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)
    series_group_fn_list = []

    series_rename_dict = {
        MRSeriesRenameEnum.eSWAN: {SeriesEnum.eSWAN},
        MRSeriesRenameEnum.eSWANmIP: {SeriesEnum.eSWAN, SeriesEnum.mIP},
    }

    @classmethod
    def get_mIP(cls, dicom_ds: FileDataset):
        image_type = dicom_ds.get((0x08, 0x08))
        instance_creation_time = dicom_ds.get((0x08, 0x13))
        if image_type and instance_creation_time is not None:
            if image_type.value[-1] == 'MIN IP':
                return SeriesEnum.mIP

        #     MRSeriesRenameEnum.eSWANmag
        return NullEnum.NULL

    @classmethod
    def get_eswan(cls, dicom_ds: FileDataset):
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            if str(pulse_sequence_name.value).lower() == SeriesEnum.eSWAN.value.lower():
                return SeriesEnum.eSWAN
        return NullEnum.NULL

    @classmethod
    def get_series_group_fn_list(cls):
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_eswan)
            cls.series_group_fn_list.append(cls.get_mIP)
        return cls.series_group_fn_list

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in self.series_rename_mapping.items():
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
                for series_rename_enum, series_rename_group_set in self.series_rename_dict.items():
                    if series_group_set == series_rename_group_set:
                        return series_rename_enum
        return NullEnum.NULL


class MRABrainProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        MRSeriesRenameEnum.MRA_BRAIN: re.compile('.+(TOF)(((?!Neck).)*)$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        #  modality == MR
        # # mr_acquisition_type == 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        pattern = self.series_rename_mapping[MRSeriesRenameEnum.MRA_BRAIN]
        match_result = pattern.match(series_description.value)
        if match_result:
            image_type_tag = dicom_ds.get((0x08, 0x08))
            if image_type_tag[0] == 'ORIGINAL':
                return MRSeriesRenameEnum.MRA_BRAIN
        return NullEnum.NULL


class MRANeckProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        MRSeriesRenameEnum.MRA_NECK: re.compile('.*(TOF).*((Neck+).*)$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        #  modality == MR
        # # mr_acquisition_type == 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        pattern = self.series_rename_mapping[MRSeriesRenameEnum.MRA_NECK]
        match_result = pattern.match(series_description.value)
        if match_result:
            image_type_tag = dicom_ds.get((0x08, 0x08))
            if image_type_tag[0] == 'ORIGINAL':
                return MRSeriesRenameEnum.MRA_NECK
        return NullEnum.NULL


class MRAVRBrainProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        MRSeriesRenameEnum.MRAVR_BRAIN: re.compile('((?!TOF|Neck).)*(MRA)((?!Neck).)*$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        #  modality == MR
        # # mr_acquisition_type == 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        pattern = self.series_rename_mapping[MRSeriesRenameEnum.MRAVR_BRAIN]
        match_result = pattern.match(series_description.value)
        if match_result:
            return MRSeriesRenameEnum.MRAVR_BRAIN
        return NullEnum.NULL


class MRAVRNeckProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        MRSeriesRenameEnum.MRAVR_NECK: re.compile('((?!TOF).)*(Neck.*MRA)|(MRA.*Neck).*$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        #  modality == MR
        # # mr_acquisition_type == 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        pattern = self.series_rename_mapping[MRSeriesRenameEnum.MRAVR_NECK]
        match_result = pattern.match(series_description.value)
        if match_result:
            return MRSeriesRenameEnum.MRAVR_NECK
        return NullEnum.NULL


class T1ProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        T1SeriesRenameEnum.T1: re.compile('.*(T1).*', re.IGNORECASE),
        SeriesEnum.FLAIR: re.compile('(FLAIR)', re.IGNORECASE),
        SeriesEnum.CUBE: re.compile('.*(CUBE).*', re.IGNORECASE),
        SeriesEnum.BRAVO: re.compile('.*(BRAVO).*', re.IGNORECASE),
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
        T1SeriesRenameEnum.T1BRAVOCE_AXI: {T1SeriesRenameEnum.T1, SeriesEnum.BRAVO, ImageOrientationEnum.AXI,
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
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_image_orientation)
            cls.series_group_fn_list.append(cls.get_contrast)
            cls.series_group_fn_list.append(cls.get_flair)
            cls.series_group_fn_list.append(cls.get_cube)
            cls.series_group_fn_list.append(cls.get_bravo)
        return cls.series_group_fn_list

    @classmethod
    def get_image_orientation(cls, dicom_ds: FileDataset):
        image_orientation = cls.image_orientation_processing_strategy.process(dicom_ds=dicom_ds)
        return image_orientation

    @classmethod
    def get_contrast(cls, dicom_ds: FileDataset):
        contrast = cls.contrast_processing_strategy.process(dicom_ds=dicom_ds)
        return contrast

    @classmethod
    def get_flair(cls, dicom_ds: FileDataset, ):
        tr = float(dicom_ds[0x18, 0x80].value)
        te = float(dicom_ds[0x18, 0x81].value)
        if 800 <= tr <= 3000 and te <= 30:
            return T1SeriesRenameEnum.T1, SeriesEnum.FLAIR
        if tr <= 800 and te <= 30:
            return T1SeriesRenameEnum.T1
        return NullEnum.NULL

    @classmethod
    def get_cube(cls, dicom_ds: FileDataset, ):
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            if str(pulse_sequence_name.value).lower() == SeriesEnum.CUBE.value.lower():
                return SeriesEnum.CUBE
        return NullEnum.NULL

    @classmethod
    def get_bravo(cls, dicom_ds: FileDataset, ):
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            if str(pulse_sequence_name.value).lower() == SeriesEnum.BRAVO.value.lower():
                return T1SeriesRenameEnum.T1, SeriesEnum.BRAVO
            elif str(pulse_sequence_name.value).lower() == SeriesEnum.FSPGR.value.lower():
                return T1SeriesRenameEnum.T1, SeriesEnum.BRAVO
        return NullEnum.NULL

    def type_process(self, dicom_ds: FileDataset, type_series_rename_mapping, type_series_rename_dict) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
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
                        # 3D (BRAVO, CUBE 會有一組"原始檔 最細切"，放射師會再重組成AXI, SAG, COR)

        return NullEnum.NULL

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        #  modality == MR
        # # mr_acquisition_type == 3D or 2D
        # mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        #
        # if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
        #     return self.type_2d_process(dicom_ds)
        # if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_3D:
        #     return self.type_3d_process(dicom_ds)
        # return NullEnum.NULL
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
            return self.type_process(dicom_ds, self.type_2D_series_rename_mapping, self.type_2D_series_rename_dict)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_3D:
            return self.type_process(dicom_ds, self.type_3D_series_rename_mapping, self.type_3D_series_rename_dict)
        return NullEnum.NULL


class T2ProcessingStrategy(MRRenameSeriesProcessingStrategy):
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
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_image_orientation)
            cls.series_group_fn_list.append(cls.get_contrast)
            cls.series_group_fn_list.append(cls.get_flair)
            cls.series_group_fn_list.append(cls.get_cube)
        return cls.series_group_fn_list

    @classmethod
    def get_image_orientation(cls, dicom_ds: FileDataset):
        image_orientation = cls.image_orientation_processing_strategy.process(dicom_ds=dicom_ds)
        return image_orientation

    @classmethod
    def get_contrast(cls, dicom_ds: FileDataset):
        contrast = cls.contrast_processing_strategy.process(dicom_ds=dicom_ds)
        return contrast

    @classmethod
    def get_flair(cls, dicom_ds: FileDataset, ):
        tr = float(dicom_ds[0x18, 0x80].value)
        te = float(dicom_ds[0x18, 0x81].value)
        if 5990 <= tr <= 10000 and te >= 80:
            return T2SeriesRenameEnum.T2, SeriesEnum.FLAIR
        if 1000 <= tr < 5990 and te >= 80:
            return T2SeriesRenameEnum.T2
        return NullEnum.NULL

    @classmethod
    def get_cube(cls, dicom_ds: FileDataset, ):
        pulse_sequence_name = dicom_ds.get((0x19, 0x109c))
        if pulse_sequence_name:
            if SeriesEnum.CUBE.value.lower() in str(pulse_sequence_name.value).lower():
                return SeriesEnum.CUBE
        return NullEnum.NULL

    def type_process(self, dicom_ds: FileDataset, type_series_rename_mapping, type_series_rename_dict) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
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
        #  modality == MR
        # # mr_acquisition_type == 3D or 2D
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)

        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
            return self.type_process(dicom_ds, self.type_2D_series_rename_mapping, self.type_2D_series_rename_dict)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_3D:
            return self.type_process(dicom_ds, self.type_3D_series_rename_mapping, self.type_3D_series_rename_dict)
        return NullEnum.NULL


class ASLProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        ASLSEQSeriesRenameEnum.ASLSEQ: re.compile('(multi-Delay ASL SEQ)', re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLPROD: re.compile('(3D ASL [(]non-contrast[)])', re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLSEQATT: re.compile('([(]Transit delay[)] multi-Delay ASL SEQ)', re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR: re.compile('([(]Color Transit delay[)] multi-Delay ASL SEQ)',
                                                           re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLSEQCBF: re.compile('([(]Transit corrected CBF[)] multi-Delay ASL SEQ)',
                                                     re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR: re.compile('([(]Color Transit corrected CBF[)] multi-Delay ASL SEQ)',
                                                           re.IGNORECASE),
        ASLSEQSeriesRenameEnum.ASLSEQPW: re.compile('([(]per del, mean PW, REF[)] multi-Delay ASL SEQ)', re.IGNORECASE),

    }
    type_3D_series_rename_mapping = series_rename_mapping

    type_3D_series_rename_dict = {

    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_3D,
                                                                          )
    series_group_fn_list = []

    @classmethod
    def get_series_group_fn_list(cls):
        if len(cls.series_group_fn_list) == 0:
            pass
        return cls.series_group_fn_list

    def type_process(self, dicom_ds: FileDataset, type_series_rename_mapping, type_series_rename_dict) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in type_series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                return series_rename_enum
        return NullEnum.NULL

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        #  modality == MR
        # # mr_acquisition_type == 3D or 2D
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_3D:
            return self.type_process(dicom_ds, self.type_3D_series_rename_mapping, self.type_3D_series_rename_dict)
        return NullEnum.NULL


class DSCProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        DSCSeriesRenameEnum.DSC: re.compile('.*(AUTOPWI).*', re.IGNORECASE),
        DSCSeriesRenameEnum.rCBF: re.compile('.*(CBF).*', re.IGNORECASE),
        DSCSeriesRenameEnum.rCBV: re.compile('.*(CBV).*', re.IGNORECASE),
        DSCSeriesRenameEnum.MTT: re.compile('.*(MTT).*', re.IGNORECASE),
    }
    type_2D_series_rename_mapping = {
        DSCSeriesRenameEnum.DSC: re.compile('.*(AUTOPWI).*', re.IGNORECASE),
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
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,
                                                                          NullEnum.NULL,
                                                                          )
    series_group_fn_list = []

    @classmethod
    def get_functional_processing_name(cls, dicom_ds: FileDataset):
        functional_processing_name = dicom_ds.get((0x51, 0x1002))
        if functional_processing_name:
            for dsc_series_rename_enum in DSCSeriesRenameEnum.to_list():
                if dsc_series_rename_enum.name == functional_processing_name.value:
                    return dsc_series_rename_enum
        return NullEnum.NULL

    @classmethod
    def get_series_group_fn_list(cls):
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_functional_processing_name)
        return cls.series_group_fn_list

    def type_2D_process(self, dicom_ds: FileDataset) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in self.type_2D_series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                return series_rename_enum
        return NullEnum.NULL

    def type_null_process(self, dicom_ds: FileDataset) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
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
        #  modality == MR
        # # mr_acquisition_type == 3D or 2D
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
            return self.type_2D_process(dicom_ds, )
        if mr_acquisition_type_enum == NullEnum.NULL:
            return self.type_null_process(dicom_ds, )
        return NullEnum.NULL


class CVRProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        MRSeriesRenameEnum.CVR2000_EAR: re.compile('.*(CVR).*(2000).*(ear).*$', re.IGNORECASE),
        MRSeriesRenameEnum.CVR2000_EYE: re.compile('.*(CVR).*(2000).*(eye).*$', re.IGNORECASE),
        MRSeriesRenameEnum.CVR2000: re.compile('.*(CVR).*(2000).*$', re.IGNORECASE),
        MRSeriesRenameEnum.CVR1000: re.compile('.*(CVR).*(1000).*$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        #  modality == MR
        # # mr_acquisition_type == 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in self.series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                return series_rename_enum
        return NullEnum.NULL


class RestingProcessingStrategy(MRRenameSeriesProcessingStrategy):
    series_rename_mapping = {
        MRSeriesRenameEnum.RESTING: re.compile('.*(resting).*$', re.IGNORECASE),
    }
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = (MRAcquisitionTypeEnum.TYPE_2D,)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, MRSeriesRenameEnum]:
        #  modality == MR
        # # mr_acquisition_type == 3D
        series_description = dicom_ds.get((0x08, 0x103E))
        for series_rename_enum, series_pattern in self.series_rename_mapping.items():
            match_result = series_pattern.match(series_description.value)
            if match_result:
                return series_rename_enum
        return NullEnum.NULL


class DTIProcessingStrategy(MRRenameSeriesProcessingStrategy):
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
        if len(cls.series_group_fn_list) == 0:
            cls.series_group_fn_list.append(cls.get_dti_diffusion)
        return cls.series_group_fn_list

    @classmethod
    def get_dti_diffusion(cls, dicom_ds: FileDataset):
        dti_diffusion = dicom_ds.get((0x19, 0x10E0))
        if dti_diffusion:
            for dti_diffusion_rename_enum in DTISeriesEnum.to_list():
                if dti_diffusion_rename_enum.value == dti_diffusion.value:
                    return dti_diffusion_rename_enum
        return NullEnum.NULL

    def type_process(self, dicom_ds: FileDataset, type_series_rename_mapping, type_series_rename_dict) -> Union[
        BaseEnum, MRSeriesRenameEnum]:
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
        #  modality == MR
        # # mr_acquisition_type == 3D or 2D
        mr_acquisition_type_enum = self.mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
        if mr_acquisition_type_enum == MRAcquisitionTypeEnum.TYPE_2D:
            return self.type_process(dicom_ds, self.series_rename_mapping, self.series_rename_dict)

        return NullEnum.NULL


class ConvertManager:
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

    def __init__(self, input_path: Union[str, pathlib.Path],
                 output_path: Union[str, pathlib.Path],
                 *args, **kwargs):
        self._input_path = pathlib.Path(input_path)
        self.output_path = pathlib.Path(output_path)

    @staticmethod
    def get_output_study(dicom_ds: FileDataset, output_path: pathlib.Path):
        study_folder_name = ConvertManager.get_study_folder_name(dicom_ds=dicom_ds)
        if output_path.name == study_folder_name:
            return output_path
        if study_folder_name:
            output_study = output_path.joinpath(study_folder_name)
            return output_study
        return None

    @staticmethod
    def get_study_folder_name(dicom_ds: FileDataset):
        modality = dicom_ds[0x08, 0x60].value
        patient_id = dicom_ds[0x10, 0x20].value
        accession_number = dicom_ds[0x08, 0x50].value
        series_date = dicom_ds.get((0x08, 0x21), None)
        if series_date is None:
            return None
        else:
            series_date = series_date.value
        return f'{patient_id}_{series_date}_{modality}_{accession_number}'

    def rename_dicom_path(self, dicom_ds: FileDataset):
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

    def rename_process(self, instances_list, *args, **kwargs):
        for instances in instances_list:
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
                        continue
                    else:
                        print(output_study_instances)
                        shutil.copyfile(instances, output_study_instances)

    def run(self, executor: Union[ThreadPoolExecutor, None] = None):

        is_dir_flag = all(list(map(lambda x: x.is_dir(), self.input_path.iterdir())))
        if is_dir_flag:
            for sub_dir in self.input_path.iterdir():
                instances_list = list(sub_dir.rglob('*.dcm'))
                if executor:
                    executor.map(self.rename_process, (instances_list,))
                else:
                    self.rename_process(instances_list=instances_list)
        else:
            instances_list = list(self.input_path.rglob('*.dcm'))
            if executor:
                executor.map(self.rename_process, (instances_list,))
            else:
                self.rename_process(instances_list=instances_list)

    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self, value: str):
        self._input_path = pathlib.Path(value)


if __name__ == '__main__':
    # input_path = r'D:\00_Chen\Task08\data\raw'
    # input_path = r'D:\00_Chen\Task08\data\Study_Glymphatics\20231012_00925219_MRI perfusion Glymphatics (-C +C)'
    # input_path = r'D:\00_Chen\Task08\data\Study_Glymphatics\20220830_14951998_MRI perfusion Glymphatics (-C +C)'
    # input_path = r'D:\00_Chen\Task08\data\Study_Glymphatics\20230829_15996932_MRI perfusion Glymphatics (-C +C)'
    # input_path = r'D:\00_Chen\Task08\data\Study_Glymphatics\20230525_10671432_COVID Study'
    # output_path = r'D:\00_Chen\Task08\data\rename_dicom_processing_strategy_0103'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help="input the raw dicom folder.\r\n")
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                        help="output the rename dicom folder.\r\n"
                             "Example ： python rename_folder.py -i input_path -o output_path")
    parser.add_argument('--work', dest='work', type=int, default=4,
                        help="Thread cont .\r\n"
                             "Example ： python rename_folder.py -i input_path -o output_path --work 8")
    args = parser.parse_args()
    output_path = args.output
    input_path = args.input
    work = min(8, max(1, args.work))
    executor = ThreadPoolExecutor(max_workers=work)
    with executor:
        convert_manager = ConvertManager(input_path=input_path,
                                         output_path=output_path, )
        convert_manager.run(executor=executor)
