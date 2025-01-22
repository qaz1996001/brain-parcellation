import pathlib
import re
from enum import Enum
from abc import ABCMeta, abstractmethod, ABC
from typing import Tuple, Union, List
import numpy as np
from pydicom import FileDataset
from .config import BaseEnum, NullEnum, MRSeriesRenameEnum, ModalityEnum, MRAcquisitionTypeEnum, ImageOrientationEnum, \
    ContrastEnum


class ProcessingStrategy(metaclass=ABCMeta):
    @abstractmethod
    def process(self, dicom_ds: FileDataset, input_path: pathlib.Path, output_path: pathlib.Path):
        pass

    def __call__(self, *args, **kwargs):
        self.process(*args, **kwargs)


class SeriesProcessingStrategy(metaclass=ABCMeta):
    """
    Abstract base class for defining series processing strategies.

    Attributes
    ----------
    metaclass : ABCMeta
        Metaclass for defining abstract base classes.
    """

    @abstractmethod
    def process(self, dicom_ds: FileDataset) -> str:
        """
        Abstract method for processing a DICOM dataset.

        Parameters
        ----------
        dicom_ds : FileDataset
            The DICOM dataset to be processed.

        Returns
        -------
        str
            Result of the processing operation.
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Callable method to invoke the series processing.

        Parameters
        ----------
        args : tuple
            Positional arguments passed to the processing method.
        kwargs : dict
            Keyword arguments passed to the processing method.
        """
        self.process(*args, **kwargs)


class ModalityProcessingStrategy(SeriesProcessingStrategy):
    """
    Concrete class for processing DICOM datasets based on modality.

    Attributes
    ----------
    modality_list : list
        List of modalities from the ModalityEnum.

    Methods
    -------
    process(dicom_ds: FileDataset) -> Union[Enum, BaseEnum, ImageOrientationEnum]
        Process the DICOM dataset based on modality and return the result.
    """

    modality_list = ModalityEnum.to_list()

    def process(self, dicom_ds: FileDataset) -> Union[Enum, BaseEnum, ImageOrientationEnum]:
        """
        Process the DICOM dataset based on modality.

        Parameters
        ----------
        dicom_ds : FileDataset
            The DICOM dataset to be processed.

        Returns
        -------
        Union[Enum, BaseEnum, ImageOrientationEnum]
            The result of processing based on modality.
        """
        # Extract the modality information from the DICOM dataset
        modality = dicom_ds.get((0x08, 0x60))

        # Check if modality information is available
        if modality:
            # Iterate through the modality enums to find a match
            for modality_enum in self.modality_list:
                if modality_enum.value == modality.value:
                    return modality_enum

        # Return NullEnum.NULL if no match is found
        return NullEnum.NULL


class ImageOrientationProcessingStrategy(SeriesProcessingStrategy):
    """
    Concrete class for processing DICOM datasets based on image orientation.
        ['1', '0', '0', '0', '0', '-1'] you are dealing with Coronal plane view
        ['0', '1', '0', '0', '0', '-1'] you are dealing with Sagittal plane view
        ['1', '0', '0', '0', '1', '0'] you are dealing  with Axial plane view
    Methods
    -------
    process(dicom_ds: FileDataset) -> Union[BaseEnum, ImageOrientationEnum]
        Process the DICOM dataset based on image orientation and return the result.
    """

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, ImageOrientationEnum]:
        """
        Process the DICOM dataset based on image orientation.

        Parameters
        ----------
        dicom_ds : FileDataset
            The DICOM dataset to be processed.

        Returns
        -------
        Union[BaseEnum, ImageOrientationEnum]
            The result of processing based on image orientation.
        """
        # Extract the image orientation information from the DICOM dataset
        image_orientation = dicom_ds.get((0x20, 0x37))

        # Return NullEnum.NULL if image orientation information is not available
        if image_orientation is None:
            return NullEnum.NULL
        else:
            # Process the image orientation information
            image_orientation = image_orientation.value
            image_orientation = np.round(image_orientation)
            image_orientation = image_orientation.astype(int)
            image_orientation_abs = np.abs(image_orientation)
            index_sort = np.argsort(image_orientation_abs)

            # Determine the plane view based on the sorted indices
            if ((index_sort[-1] == 0) and (index_sort[-2] == 5)) or ((index_sort[-1] == 5) and (index_sort[-2] == 0)):
                return ImageOrientationEnum.COR
            if ((index_sort[-1] == 1) and (index_sort[-2] == 5)) or ((index_sort[-1] == 5) and (index_sort[-2] == 1)):
                return ImageOrientationEnum.SAG
            if ((index_sort[-1] == 0) and (index_sort[-2] == 4)) or ((index_sort[-1] == 4) and (index_sort[-2] == 0)):
                return ImageOrientationEnum.AXI

            # Return NullEnum.NULL if no specific plane view is detected
            return NullEnum.NULL


class ContrastProcessingStrategy(SeriesProcessingStrategy):
    """
    Concrete class for processing DICOM datasets based on modality and contrast.

    Attributes
    ----------
    modality_processing_strategy : ModalityProcessingStrategy
        An instance of ModalityProcessingStrategy used for modality processing.

    Methods
    -------
    process(dicom_ds: FileDataset) -> Union[BaseEnum, ContrastEnum]
        Process the DICOM dataset based on modality and contrast and return the result.
    """

    modality_processing_strategy: ModalityProcessingStrategy = ModalityProcessingStrategy()
    pattern = re.compile(r'(\+C|C\+)', re.IGNORECASE)

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, ContrastEnum]:
        """
        Process the DICOM dataset based on modality and contrast.

        Parameters
        ----------
        dicom_ds : FileDataset
            The DICOM dataset to be processed.

        Returns
        -------
        Union[BaseEnum, ContrastEnum]
            The result of processing based on modality and contrast.
        """
        # Process the modality information using ModalityProcessingStrategy
        modality_enum = self.modality_processing_strategy.process(dicom_ds=dicom_ds)

        # Extract the contrast information from the DICOM dataset
        contrast = dicom_ds.get((0x18, 0x10))
        series_description = dicom_ds.get((0x08, 0x103E)).value

        # Check modality and determine contrast category
        if modality_enum == ModalityEnum.MR:
            if contrast and len(str(contrast.value)) > 0:
                return ContrastEnum.CE
            result = self.pattern.search(series_description)
            if result:
                return ContrastEnum.CE
            return ContrastEnum.NE
        elif modality_enum == ModalityEnum.CT:
            if contrast:
                return ContrastEnum.CE
            else:
                return ContrastEnum.NE

        # Return NullEnum.NULL if modality is not MR or CT
        return NullEnum.NULL


class MRAcquisitionTypeProcessingStrategy(SeriesProcessingStrategy):
    """
    Concrete class for processing DICOM datasets based on MR acquisition type.

    Attributes
    ----------
    mr_acquisition_type_list : List[MRAcquisitionTypeEnum]
        List of MR acquisition types from MRAcquisitionTypeEnum.

    Methods
    -------
    process(dicom_ds: FileDataset) -> Union[BaseEnum, ImageOrientationEnum]
        Process the DICOM dataset based on MR acquisition type and return the result.
    """

    mr_acquisition_type_list: List[MRAcquisitionTypeEnum] = MRAcquisitionTypeEnum.to_list()

    def process(self, dicom_ds: FileDataset) -> Union[BaseEnum, ImageOrientationEnum]:
        """
        Process the DICOM dataset based on MR acquisition type.

        Parameters
        ----------
        dicom_ds : FileDataset
            The DICOM dataset to be processed.

        Returns
        -------
        Union[BaseEnum, ImageOrientationEnum]
            The result of processing based on MR acquisition type.
        """
        # Extract the MR acquisition type information from the DICOM dataset
        mr_acquisition_type = dicom_ds.get((0x18, 0x23))

        # Check if MR acquisition type information is available
        if mr_acquisition_type:
            # Iterate through the MR acquisition type enums to find a match
            for mr_acquisition_type_enum in self.mr_acquisition_type_list:
                if mr_acquisition_type_enum.value == mr_acquisition_type.value:
                    return mr_acquisition_type_enum

        # Return NullEnum.NULL if no match is found
        return NullEnum.NULL


class MRRenameSeriesProcessingStrategy(SeriesProcessingStrategy, ABC):
    """
    Abstract class for processing and renaming MR series based on specific criteria.

    Attributes
    ----------
    modality : ModalityEnum
        The modality for the series (assumed to be MR).
    mr_acquisition_type : Tuple[Union[MRAcquisitionTypeEnum, NullEnum]]
        Tuple of MR acquisition types from MRAcquisitionTypeEnum.
    modality_processing_strategy : ModalityProcessingStrategy
        An instance of ModalityProcessingStrategy used for modality processing.
    mr_acquisition_type_processing_strategy : MRAcquisitionTypeProcessingStrategy
        An instance of MRAcquisitionTypeProcessingStrategy used for MR acquisition type processing.

    Methods
    -------
    process(dicom_ds: FileDataset) -> Union[MRSeriesRenameEnum, NullEnum]
        Abstract method to be implemented by subclasses for processing and renaming MR series.
    """

    modality: ModalityEnum = ModalityEnum.MR
    mr_acquisition_type: Tuple[Union[MRAcquisitionTypeEnum, NullEnum]] = tuple(MRAcquisitionTypeEnum.to_list())
    modality_processing_strategy: ModalityProcessingStrategy = ModalityProcessingStrategy()
    mr_acquisition_type_processing_strategy: MRAcquisitionTypeProcessingStrategy = MRAcquisitionTypeProcessingStrategy()

    @abstractmethod
    def process(self, dicom_ds: FileDataset) -> Union[MRSeriesRenameEnum, NullEnum]:
        """
        Abstract method for processing and renaming MR series.

        Parameters
        ----------
        dicom_ds : FileDataset
            The DICOM dataset to be processed.

        Returns
        -------
        Union[MRSeriesRenameEnum, NullEnum]
            The result of processing and renaming the MR series.
        """
        pass


class CTRenameSeriesProcessingStrategy(SeriesProcessingStrategy, ABC):
    modality: ModalityEnum = ModalityEnum.CT
    modality_processing_strategy: ModalityProcessingStrategy = ModalityProcessingStrategy()

    @abstractmethod
    def process(self, dicom_ds: FileDataset) -> Union[MRSeriesRenameEnum, NullEnum]:
        pass
