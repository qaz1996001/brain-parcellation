import os
import pydicom
from shhai import RESOURCE_DIR
from .schema import SeriesTypeEnum,InstanceRequest
from .schema import SeriesRequest,SortedRequest
from .schema import StudyRequest,MaskInstanceRequest
from .schema import MaskSeriesRequest,MaskRequest
from .schema import AITeamRequest

EXAMPLE_FILE = os.path.join(RESOURCE_DIR,'dicomseg', 'SEG_20230210_160056_635_S3.dcm')
DCM_EXAMPLE = pydicom.dcmread(EXAMPLE_FILE)

__all__ = ["SeriesTypeEnum","InstanceRequest","SeriesRequest","SortedRequest","StudyRequest",
           "MaskInstanceRequest","MaskSeriesRequest","MaskRequest","AITeamRequest"
           "EXAMPLE_FILE","DCM_EXAMPLE"]