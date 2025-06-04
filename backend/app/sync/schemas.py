# app/sync/schemas.py
import re
from typing import List, Annotated, Dict

from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator,AfterValidator


def validate_orthanc_id(v: str) -> str:
    orthanc_pattern = r'^[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}$'
    if not re.match(orthanc_pattern, v, re.IGNORECASE):
        raise ValueError(f'Invalid Orthanc ID format: {v}')
    return v


OrthancID = Annotated[str, AfterValidator(validate_orthanc_id)]


class OrthancIDRequest(BaseModel):
    ids: list[OrthancID]


class DCOPEventRequest(BaseModel):
    study_uid  : OrthancID           = OrthancID('ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30')
    series_uid : Optional[OrthancID] = OrthancID('31fb1be1-71d25700-b131126f-c73708af-42d28093')
    ope_no     : str                 =  Field(...,min_length=7,max_length=7,pattern=r"^\d{3}\.\d{3}$",
                                                  description="格式必須為 xxx.xxx，其中 x 為數字")
    tool_id   : str                  = 'DICOM_TOOL'
    study_id  : Optional[str]           = None
    params_data:Optional[Dict[str,str]] = None
    result_data:Optional[Dict[str,str]] = None


# class DCOPStatus(str, Enum):
#     STUDY_NEW                  = "STUDY_NEW",
#     STUDY_TRANSFERRING         = "STUDY_TRANSFERRING",
#     STUDY_TRANSFER_COMPLETE    = "STUDY_TRANSFER_COMPLETE",
#     STUDY_CONVERTING           = "STUDY_CONVERTING",
#     STUDY_CONVERSION_COMPLETE  = "STUDY_CONVERSION_COMPLETE"
#     STUDY_INFERENCE_READY      = "STUDY_INFERENCE_READY",
#     STUDY_INFERENCE_QUEUED     = "STUDY_INFERENCE_QUEUED",
#     STUDY_INFERENCE_RUNNING    = "STUDY_INFERENCE_RUNNING",
#     STUDY_INFERENCE_FAILED     = "STUDY_INFERENCE_FAILED",
#     STUDY_INFERENCE_COMPLETE   = "STUDY_INFERENCE_COMPLETE",
#     STUDY_RESULTS_SENT         = "STUDY_RESULTS_SENT",
#     SERIES_NEW                 = "SERIES_NEW",
#     SERIES_TRANSFERRING        = "SERIES_TRANSFERRING",
#     SERIES_TRANSFER_COMPLETE   = "SERIES_TRANSFER_COMPLETE",
#     SERIES_CONVERTING          = "SERIES_CONVERTING",
#     SERIES_CONVERSION_COMPLETE = "SERIES_CONVERSION_COMPLETE"
#     SERIES_INFERENCE_FAILED    = "SERIES_INFERENCE_FAILED",
#     SERIES_INFERENCE_READY     = "SERIES_INFERENCE_READY",
#     SERIES_INFERENCE_QUEUED    = "SERIES_INFERENCE_QUEUED",
#     SERIES_INFERENCE_RUNNING   = "SERIES_INFERENCE_RUNNING",
#     SERIES_INFERENCE_COMPLETE  = "SERIES_INFERENCE_COMPLETE"
#     SERIES_RESULTS_SENT        = "SERIES_RESULTS_SENT"
#     pass

class DCOPStatus(str, Enum):

    STUDY_NEW                  = "100.020",
    STUDY_TRANSFERRING         = "100.050",
    STUDY_TRANSFER_COMPLETE    = "100.100",

    SERIES_NEW                 = "100.025",
    SERIES_TRANSFERRING        = "100.055",
    SERIES_TRANSFER_COMPLETE   = "100.095",


    STUDY_CONVERTING           = "200.150",
    STUDY_CONVERSION_COMPLETE  = "200.155"

    SERIES_CONVERTING          = "200.195",
    SERIES_CONVERSION_COMPLETE = "200.200"

    STUDY_INFERENCE_FAILED     = "300.000",
    STUDY_INFERENCE_READY      = "300.050",
    STUDY_INFERENCE_QUEUED     = "300.100",
    STUDY_INFERENCE_RUNNING    = "300.150",
    STUDY_INFERENCE_COMPLETE   = "300.300",

    # SERIES_INFERENCE_FAILED    = "SERIES_INFERENCE_FAILED",
    SERIES_INFERENCE_READY     = "300.055",
    SERIES_INFERENCE_QUEUED    = "300.105",
    SERIES_INFERENCE_RUNNING   = "300.155",
    SERIES_INFERENCE_COMPLETE  = "300.295"

    # SERIES_RESULTS_SENT        = "SERIES_RESULTS_SENT"
    STUDY_RESULTS_SENT         = "500.500",

    pass