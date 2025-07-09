
from typing import  List,  Optional
from pydantic import Field
from .base import MaskRequest,MaskSeriesRequest, MaskInstanceRequest
from .base import AITeamRequest, StudyRequest,SortedRequest


class AneurysmMaskInstanceRequest(MaskInstanceRequest):
    sub_location: Optional[str] = Field(None)  # 標記為排除


class AneurysmMaskSeriesRequest(MaskSeriesRequest):
    instances: List[AneurysmMaskInstanceRequest]


class AneurysmMaskRequest(MaskRequest):
    series: List[AneurysmMaskSeriesRequest]


class AneurysmAITeamRequest(AITeamRequest):
    study   : Optional[StudyRequest]   = Field(None)
    sorted  : Optional[SortedRequest]  = Field(None)
    mask    : Optional[AneurysmMaskRequest] = Field(None)
