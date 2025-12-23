
from typing import  List,  Optional
from pydantic import Field, ConfigDict, BaseModel
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


class AneurysmMaskSeries2Request(MaskSeriesRequest):
    model_type   : Optional[str] = Field(None, exclude=True)
    model_config = ConfigDict(from_attributes=True)



class AneurysmMaskModel2Request(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    model_type : Optional[str] = Field(None, exclude=True)
    series     : Optional[List[AneurysmMaskSeries2Request]] = Field(None)



class AneurysmMask2Request(MaskRequest):
    model_config = ConfigDict(from_attributes=True)
    model  : Optional[List[AneurysmMaskModel2Request]] = Field(None)
    # 排除 series
    series : Optional[List[AneurysmMaskSeriesRequest]] = Field(None,exclude=True)


class AneurysmAITeam2Request(AneurysmAITeamRequest):
    mask    : Optional[AneurysmMask2Request] = Field(None)