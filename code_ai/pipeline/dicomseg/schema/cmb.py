
from typing import List,  Optional
from pydantic import Field, BaseModel, ConfigDict
from pydantic import field_validator


from .base import MaskRequest,MaskSeriesRequest, MaskInstanceRequest, SeriesTypeEnum
from .base import AITeamRequest, StudyRequest,SortedRequest,StudyModelRequest


class CMBMaskInstanceRequest(MaskInstanceRequest):
    sub_location: Optional[str] = Field(None, exclude=True)  # 標記為排除


class CMBMaskSeriesRequest(MaskSeriesRequest):
    instances: List[CMBMaskInstanceRequest]


class CMBMaskRequest(MaskRequest):
    series: List[CMBMaskSeriesRequest]



class CMBStudyModelRequest(StudyModelRequest):
    series_type:List[str]

    @field_validator('series_type', mode='before')
    @classmethod
    def extract_series_type(cls, value):
        if value is None:
            return ["1"]
        if isinstance(value, list):
            series_type_list = []
            series_type_enum_list: List[SeriesTypeEnum] = SeriesTypeEnum.to_list()
            for series_type_enum in series_type_enum_list:
                if value == series_type_enum.name:
                    series_type_list.append(series_type_enum.value)
            return series_type_list
        else:
            return list(value)


class CMBStudyRequest(StudyRequest):
    model: List[StudyModelRequest]


class CMBAITeamRequest(AITeamRequest):
    study   : Optional[StudyRequest]   = Field(None)
    sorted  : Optional[SortedRequest]  = Field(None)
    mask    : Optional[CMBMaskRequest] = Field(None)



class CMBMaskSeries2Request(MaskSeriesRequest):
    model_type   : Optional[str] = Field(None, exclude=True)
    model_config = ConfigDict(from_attributes=True)



class CMBMaskModel2Request(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    model_type : Optional[str] = Field(None)
    series     : Optional[List[CMBMaskSeries2Request]] = Field(None)



class CMBMask2Request(MaskRequest):
    model_config = ConfigDict(from_attributes=True)
    model  : Optional[List[CMBMaskModel2Request]] = Field(None)
    # 排除 series
    series : Optional[List[CMBMaskSeriesRequest]] = Field(None,exclude=True)


class CMBAITeam2Request(AITeamRequest):
    mask    : Optional[CMBMask2Request] = Field(None)