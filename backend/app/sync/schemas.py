"""Pydantic schema 定義，提供 FastAPI 請求與回應的驗證邏輯。"""

# app/sync/schemas.py
import datetime
import re
from typing import List, Annotated, Dict, Any

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, AfterValidator, ConfigDict, field_validator, model_validator


def validate_orthanc_id(v: str) -> str:
    """Orthanc UID 由 5 組 8 碼十六進位組成，違反格式時直接拒絕。"""
    orthanc_pattern = r"^[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}$"
    if not re.match(orthanc_pattern, v, re.IGNORECASE):
        raise ValueError(f"Invalid Orthanc ID format: {v}")
    return v


OrthancID = Annotated[str, AfterValidator(validate_orthanc_id)]

OpeNo = Annotated[
    str,
    Field(
        ...,
        min_length=7,
        max_length=7,
        pattern=r"^\d{3}\.\d{3}$",
        description="格式必須為 xxx.xxx，其中 x 為數字",
    ),
]

# class OrthancIDRequest(BaseModel):
#     ids: list[OrthancID]


class PostStudyRequest(BaseModel):
    """同步 study 的請求載體，支援舊版 ids 以及新版 studyUid 欄位。"""

    ids: list[OrthancID] = Field(default_factory=list)
    study_uid: Optional[OrthancID] = None
    prev_study_uid: Optional[OrthancID] = None
    msg: Optional[str] = Field(
        default=None, description="格式必須為 xxx.xxx，其中 x 為數字"
    )

    @field_validator("ids", mode="before")
    @classmethod
    def default_ids(cls, value: Optional[List[OrthancID]]) -> List[OrthancID]:
        if value is None:
            return []
        return value

    @field_validator("study_uid", "prev_study_uid", mode="before")
    @classmethod
    def normalize_optional_uid(cls, value: Optional[str]) -> Optional[str]:
        if value in ("", None):
            return None
        return value

    @model_validator(mode="after")
    def ensure_payload(self) -> "PostStudyRequest":
        if not self.ids and not self.study_uid:
            raise ValueError("ids 或 study_uid 需至少提供一種")
        return self

    def resolved_ids(self) -> List[OrthancID]:
        """將輸入正規化為 OrthancID 清單。"""
        if self.ids:
            return self.ids
        return [self.study_uid] if self.study_uid else []


class DCOPEventRequest(BaseModel):
    """DICOM 同步事件標準欄位，用於 webhook 與 DB 互轉。"""

    model_config = ConfigDict(from_attributes=True)
    study_uid: OrthancID = OrthancID("ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30")
    series_uid: Optional[OrthancID] = OrthancID(
        "31fb1be1-71d25700-b131126f-c73708af-42d28093"
    )
    ope_no: str = Field(
        ...,
        min_length=7,
        max_length=7,
        pattern=r"^\d{3}\.\d{3}$",
        description="格式必須為 xxx.xxx，其中 x 為數字",
    )
    tool_id: str = "DICOM_TOOL"
    study_id: Optional[str] = None
    params_data: Optional[Dict[str, Any]] = None
    result_data: Optional[Dict[str, Any]] = None
    create_time: Optional[datetime.datetime] = None


class DCOPEventNIFTITOOLRequest(BaseModel):
    """NIFTI_TOOL 上報狀態使用的最小 payload。"""

    ope_no: str = Field(
        ...,
        min_length=7,
        max_length=7,
        pattern=r"^\d{3}\.\d{3}$",
        description="格式必須為 xxx.xxx，其中 x 為數字",
    )
    tool_id: str = "NIFTI_TOOL"
    study_id: Optional[str] = None
    params_data: Optional[Dict[str, Any]] = None
    result_data: Optional[Dict[str, Any]] = None


class DCOPStatus(str, Enum):
    """同步/轉檔/推論所有狀態碼一覽，方便前後端共享。"""

    STUDY_NEW = "100.020"
    STUDY_TRANSFERRING = "100.050"
    STUDY_TRANSFER_COMPLETE = "100.100"

    SERIES_NEW = "100.025"
    SERIES_TRANSFERRING = "100.055"
    SERIES_TRANSFER_COMPLETE = "100.095"

    STUDY_CONVERTING = "200.150"
    STUDY_CONVERSION_COMPLETE = "200.200"

    SERIES_CONVERTING = "200.155"
    SERIES_CONVERSION_SKIP = "200.190"
    SERIES_CONVERSION_COMPLETE = "200.195"

    STUDY_INFERENCE_FAILED = "300.000"
    STUDY_INFERENCE_READY = "300.050"
    STUDY_INFERENCE_QUEUED = "300.100"
    STUDY_INFERENCE_RUNNING = "300.150"

    STUDY_INFERENCE_COMPLETE = "300.300"

    # SERIES_INFERENCE_FAILED    = "SERIES_INFERENCE_FAILED"
    SERIES_INFERENCE_READY = "300.055"
    SERIES_INFERENCE_QUEUED = "300.105"
    SERIES_INFERENCE_RUNNING = "300.155"
    SERIES_INFERENCE_COMPLETE = "300.295"

    # SERIES_RESULTS_SENT        = "SERIES_RESULTS_SENT"
    STUDY_RESULTS_SENT = "500.500"

    STUDY_NEW_RE = "100.021"
    STUDY_TRANSFERRING_RE = "100.051"
    STUDY_CONVERTING_RE = "200.151"
    STUDY_INFERENCE_RUNNING_RE = "300.151"

    pass


class StydySeriesOpeNoStatus(BaseModel):
    """查詢 study/series 所有 ope_no 狀態時的回傳模型。"""

    model_config = ConfigDict(from_attributes=True)
    study_uid: OrthancID
    series_uid: Optional[OrthancID] = None
    study_id: Optional[str] = None
    ope_no: List[OpeNo]
    result_data: Optional[List[Any]] = None
    params_data: Optional[List[Any]] = None
