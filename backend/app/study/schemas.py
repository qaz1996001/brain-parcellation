"""
研究 (Study) 模組的資料序列化方案和驗證規則。

此模組定義了所有與研究管理相關的 Pydantic 模型，包括：
- 驗證器：Orthanc ID、操作編號等格式驗證
- 請求方案：API 請求的資料結構
- 響應方案：API 響應的資料結構
- 狀態列舉：Study 和 Series 的所有可能狀態

Data Flow
---------
HTTP Request
    ↓
Pydantic 驗證 (schemas.py)
    ↓
業務邏輯 (service.py)
    ↓
資料庫 (model.py)
    ↓
序列化回應 (schemas.py)
    ↓
HTTP Response

Notes
-----
所有 ID 欄位都使用 Orthanc UUID 格式：
    8 hex digits - 8 hex digits - 8 hex digits - 8 hex digits - 8 hex digits

所有操作編號都使用 NNN.NNN 格式：
    e.g. "100.095" (Study Transfer Complete)
         "200.195" (Series Conversion Complete)
         "300.100" (Study Inference Queued)

See Also
--------
backend.app.study.routers : API 端點定義
backend.app.study.service : 業務邏輯層
"""

import re
from typing import Annotated, Dict, Any

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, AfterValidator, ConfigDict


def validate_orthanc_id(v: str) -> str:
    """
    驗證 Orthanc ID 的格式。
    
    Orthanc ID 必須符合以下格式：
    8 個十六進制數字 - 8 個十六進制數字 - 8 個十六進制數字 - 
    8 個十六進制數字 - 8 個十六進制數字
    
    例如：
        ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30
    
    Parameters
    ----------
    v : str
        待驗證的 Orthanc ID 字符串。
    
    Returns
    -------
    str
        驗證通過的 Orthanc ID（大小寫不區分）。
    
    Raises
    ------
    ValueError
        若 ID 格式不符合 Orthanc UUID 標準。
    
    Examples
    --------
    >>> validate_orthanc_id("ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30")
    "ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30"
    
    >>> validate_orthanc_id("invalid-id")
    ValueError: Invalid Orthanc ID format: invalid-id
    """
    orthanc_pattern = r"^[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}$"
    if not re.match(orthanc_pattern, v, re.IGNORECASE):
        raise ValueError(f"Invalid Orthanc ID format: {v}")
    return v


# Orthanc ID 類型別名 - 自動驗證格式
OrthancID = Annotated[str, AfterValidator(validate_orthanc_id)]


# class OrthancIDRequest(BaseModel):
#     ids: list[OrthancID]


class PostStudyRequest(BaseModel):
    """
    POST 請求 - 新增研究。
    
    此模型用於接收來自客戶端的新 Study 創建請求。
    
    Attributes
    ----------
    ids : list[OrthancID]
        待添加的 Study UID 列表。
        每個 ID 必須符合 Orthanc UUID 格式。
    msg : str
        操作編號，格式為 xxx.xxx。
        例如 "100.020" 表示 STUDY_NEW。
    
    Examples
    --------
    >>> request = PostStudyRequest(
    ...     ids=["ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30"],
    ...     msg="100.020"
    ... )
    """
    ids: list[OrthancID] = Field(
        default=[OrthancID("ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30")],
        description="Study UID 列表，每個必須符合 Orthanc UUID 格式"
    )
    msg: str = Field(
        ..., 
        description="操作編號，格式必須為 xxx.xxx（e.g. 100.020）"
    )


class DCOPEventRequest(BaseModel):
    """
    DCOP 事件請求模型。
    
    此模型表示系統中的一個事件記錄，用於追蹤 Study 和 Series 
    在各個階段的狀態轉遷和相關操作。
    
    Attributes
    ----------
    study_uid : OrthancID
        Study 的全局唯一標識符（Orthanc UUID）。
    series_uid : Optional[OrthancID]
        Series 的全局唯一標識符（Orthanc UUID）。
        對於 Study 級別的事件，此欄位可為 None。
    ope_no : str
        操作編號，格式為 xxx.xxx。
        定義了事件的類型和階段。
    tool_id : str
        生成此事件的工具 ID。
        可能的值：DICOM_TOOL、NIFTI_TOOL、INFERENCE_TOOL。
    study_id : Optional[str]
        Study 的本地標識符（與 study_uid 不同）。
        用於文件系統中的路徑組織。
    params_data : Optional[Dict[str, Any]]
        事件的參數數據（如文件路徑、配置等）。
    result_data : Optional[Dict[str, Any]]
        事件的結果數據（如轉檔結果、推理輸出等）。
    
    Examples
    --------
    >>> event = DCOPEventRequest(
    ...     study_uid="ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30",
    ...     series_uid="31fb1be1-71d25700-b131126f-c73708af-42d28093",
    ...     ope_no="100.095",
    ...     tool_id="DICOM_TOOL",
    ...     study_id="12345",
    ...     params_data={"sub_dir": "/path/to/dicom"},
    ...     result_data={"renamed_path": "/path/to/renamed"}
    ... )
    
    Notes
    -----
    - 操作編號定義了事件的語義，遵循固定的格式 NNN.NNN
    - 每個 Study 經過特定的狀態序列才能進入下一階段
    - 完整的事件歷史記錄用於審計和故障排查
    """
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
        description="操作編號，格式必須為 xxx.xxx（e.g. 100.095）",
    )
    tool_id: str = Field(
        default="DICOM_TOOL",
        description="生成事件的工具 ID（DICOM_TOOL、NIFTI_TOOL、INFERENCE_TOOL）"
    )
    study_id: Optional[str] = Field(
        default=None,
        description="Study 的本地標識符"
    )
    params_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="事件的參數數據（如文件路徑、配置等）"
    )
    result_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="事件的結果數據（如轉檔結果、推理輸出等）"
    )


class DCOPEventNIFTITOOLRequest(BaseModel):
    """
    NIFTI 轉檔工具事件請求模型。
    
    此模型專門用於 NIFTI 轉檔工具上報的事件。
    
    Attributes
    ----------
    ope_no : str
        操作編號，格式為 xxx.xxx。
    tool_id : str
        工具 ID，固定為 "NIFTI_TOOL"。
    study_id : Optional[str]
        Study 的本地標識符。
    params_data : Optional[Dict[str, Any]]
        轉檔參數（輸入/輸出路徑等）。
    result_data : Optional[Dict[str, Any]]
        轉檔結果。
    
    Examples
    --------
    >>> request = DCOPEventNIFTITOOLRequest(
    ...     ope_no="200.150",
    ...     tool_id="NIFTI_TOOL",
    ...     study_id="12345",
    ...     params_data={
    ...         "output_dicom_path": "/path/to/dicom",
    ...         "output_nifti_path": "/path/to/nifti"
    ...     }
    ... )
    """
    ope_no: str = Field(
        ...,
        min_length=7,
        max_length=7,
        pattern=r"^\d{3}\.\d{3}$",
        description="操作編號，格式必須為 xxx.xxx",
    )
    tool_id: str = Field(
        default="NIFTI_TOOL",
        description="工具 ID，固定為 NIFTI_TOOL"
    )
    study_id: Optional[str] = Field(
        default=None,
        description="Study 的本地標識符"
    )
    params_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="轉檔參數"
    )
    result_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="轉檔結果"
    )


class DCOPStatus(str, Enum):
    """
    DCOP 系統中所有可能的事件狀態列舉。
    
    此列舉定義了 Study 和 Series 在生命週期中的所有狀態。
    每個狀態都有一個唯一的操作編號（ope_no），格式為 NNN.NNN。
    
    狀態編號的含義
    ---------------
    第一個 NNN（百位）表示階段：
    - 100: 傳輸階段 (Transfer)
    - 200: 轉檔階段 (Conversion)
    - 300: 推理階段 (Inference)
    - 500: 完成階段 (Result)
    
    第二個 NNN（個位）表示具體狀態：
    - 020/025: NEW（新建）
    - 050/055: TRANSFERRING（轉移中）
    - 100/095: TRANSFER_COMPLETE（轉移完成）
    - 150/155: CONVERTING（轉檔中）
    - 190: CONVERSION_SKIP（轉檔跳過）
    - 195/200: CONVERSION_COMPLETE（轉檔完成）
    - 等等...
    
    Study 生命週期
    ---------------
    1. 傳輸階段:
       STUDY_NEW (100.020)
       → STUDY_TRANSFERRING (100.050)
       → STUDY_TRANSFER_COMPLETE (100.100)
    
    2. 轉檔階段:
       STUDY_CONVERTING (200.150)
       → STUDY_CONVERSION_COMPLETE (200.200)
    
    3. 推理階段:
       STUDY_INFERENCE_READY (300.050)
       → STUDY_INFERENCE_QUEUED (300.100)
       → STUDY_INFERENCE_RUNNING (300.150)
       → STUDY_INFERENCE_COMPLETE (300.300)
    
    4. 完成階段:
       STUDY_RESULTS_SENT (500.500)
    
    Series 生命週期
    ---------------
    與 Study 相似，但針對單個 Series。
    
    重試狀態
    --------
    帶有 _RE 後綴的狀態用於故障後重試：
    - STUDY_NEW_RE (100.021)
    - STUDY_TRANSFERRING_RE (100.051)
    - STUDY_CONVERTING_RE (200.151)
    - STUDY_INFERENCE_RUNNING_RE (300.151)
    
    Attributes
    ----------
    STUDY_NEW : str
        Study 新建狀態。
    STUDY_TRANSFERRING : str
        Study 傳輸進行中。
    STUDY_TRANSFER_COMPLETE : str
        Study 傳輸完成。
    SERIES_NEW : str
        Series 新建狀態。
    SERIES_TRANSFERRING : str
        Series 傳輸進行中。
    SERIES_TRANSFER_COMPLETE : str
        Series 傳輸完成。
    STUDY_CONVERTING : str
        Study 轉檔進行中。
    STUDY_CONVERSION_COMPLETE : str
        Study 轉檔完成。
    SERIES_CONVERTING : str
        Series 轉檔進行中。
    SERIES_CONVERSION_SKIP : str
        Series 轉檔被跳過（例如不需要轉檔的序列）。
    SERIES_CONVERSION_COMPLETE : str
        Series 轉檔完成。
    STUDY_INFERENCE_FAILED : str
        Study 推理失敗。
    STUDY_INFERENCE_READY : str
        Study 已準備推理。
    STUDY_INFERENCE_QUEUED : str
        Study 已入隊推理。
    STUDY_INFERENCE_RUNNING : str
        Study 推理進行中。
    STUDY_INFERENCE_COMPLETE : str
        Study 推理完成。
    SERIES_INFERENCE_READY : str
        Series 已準備推理。
    SERIES_INFERENCE_QUEUED : str
        Series 已入隊推理。
    SERIES_INFERENCE_RUNNING : str
        Series 推理進行中。
    SERIES_INFERENCE_COMPLETE : str
        Series 推理完成。
    STUDY_RESULTS_SENT : str
        Study 結果已發送。
    STUDY_NEW_RE : str
        Study 新建重試。
    STUDY_TRANSFERRING_RE : str
        Study 傳輸重試。
    STUDY_CONVERTING_RE : str
        Study 轉檔重試。
    STUDY_INFERENCE_RUNNING_RE : str
        Study 推理重試。
    
    Examples
    --------
    >>> status = DCOPStatus.STUDY_NEW
    >>> print(status.value)
    "100.020"
    
    >>> if event.ope_no == DCOPStatus.STUDY_TRANSFER_COMPLETE.value:
    ...     print("Study 傳輸已完成")
    """
    
    # ========== 傳輸階段 (100.xxx) ==========
    # Study 傳輸階段
    STUDY_NEW = "100.020"
    """Study 新建狀態。"""
    
    STUDY_TRANSFERRING = "100.050"
    """Study 傳輸進行中。"""
    
    STUDY_TRANSFER_COMPLETE = "100.100"
    """Study 傳輸完成。"""

    # Series 傳輸階段
    SERIES_NEW = "100.025"
    """Series 新建狀態。"""
    
    SERIES_TRANSFERRING = "100.055"
    """Series 傳輸進行中。"""
    
    SERIES_TRANSFER_COMPLETE = "100.095"
    """Series 傳輸完成。"""

    # ========== 轉檔階段 (200.xxx) ==========
    # Study 轉檔階段
    STUDY_CONVERTING = "200.150"
    """Study 轉檔進行中。"""
    
    STUDY_CONVERSION_COMPLETE = "200.200"
    """Study 轉檔完成。"""

    # Series 轉檔階段
    SERIES_CONVERTING = "200.155"
    """Series 轉檔進行中。"""
    
    SERIES_CONVERSION_SKIP = "200.190"
    """Series 轉檔被跳過。"""
    
    SERIES_CONVERSION_COMPLETE = "200.195"
    """Series 轉檔完成。"""

    # ========== 推理階段 (300.xxx) ==========
    # Study 推理階段
    STUDY_INFERENCE_FAILED = "300.000"
    """Study 推理失敗。"""
    
    STUDY_INFERENCE_READY = "300.050"
    """Study 已準備推理。"""
    
    STUDY_INFERENCE_QUEUED = "300.100"
    """Study 已入隊推理。"""
    
    STUDY_INFERENCE_RUNNING = "300.150"
    """Study 推理進行中。"""

    STUDY_INFERENCE_COMPLETE = "300.300"
    """Study 推理完成。"""

    # Series 推理階段
    # SERIES_INFERENCE_FAILED    = "SERIES_INFERENCE_FAILED"
    SERIES_INFERENCE_READY = "300.055"
    """Series 已準備推理。"""
    
    SERIES_INFERENCE_QUEUED = "300.105"
    """Series 已入隊推理。"""
    
    SERIES_INFERENCE_RUNNING = "300.155"
    """Series 推理進行中。"""
    
    SERIES_INFERENCE_COMPLETE = "300.295"
    """Series 推理完成。"""

    # ========== 完成階段 (500.xxx) ==========
    # SERIES_RESULTS_SENT        = "SERIES_RESULTS_SENT"
    STUDY_RESULTS_SENT = "500.500"
    """Study 結果已發送。"""

    # ========== 重試狀態 ==========
    STUDY_NEW_RE = "100.021"
    """Study 新建重試。"""
    
    STUDY_TRANSFERRING_RE = "100.051"
    """Study 傳輸重試。"""
    
    STUDY_CONVERTING_RE = "200.151"
    """Study 轉檔重試。"""
    
    STUDY_INFERENCE_RUNNING_RE = "300.151"
    """Study 推理重試。"""
