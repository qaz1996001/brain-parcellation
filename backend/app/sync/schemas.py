"""
Pydantic schema 定義，提供 FastAPI 請求與回應的驗證邏輯。

此模組定義了 DICOM 同步服務的所有 Pydantic 模型，確保 API 請求和回應
的類型安全。包括：
- 自訂驗證器和類型別名
- 請求載體（PostStudyRequest、DCOPEventRequest 等）
- 狀態列舉（DCOPStatus）
- 回應模型（StydySeriesOpeNoStatus）

所有驗證遵循 "Good Taste" 設計原則：
- 驗證器簡潔而明確
- 錯誤消息清晰可讀
- 支援向後相容性（如 PostStudyRequest 的舊版欄位）
"""

import datetime
import re
from typing import List, Annotated, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, AfterValidator, ConfigDict, field_validator, model_validator


def validate_orthanc_id(v: str) -> str:
    """
    驗證 Orthanc ID 格式。
    
    Orthanc 系統使用 40 碼十六進位 ID，格式為 8-8-8-8-8 的 5 組。
    此驗證器確保輸入符合此格式。
    
    Parameters
    ----------
    v : str
        待驗證的 Orthanc ID 字符串。
    
    Returns
    -------
    str
        驗證成功時返回原字符串。
    
    Raises
    ------
    ValueError
        ID 格式不符合預期時拋出異常。
    
    Examples
    --------
    有效的 ID：
    
    >>> id = validate_orthanc_id("ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30")
    >>> print(id)
    ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30
    
    無效的 ID 將拋出 ValueError：
    
    >>> validate_orthanc_id("invalid-id")  # doctest: +SKIP
    ValueError: Invalid Orthanc ID format: invalid-id
    """
    # 定義 Orthanc ID 的正則表達式：5 組 8 碼十六進位
    orthanc_pattern = r"^[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}$"
    
    # 檢查格式，忽略大小寫
    if not re.match(orthanc_pattern, v, re.IGNORECASE):
        raise ValueError(f"Invalid Orthanc ID format: {v}")
    
    # 返回標準化後的 ID（小寫）
    return v.lower()


# Annotated 類型別名：自動應用驗證器
OrthancID = Annotated[
    str,
    AfterValidator(validate_orthanc_id),
    Field(description="Orthanc Study/Series UID（40 碼十六進位，格式 xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx）")
]

# Annotated 類型別名：操作編號欄位
OpeNo = Annotated[
    str,
    Field(
        ...,
        min_length=7,
        max_length=7,
        pattern=r"^\d{3}\.\d{3}$",
        description="操作編號，格式必須為 xxx.xxx，其中 x 為數字",
    ),
]

# class OrthancIDRequest(BaseModel):
#     ids: list[OrthancID]


class PostStudyRequest(BaseModel):
    """
    Study 同步請求載體，支援新舊版本的相容性。
    
    此模型用於接收 API 客戶端的 Study 同步請求。為了維持
    向後相容性，同時支援舊版本的 ids 欄位和新版本的 study_uid 欄位。
    
    內部驗證邏輯確保：
    1. 兩種欄位至少提供一種
    2. 空字符串被正規化為 None
    3. None 列表被轉換為空列表
    
    Attributes
    ----------
    ids : list[OrthancID]
        舊版本欄位，支援多個 Study UID（向後相容）。
        預設為空列表。
    study_uid : OrthancID, optional
        新版本欄位，單個 Study UID，建議使用。
    prev_study_uid : OrthancID, optional
        前一個 Study UID，用於建立時序鏈結。
    msg : str, optional
        請求附帶的消息或說明。
    
    Methods
    -------
    resolved_ids()
        將輸入正規化為統一的 OrthancID 列表。
    
    Examples
    --------
    使用新版本 study_uid：
    
    >>> req = PostStudyRequest(
    ...     study_uid="ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30"
    ... )
    >>> req.resolved_ids()
    ['ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30']
    
    使用舊版本 ids 列表：
    
    >>> req = PostStudyRequest(
    ...     ids=[
    ...         "ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30",
    ...         "11111111-22222222-33333333-44444444-55555555"
    ...     ]
    ... )
    >>> len(req.resolved_ids())
    2
    
    建立 Study 鏈結：
    
    >>> req = PostStudyRequest(
    ...     study_uid="current-123",
    ...     prev_study_uid="baseline-001"
    ... )
    >>> req.resolved_ids()
    ['current-123']
    
    Notes
    -----
    Good Taste 設計：
    - 驗證器消除了特殊情況（None vs []）
    - resolved_ids() 提供統一的介面
    - 支援多種輸入格式卻保持簡潔
    """

    ids: list[OrthancID] = Field(
        default_factory=list,
        description="舊版本欄位：多個 Study UID（向後相容）"
    )
    study_uid: Optional[OrthancID] = Field(
        default=None,
        description="新版本欄位：單個 Study UID（建議使用）"
    )
    prev_study_uid: Optional[OrthancID] = Field(
        default=None,
        description="前一個 Study UID，用於建立時序鏈結"
    )
    msg: Optional[str] = Field(
        default=None,
        description="請求附帶的消息"
    )

    @field_validator("ids", mode="before")
    @classmethod
    def default_ids(cls, value: Optional[List[OrthancID]]) -> List[OrthancID]:
        """
        確保 ids 欄位總是列表（消除 None 特殊情況）。
        
        Parameters
        ----------
        value : list or None
            輸入值。
        
        Returns
        -------
        list
            如果輸入是 None 或未提供，返回空列表；否則返回原列表。
        """
        if value is None:
            return []
        return value

    @field_validator("study_uid", "prev_study_uid", mode="before")
    @classmethod
    def normalize_optional_uid(cls, value: Optional[str]) -> Optional[str]:
        """
        正規化可選 UID 欄位，將空字符串轉換為 None。
        
        Parameters
        ----------
        value : str or None
            輸入值。
        
        Returns
        -------
        str or None
            如果輸入是空字符串或 None，返回 None；否則返回原值。
        """
        if value in ("", None):
            return None
        return value

    @model_validator(mode="after")
    def ensure_payload(self) -> "PostStudyRequest":
        """
        驗證至少提供了一種必要的 Study 標識符。
        
        Raises
        ------
        ValueError
            如果既未提供 ids 也未提供 study_uid。
        
        Returns
        -------
        PostStudyRequest
            驗證成功時返回模型實例。
        """
        if not self.ids and not self.study_uid:
            raise ValueError(
                "至少需提供 ids 或 study_uid 中的一種"
            )
        return self

    def resolved_ids(self) -> List[OrthancID]:
        """
        將輸入正規化為統一的 OrthancID 清單。
        
        優先返回 ids 列表，若為空則返回 study_uid 作為單元素列表。
        此方法確保調用方總能獲得一致的列表格式。
        
        Returns
        -------
        list[OrthancID]
            Study UID 列表。
        
        Examples
        --------
        >>> req = PostStudyRequest(study_uid="abc-123")
        >>> req.resolved_ids()
        ['abc-123']
        
        >>> req = PostStudyRequest(ids=["abc-123", "def-456"])
        >>> req.resolved_ids()
        ['abc-123', 'def-456']
        """
        if self.ids:
            return self.ids
        return [self.study_uid] if self.study_uid else []


class DCOPEventRequest(BaseModel):
    """
    DICOM 同步事件標準 schema，用於 webhook 與資料庫之間的互轉。
    
    此模型定義了 DICOM 同步事件的標準載體格式，支援從資料庫模型
    自動轉換（via from_attributes=True）。API 客戶端和內部服務
    都使用此模型進行事件通信。
    
    Attributes
    ----------
    study_uid : OrthancID
        事件所屬的 DICOM Study UID，必須提供。
    series_uid : OrthancID, optional
        事件所屬的 Series UID，Series 維度事件時提供。
    ope_no : str
        操作編號，格式 xxx.xxx，必須提供。
    tool_id : str
        觸發事件的工具代碼，預設 "DICOM_TOOL"。
    study_id : str, optional
        醫院 HIS 系統的 Study 識別碼。
    params_data : dict, optional
        JSON 格式的參數資料。
    result_data : dict, optional
        JSON 格式的結果資料。
    create_time : datetime, optional
        事件建立時間。
    
    Notes
    -----
    此模型設置 from_attributes=True，允許從 DCOPEventModel
    資料庫物件直接初始化。
    
    Examples
    --------
    從資料庫模型轉換：
    
    >>> event_model = DCOPEventModel(...)
    >>> event_request = DCOPEventRequest.from_attributes(event_model)
    
    建立請求載體：
    
    >>> event = DCOPEventRequest(
    ...     study_uid="ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30",
    ...     ope_no="100.100",
    ...     tool_id="DICOM_TOOL"
    ... )
    """

    model_config = ConfigDict(from_attributes=True)
    
    study_uid: OrthancID = Field(
        ...,
        description="DICOM Study UID"
    )
    series_uid: Optional[OrthancID] = Field(
        default=None,
        description="DICOM Series UID（可選）"
    )
    ope_no: OpeNo = Field(
        ...,
        description="操作編號 xxx.xxx"
    )
    tool_id: str = Field(
        default="DICOM_TOOL",
        description="觸發事件的工具代碼"
    )
    study_id: Optional[str] = Field(
        default=None,
        description="醫院 HIS Study 識別碼"
    )
    params_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="參數資料"
    )
    result_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="結果資料"
    )
    create_time: Optional[datetime.datetime] = Field(
        default=None,
        description="事件建立時間"
    )


class DCOPEventNIFTITOOLRequest(BaseModel):
    """
    NIFTI_TOOL 狀態上報的最小 payload schema。
    
    NIFTI_TOOL 是獨立的外部工具，不與 DICOM sync 服務直接共享資料庫。
    此模型定義了 NIFTI_TOOL 透過 webhook 上報執行結果時的最小必要欄位。
    相比 DCOPEventRequest，此模型省略了 Study/Series UID，
    因為這些資訊由 NIFTI_TOOL 無法提供。
    
    Attributes
    ----------
    ope_no : str
        操作編號，格式 xxx.xxx，用於在配置表中查詢對應的 Study。
    tool_id : str
        工具代碼，固定值 "NIFTI_TOOL"。
    study_id : str, optional
        醫院 HIS 系統的 Study 識別碼。
    params_data : dict, optional
        NIFTI_TOOL 呼叫時傳入的參數。
    result_data : dict, optional
        NIFTI_TOOL 執行的結果。
        典型值：{"output_path": "/path/to/result.nii.gz", "success": true}
    
    Notes
    -----
    設計原則：
    - 最小化載體：只傳輸必要的資訊
    - 依賴 ope_no 進行反向查詢：系統根據 ope_no 查詢配置表
    - 支援鬆散耦合：NIFTI_TOOL 無需知道系統內部架構
    
    Examples
    --------
    NIFTI_TOOL 上報轉檔完成：
    
    >>> report = DCOPEventNIFTITOOLRequest(
    ...     ope_no="200.200",
    ...     tool_id="NIFTI_TOOL",
    ...     study_id="HIS_001",
    ...     result_data={
    ...         "output_path": "/data/nifti/result.nii.gz",
    ...         "success": True,
    ...         "format": "nifti"
    ...     }
    ... )
    """

    ope_no: OpeNo = Field(
        ...,
        description="操作編號 xxx.xxx"
    )
    tool_id: str = Field(
        default="NIFTI_TOOL",
        description="工具代碼，固定為 NIFTI_TOOL"
    )
    study_id: Optional[str] = Field(
        default=None,
        description="醫院 HIS Study 識別碼"
    )
    params_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="呼叫參數"
    )
    result_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="執行結果"
    )


class DCOPStatus(str, Enum):
    """
    DICOM 同步流程的所有狀態碼列舉。
    
    此列舉定義了 DICOM 同步系統中所有可能的狀態轉遷，涵蓋：
    1. Study/Series 傳輸階段（100.xxx）
    2. NIFTI 轉檔階段（200.xxx）
    3. AI 推論階段（300.xxx）
    4. 結果上報階段（500.xxx）
    
    狀態碼遵循統一的命名規範：
    - 第一位數字：流程階段
    - 第二位數字：子階段
    - 第三位數字：狀態等級
    
    此設計與 DCOPConfModel 一起使用，確保前後端狀態機一致。
    
    State Transitions (流程狀態轉遷圖)
    ===================================
    
    Study Transfer Phase (傳輸階段):
        STUDY_NEW → STUDY_TRANSFERRING → STUDY_TRANSFER_COMPLETE
    
    Series Transfer Phase:
        SERIES_NEW → SERIES_TRANSFERRING → SERIES_TRANSFER_COMPLETE
    
    NIFTI Conversion Phase (轉檔階段):
        STUDY_CONVERTING → STUDY_CONVERSION_COMPLETE
        SERIES_CONVERTING → SERIES_CONVERSION_COMPLETE or SERIES_CONVERSION_SKIP
    
    Inference Phase (推論階段):
        STUDY_INFERENCE_READY → STUDY_INFERENCE_QUEUED → 
        STUDY_INFERENCE_RUNNING → STUDY_INFERENCE_COMPLETE
    
    Error Path:
        任何階段 → STUDY_INFERENCE_FAILED（終止狀態）
    
    Results Phase:
        STUDY_INFERENCE_COMPLETE → STUDY_RESULTS_SENT
    
    Retry Path (重試):
        STUDY_NEW_RE, STUDY_TRANSFERRING_RE, STUDY_CONVERTING_RE, 
        STUDY_INFERENCE_RUNNING_RE（帶 _RE 後綴表示重試）
    
    Examples
    --------
    在狀態機中使用：
    
    >>> status = DCOPStatus.STUDY_TRANSFER_COMPLETE
    >>> print(status.value)
    100.100
    
    前後端通信中使用：
    
    >>> from backend.app.sync.schemas import DCOPStatus
    >>> statuses = [s.value for s in DCOPStatus]
    >>> print(statuses)
    ['100.020', '100.050', '100.100', ...]
    
    Attributes
    ----------
    STUDY_NEW : "100.020"
        新 Study 已接收。
    STUDY_TRANSFERRING : "100.050"
        Study 傳輸中。
    STUDY_TRANSFER_COMPLETE : "100.100"
        Study 傳輸完成。
    
    SERIES_NEW : "100.025"
        新 Series 已接收。
    SERIES_TRANSFERRING : "100.055"
        Series 傳輸中。
    SERIES_TRANSFER_COMPLETE : "100.095"
        Series 傳輸完成。
    
    STUDY_CONVERTING : "200.150"
        Study NIFTI 轉檔中。
    STUDY_CONVERSION_COMPLETE : "200.200"
        Study NIFTI 轉檔完成。
    
    SERIES_CONVERTING : "200.155"
        Series NIFTI 轉檔中。
    SERIES_CONVERSION_SKIP : "200.190"
        Series 轉檔跳過（例如不支援的格式）。
    SERIES_CONVERSION_COMPLETE : "200.195"
        Series NIFTI 轉檔完成。
    
    STUDY_INFERENCE_FAILED : "300.000"
        Study 推論失敗（終止狀態）。
    STUDY_INFERENCE_READY : "300.050"
        Study 推論準備就緒。
    STUDY_INFERENCE_QUEUED : "300.100"
        Study 推論已排隊。
    STUDY_INFERENCE_RUNNING : "300.150"
        Study 推論執行中。
    STUDY_INFERENCE_COMPLETE : "300.300"
        Study 推論完成。
    
    SERIES_INFERENCE_READY : "300.055"
        Series 推論準備就緒。
    SERIES_INFERENCE_QUEUED : "300.105"
        Series 推論已排隊。
    SERIES_INFERENCE_RUNNING : "300.155"
        Series 推論執行中。
    SERIES_INFERENCE_COMPLETE : "300.295"
        Series 推論完成。
    
    STUDY_RESULTS_SENT : "500.500"
        Study 結果已上報。
    
    STUDY_NEW_RE : "100.021"
        Study 新建（重試）。
    STUDY_TRANSFERRING_RE : "100.051"
        Study 傳輸中（重試）。
    STUDY_CONVERTING_RE : "200.151"
        Study 轉檔中（重試）。
    STUDY_INFERENCE_RUNNING_RE : "300.151"
        Study 推論中（重試）。
    
    Notes
    -----
    Good Taste 設計原則：
    - 狀態碼命名一致，易於理解
    - 前後端共享同一列舉，消除不必要的轉換
    - 支援重試路徑而不是引入複雜的錯誤恢復機制
    """

    # Study 傳輸階段
    STUDY_NEW = "100.020"
    STUDY_TRANSFERRING = "100.050"
    STUDY_TRANSFER_COMPLETE = "100.100"

    # Series 傳輸階段
    SERIES_NEW = "100.025"
    SERIES_TRANSFERRING = "100.055"
    SERIES_TRANSFER_COMPLETE = "100.095"

    # NIFTI 轉檔階段
    STUDY_CONVERTING = "200.150"
    STUDY_CONVERSION_COMPLETE = "200.200"

    SERIES_CONVERTING = "200.155"
    SERIES_CONVERSION_SKIP = "200.190"
    SERIES_CONVERSION_COMPLETE = "200.195"

    # 推論階段
    STUDY_INFERENCE_FAILED = "300.000"
    STUDY_INFERENCE_READY = "300.050"
    STUDY_INFERENCE_QUEUED = "300.100"
    STUDY_INFERENCE_RUNNING = "300.150"
    STUDY_INFERENCE_COMPLETE = "300.300"

    # Series 推論階段
    SERIES_INFERENCE_READY = "300.055"
    SERIES_INFERENCE_QUEUED = "300.105"
    SERIES_INFERENCE_RUNNING = "300.155"
    SERIES_INFERENCE_COMPLETE = "300.295"

    # 結果階段
    STUDY_RESULTS_SENT = "500.500"

    # 重試路徑
    STUDY_NEW_RE = "100.021"
    STUDY_TRANSFERRING_RE = "100.051"
    STUDY_CONVERTING_RE = "200.151"
    STUDY_INFERENCE_RUNNING_RE = "300.151"


class StydySeriesOpeNoStatus(BaseModel):
    """
    查詢 Study/Series 所有操作狀態時的回傳模型。
    
    此模型用於聚合查詢，返回特定 Study 或 Series 的所有操作狀態。
    典型使用場景：前端查詢某個 Study 的完整流程進度。
    
    Attributes
    ----------
    study_uid : OrthancID
        Study UID，必須提供。
    series_uid : OrthancID, optional
        Series UID，Series 維度查詢時提供。
    study_id : str, optional
        醫院 HIS 系統的 Study 識別碼。
    ope_no : list[OpeNo]
        此 Study/Series 經過的所有 ope_no 清單。
        按時序排列，表示狀態轉遷歷史。
    result_data : list, optional
        對應 ope_no 清單的結果資料清單。
    params_data : list, optional
        對應 ope_no 清單的參數資料清單。
    
    Notes
    -----
    此模型設置 from_attributes=True，允許從資料庫查詢結果
    直接初始化。
    
    Examples
    --------
    查詢 Study 的完整流程：
    
    >>> status = StydySeriesOpeNoStatus(
    ...     study_uid="abc-123",
    ...     ope_no=["100.020", "100.050", "100.100", "200.150", "200.200"],
    ...     result_data=[None, None, None, None, {"path": "/nifti/result.nii.gz"}],
    ...     params_data=[None, None, None, {"format": "nifti"}, None]
    ... )
    
    查詢 Series 的流程：
    
    >>> series_status = StydySeriesOpeNoStatus(
    ...     study_uid="abc-123",
    ...     series_uid="def-456",
    ...     ope_no=["100.025", "100.055", "100.095", "200.155", "200.195"]
    ... )
    """

    model_config = ConfigDict(from_attributes=True)
    
    study_uid: OrthancID = Field(
        ...,
        description="DICOM Study UID"
    )
    series_uid: Optional[OrthancID] = Field(
        default=None,
        description="DICOM Series UID（可選）"
    )
    study_id: Optional[str] = Field(
        default=None,
        description="醫院 HIS Study 識別碼"
    )
    ope_no: List[OpeNo] = Field(
        ...,
        description="所有操作編號清單（按時序）"
    )
    result_data: Optional[List[Any]] = Field(
        default=None,
        description="對應的結果資料清單"
    )
    params_data: Optional[List[Any]] = Field(
        default=None,
        description="對應的參數資料清單"
    )
