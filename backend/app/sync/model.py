"""
Database models used by the sync service layer.

此模組提供 DICOM 同步服務的所有資料庫模型，包括事件紀錄、配置管理及 Study 鏈結追蹤。
所有欄位皆對應到 DICOM 同步事件儲存表，為方便判讀在此補充欄位語意。

Models
------
DCOPConfModel
    維護 DICOM tool 對照設定，將狀態碼映射到操作資訊（ope_no）。
DCOPEventModel
    儲存 DICOM 同步過程中發生的每一筆事件。
StudyPrevLinkModel
    維護 Study 與前一個 Study 的鏈結關係。

Notes
-----
此模組被 DCOPEventDicomService 和相關路由使用，用於追蹤 DICOM 同步流程的生命週期。
所有時間戳記預設使用 UTC 時間。
"""

from typing import Dict, Optional

from advanced_alchemy.extensions.fastapi import base
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import Column, String, Integer, DateTime, JSON
from datetime import datetime, timezone


def utc_now() -> datetime:
    """返回當前 UTC 時間（時區感知）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DCOPConfModel(base.DefaultBase):
    """
    DICOM 工具操作設定模型，維護狀態碼與操作資訊的對應關係。
    
    此表格用於管理系統中所有 DICOM 工具的操作配置，將流程狀態碼對應到可讀的
    操作名稱（ope_no）。支援多個工具並行運作，每個工具可定義自己的狀態流。
    
    Attributes
    ----------
    tool_id : str
        DICOM 工具識別碼，例如 "DICOM_TOOL"、"NIFTI_TOOL"。
        與 ope_no 組合成複合主鍵，確保每個工具的 ope_no 唯一。
    ope_no : str
        操作編號，格式為 xxx.xxx（7 碼），用於在 UI 和日誌中識別流程步驟。
        與 tool_id 組合為主鍵。
    ope_name : str, optional
        人類可讀的操作名稱，例如 "Study Transfer Complete"。
        用於 UI 顯示和日誌記錄。
    status_code : str
        對應的流程狀態碼，用於內部狀態機管理。
        建立索引以加速狀態查詢。
    description : str, optional
        操作的詳細說明，用於 UI 顯示或後台文檔。
    active : int
        啟用狀態（0 或 1），預設為 1（啟用）。
        禁用的配置不會被應用程式使用。
    rec_time : datetime, optional
        記錄時間，用於審計追蹤。
    create_time : datetime
        配置建立時間，預設為 UTC 現在時間。
    update_time : datetime
        最後修改時間，預設為 UTC 現在時間，修改時自動更新。
    
    Examples
    --------
    查詢特定工具的操作配置：
    
    >>> conf = await session.get(DCOPConfModel, ("DICOM_TOOL", "100.100"))
    >>> print(f"Operation: {conf.ope_name}")
    
    Notes
    -----
    複合主鍵設計允許多個工具共用相同的 ope_no，但同一工具內 ope_no 必須唯一。
    """

    __tablename__ = "dcop_conf_bt"

    # 複合主鍵：工具代碼 + ope_no 唯一索引，每筆紀錄對應一種操作狀態
    tool_id = Column(
        String(32),
        primary_key=True,
        index=True,
        nullable=False,
        comment="工具識別碼，例如 DICOM_TOOL、NIFTI_TOOL"
    )
    ope_no = Column(
        String(7),
        primary_key=True,
        nullable=False,
        comment="操作編號，格式 xxx.xxx"
    )
    
    # 基本信息欄位
    ope_name = Column(
        String(36),
        nullable=True,
        comment="人類可讀的操作名稱"
    )
    status_code = Column(
        String(32),
        index=True,
        nullable=True,
        comment="對應的流程狀態碼"
    )
    description = Column(
        String,
        nullable=True,
        comment="操作詳細說明，用於 UI 或文檔"
    )
    
    # 管理欄位
    active = Column(
        Integer,
        default=1,
        nullable=False,
        comment="是否啟用（0=禁用, 1=啟用）"
    )
    rec_time = Column(
        DateTime,
        nullable=True,
        comment="記錄時間，用於審計追蹤"
    )
    create_time = Column(
        DateTime,
        default=utc_now,
        nullable=False,
        comment="配置建立時間"
    )
    update_time = Column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
        comment="最後修改時間"
    )


class DCOPEventModel(base.DefaultBase):
    """
    DICOM 同步事件紀錄模型，儲存同步流程中發生的每一筆事件。
    
    此表格是 DICOM 同步系統的核心，記錄從 Study 接收、Series 轉檔、
    推論執行等整個生命週期中的所有重要事件。每筆紀錄都代表一個特定時刻的
    系統狀態轉遷。
    
    Attributes
    ----------
    VsPrimaryKey : str
        複合主鍵，由 tool_id、status、ID 和時間戳組成。
        格式：{tool_id}_{status}_{study_or_series_uid}_{timestamp}
    tool_id : str
        觸發此事件的來源工具，例如 "DICOM_TOOL"、"NIFTI_TOOL"。
        建立索引以支援工具維度查詢。
    study_uid : str
        事件所屬的 DICOM Study UID，用於跨工具追蹤同一 Study。
        建立索引以加速查詢。
    series_uid : str, optional
        事件所屬的 Series UID，若為 Study 維度事件則為 None。
        建立索引以支援 Series 維度查詢。
    study_id : str, optional
        醫院或 PACS 系統的 Study 識別碼，不同於 DICOM UID。
        用於外部系統整合。
    event_cate : int, optional
        事件類別編號，用於事件分類和過濾。
    code_name : str
        狀態碼，由 DCOPConfModel 中定義的狀態碼組成。
        例如 "100.100"、"200.200" 等。
    code_desc : str, optional
        狀態碼的簡短描述，用於快速理解事件性質。
    params_data : dict, optional
        JSON 格式的參數資料，儲存呼叫此事件時傳入的參數。
        用於事件重現和審計。
    result_data : dict, optional
        JSON 格式的結果資料，儲存事件執行的結果和輸出。
        用於後續流程追蹤。
    ope_no : str
        操作編號（xxx.xxx 格式），與 DCOPConfModel 關聯。
        建立索引以支援按操作查詢。
    ope_name : str, optional
        操作的可讀名稱，冗余儲存以加速查詢（避免 JOIN）。
    claim_time : datetime
        事件聲稱發生的時間（來源系統報告的時間）。
        用於檢測時鐘偏差。
    rec_time : datetime
        事件被系統接收的時間。
        用於測量傳輸延遲。
    create_time : datetime
        事件紀錄在本系統建立的時間。
    update_time : datetime, optional
        事件紀錄最後修改的時間。
        預設為 UTC 現在時間，修改時自動更新。
    
    Methods
    -------
    create_event(study_uid, status, tool_id, series_uid, session)
        使用狀態碼建立新的事件紀錄。
    create_event_ope_no(tool_id, study_uid, series_uid, study_id, ope_no, result_data, params_data, session)
        根據 ope_no 和工具資訊建立事件。
    
    Notes
    -----
    此模型遵循 Good Taste 設計：
    - 複合主鍵消除了對唯一 ID 的特殊處理
    - 用資料結構（JSON 欄位）替代特殊情況
    - 建立適當的索引平衡查詢性能和儲存成本
    
    Examples
    --------
    建立 Study 維度事件：
    
    >>> event = await DCOPEventModel.create_event(
    ...     study_uid="abc-123",
    ...     status="100.100",
    ...     tool_id="DICOM_TOOL",
    ...     session=session
    ... )
    
    建立 Series 維度事件：
    
    >>> event = await DCOPEventModel.create_event_ope_no(
    ...     tool_id="NIFTI_TOOL",
    ...     study_uid="abc-123",
    ...     series_uid="def-456",
    ...     study_id="HIS_STUDY_001",
    ...     ope_no="200.200",
    ...     result_data={"output": "path/to/nifti.nii.gz"},
    ...     params_data={"format": "nifti"},
    ...     session=session
    ... )
    """

    __tablename__ = "dcop_event_bt"

    # 主鍵和基本識別欄位
    VsPrimaryKey = Column(
        String(128),
        primary_key=True,
        index=True,
        name="vsprimarykey",
        nullable=False,
        comment="複合主鍵：{tool_id}_{status}_{uid}_{timestamp}"
    )
    tool_id = Column(
        String(32),
        nullable=False,
        index=True,
        comment="觸發事件的工具代碼"
    )
    study_uid = Column(
        String(128),
        nullable=False,
        index=True,
        comment="事件所屬的 DICOM Study UID"
    )
    series_uid = Column(
        String(128),
        nullable=True,
        index=True,
        comment="Series 維度事件時提供，否則為 None"
    )
    study_id = Column(
        String(128),
        nullable=True,
        index=True,
        comment="醫院 HIS 系統的 Study 識別碼"
    )

    # 事件相關欄位
    event_cate = Column(
        Integer,
        nullable=True,
        comment="事件類別編號"
    )
    code_name = Column(
        String(32),
        nullable=False,
        comment="狀態碼，例如 100.100、200.200"
    )
    code_desc = Column(
        String(64),
        nullable=True,
        comment="狀態碼簡短描述"
    )

    # 數據欄位
    params_data = Column(
        JSON,
        nullable=True,
        comment="JSON 格式參數資料"
    )
    result_data = Column(
        JSON,
        nullable=True,
        comment="JSON 格式結果資料"
    )

    # 操作相關欄位
    ope_no = Column(
        String(7),
        nullable=False,
        index=True,
        comment="操作編號 xxx.xxx"
    )
    ope_name = Column(
        String(36),
        nullable=True,
        comment="操作名稱（冗余儲存）"
    )

    # 時間欄位
    claim_time = Column(
        DateTime,
        nullable=False,
        default=utc_now,
        comment="事件聲稱時間（來源系統報告）"
    )
    rec_time = Column(
        DateTime,
        nullable=False,
        default=utc_now,
        comment="事件接收時間"
    )
    create_time = Column(
        DateTime,
        nullable=False,
        default=utc_now,
        comment="事件紀錄建立時間"
    )
    update_time = Column(
        DateTime,
        nullable=True,
        default=utc_now,
        onupdate=utc_now,
        comment="最後修改時間"
    )

    # 額外方法
    def __repr__(self) -> str:
        """
        提供更易讀的 log 內容。
        
        Returns
        -------
        str
            格式化的事件表示。
        """
        return (
            f"<DicomEvent(VsPrimaryKey={self.VsPrimaryKey}, "
            f"study_uid='{self.study_uid}', status='{self.code_name}')>"
        )

    @classmethod
    async def create_event(
        cls,
        study_uid: str,
        status: str,
        tool_id: str = "DICOM_TOOL",
        series_uid: Optional[str] = None,
        session: Optional[Session | AsyncSession] = None,
    ) -> "DCOPEventModel":
        """
        使用狀態碼建立新的事件紀錄（Study 或 Series 維度）。
        
        此方法實現了流程狀態自動尋址：根據提供的狀態碼查詢配置表，
        自動取得對應的 ope_no 和 ope_name，消除了在業務邏輯中
        硬編碼操作碼的需要。
        
        Parameters
        ----------
        study_uid : str
            DICOM Study UID，事件必須對應到某個 Study。
        status : str
            目標狀態碼，將在 DCOPConfModel 中查詢對應的 ope_no。
            例如 "100.100"、"200.200" 等。
        tool_id : str, optional
            觸發此事件的來源工具代碼，預設 "DICOM_TOOL"。
            用於在配置表中查詢正確的 ope_no。
        series_uid : str, optional
            Series 維度事件時帶入，否則建立 Study 維度事件。
            若提供，事件將記錄特定 Series 的狀態。
        session : Session | AsyncSession
            Active SQLAlchemy session，必要。
            用於查詢 DCOPConfModel 配置和建立新紀錄。
        
        Returns
        -------
        DCOPEventModel
            新建立的事件紀錄實例（未自動提交）。
        
        Raises
        ------
        ValueError
            如果 session 為 None。
        ValueError
            如果配置表中不存在對應的 tool_id 和 status 組合。
        
        Examples
        --------
        建立 Study 完成事件：
        
        >>> event = await DCOPEventModel.create_event(
        ...     study_uid="abc-123",
        ...     status="100.100",
        ...     tool_id="DICOM_TOOL",
        ...     session=session
        ... )
        >>> session.add(event)
        >>> await session.commit()
        
        建立 Series 轉檔中事件：
        
        >>> event = await DCOPEventModel.create_event(
        ...     study_uid="abc-123",
        ...     status="200.150",
        ...     tool_id="NIFTI_TOOL",
        ...     series_uid="def-456",
        ...     session=session
        ... )
        
        Notes
        -----
        Good Taste 設計：用狀態碼驅動而非特殊情況判斷
        
        1. 消除 if/else：無論是 Study 還是 Series，邏輯統一
        2. 資料結構驅動：查詢配置表取得對應的 ope_no
        3. Early return：不必要的條件判斷已消除
        """
        # 檢查 session 是否提供
        if session is None:
            raise ValueError("Database session is required to create event")

        # 構建查詢：根據 tool_id 和 status 在配置表中查詢操作資訊
        conf_query = select(DCOPConfModel.ope_no, DCOPConfModel.ope_name).where(
            DCOPConfModel.tool_id == tool_id,
            DCOPConfModel.status_code == status,
            DCOPConfModel.active == 1,  # 只查詢啟用的配置
        )

        # 執行配置查詢
        result = await session.execute(conf_query)
        conf_result = result.first()

        # 驗證配置存在
        if conf_result is None:
            raise ValueError(
                f"No configuration found for tool_id: {tool_id}, status: {status}"
            )
        
        # 解包查詢結果
        ope_no, ope_name = conf_result

        # 生成唯一的複合主鍵
        # 格式：{tool_id}_{status}_{uid}_{timestamp}
        uid = series_uid if series_uid is not None else study_uid
        composite_key = (
            f"{tool_id}_{status}_{uid}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )

        # 建立事件紀錄實例
        obj = cls(
            VsPrimaryKey=composite_key,
            tool_id=tool_id,
            study_uid=study_uid,
            series_uid=series_uid,  # Study 維度事件時為 None
            code_name=status,
            ope_no=ope_no,
            ope_name=ope_name,
            claim_time=datetime.now(timezone.utc),
            rec_time=datetime.now(timezone.utc),
        )

        return obj

    @classmethod
    async def create_event_ope_no(
        cls,
        tool_id: str,
        study_uid: str,
        series_uid: str,
        study_id: str,
        ope_no: str,
        result_data: Dict[str, str],
        params_data: Dict[str, str],
        session: Optional[Session | AsyncSession] = None,
    ) -> "DCOPEventModel":
        """
        根據 ope_no 和工具資訊建立事件，通常用於外部系統回報。
        
        此方法與 create_event 的區別在於：它接收已知的 ope_no，
        而非狀態碼。這適用於外部系統（如 NIFTI_TOOL）直接回報
        自己的操作結果。
        
        Parameters
        ----------
        tool_id : str
            來源工具代號，例如 "NIFTI_TOOL"。
        study_uid : str
            事件所屬的 DICOM Study UID。
        series_uid : str
            事件所屬的 Series UID。若為 None，則建立 Study 維度事件。
        study_id : str
            醫院或 PACS 系統的 Study 識別碼。
        ope_no : str
            已知的操作狀態碼，格式 xxx.xxx。
            必須在 DCOPConfModel 中存在對應的配置。
        result_data : dict
            流程執行結果的 JSON 資料。
            例如 {"output_path": "/path/to/result.nii.gz", "status": "success"}
        params_data : dict
            呼叫此流程時傳入的參數 JSON 資料。
            例如 {"format": "nifti", "compress": True}
        session : Session | AsyncSession
            Active SQLAlchemy session，必要。
            用於查詢配置和建立紀錄。
        
        Returns
        -------
        DCOPEventModel
            新建立的事件紀錄實例（未自動提交）。
        
        Raises
        ------
        ValueError
            如果 session 為 None。
        ValueError
            如果配置表中不存在對應的 tool_id 和 ope_no 組合。
        
        Examples
        --------
        NIFTI_TOOL 回報轉檔完成：
        
        >>> event = await DCOPEventModel.create_event_ope_no(
        ...     tool_id="NIFTI_TOOL",
        ...     study_uid="abc-123",
        ...     series_uid="def-456",
        ...     study_id="HIS_001",
        ...     ope_no="200.200",
        ...     result_data={
        ...         "output_path": "/data/nifti/abc-123_def-456.nii.gz",
        ...         "success": True
        ...     },
        ...     params_data={
        ...         "format": "nifti",
        ...         "compress": True
        ...     },
        ...     session=session
        ... )
        >>> session.add(event)
        >>> await session.commit()
        
        Notes
        -----
        此方法用於集成外部工具的回報。外部工具無需知道系統的
        狀態碼，只需提供 ope_no 和結果資料。
        """
        # 檢查 session
        if session is None:
            raise ValueError("Database session is required to create event")

        # 根據 ope_no 查詢配置，反向解析狀態碼
        conf_query = select(
            DCOPConfModel.status_code,
            DCOPConfModel.ope_name,
        ).where(
            DCOPConfModel.tool_id == tool_id,
            DCOPConfModel.ope_no == ope_no,
            DCOPConfModel.active == 1,  # 只查詢啟用的配置
        )
        
        # 執行查詢
        result = await session.execute(conf_query)
        conf_result = result.first()

        # 驗證配置存在
        if conf_result is None:
            raise ValueError(
                f"No configuration found for tool_id: {tool_id}, ope_no: {ope_no}"
            )
        
        # 解包查詢結果
        status, ope_name = conf_result

        # 生成複合主鍵
        # 使用 series_uid 或 study_uid 作為基礎
        uid = series_uid if series_uid is not None else study_uid
        composite_key = (
            f"{tool_id}_{status}_{uid}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )

        # 建立事件紀錄
        obj = cls(
            VsPrimaryKey=composite_key,
            tool_id=tool_id,
            study_uid=study_uid,
            series_uid=series_uid,
            code_name=status,
            ope_no=ope_no,
            ope_name=ope_name,
            study_id=study_id,
            result_data=result_data,      # 儲存外部工具的結果
            params_data=params_data,      # 儲存傳入的參數
            claim_time=datetime.now(timezone.utc),
            rec_time=datetime.now(timezone.utc),
        )
        
        return obj


class StudyPrevLinkModel(base.DefaultBase):
    """
    Study 鏈結追蹤模型，維護 Study 與前一個 Study 的時序關係。
    
    在某些醫療場景中，同一患者的多個 Study 之間可能存在邏輯關係。
    例如，基線掃描、隨訪掃描、對比研究等。此表格用於記錄這些
    時序關係，使後續分析和追蹤成為可能。
    
    Attributes
    ----------
    id : int
        自增主鍵，用於內部識別。
    study_uid : str
        當前 Study 的 DICOM UID，具有唯一約束。
        確保每個 Study 最多只能鏈結到一個前驅 Study。
    prev_study_uid : str
        前一個 Study 的 DICOM UID。
        可以重複（多個 Study 可指向同一前驅）。
    created_at : datetime
        鏈結建立的時間，預設為 UTC 現在時間。
    
    Notes
    -----
    設計思想：
    - 單向鏈結（只記錄前驅）減少了雙向更新的複雜性
    - 唯一約束確保每個 Study 最多有一個前驅，形成清晰的時序鏈
    - 支援長鏈：可通過遞迴查詢追蹤整個 Study 序列
    
    Examples
    --------
    建立 Study 鏈結：
    
    >>> link = StudyPrevLinkModel(
    ...     study_uid="current-123",
    ...     prev_study_uid="baseline-001"
    ... )
    >>> session.add(link)
    >>> await session.commit()
    
    查詢 Study 的前驅：
    
    >>> link = await session.get(
    ...     StudyPrevLinkModel,
    ...     {"study_uid": "current-123"}
    ... )
    >>> if link:
    ...     print(f"Previous study: {link.prev_study_uid}")
    """

    __tablename__ = "study_prev_link"

    # 主鍵
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
        comment="自增主鍵"
    )
    
    # Study 識別欄位
    study_uid = Column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
        comment="當前 Study UID（唯一）"
    )
    prev_study_uid = Column(
        String(128),
        nullable=False,
        comment="前一個 Study UID"
    )
    
    # 時間欄位
    created_at = Column(
        DateTime,
        nullable=False,
        default=utc_now,
        comment="鏈結建立時間"
    )

    def __repr__(self) -> str:
        """
        提供易讀的表示形式。
        
        Returns
        -------
        str
            格式化的 Study 鏈結表示。
        """
        return (
            f"<StudyPrevLink(study_uid='{self.study_uid}', "
            f"prev='{self.prev_study_uid}')>"
        )