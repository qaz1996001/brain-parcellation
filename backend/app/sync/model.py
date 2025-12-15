"""Database models used by the sync service layer.

所有欄位皆對應到 DICOM 同步事件儲存表，為方便判讀在此補充欄位語意。
"""

from typing import Dict

from advanced_alchemy.extensions.fastapi import base
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import Column, String, Integer, DateTime, JSON
from datetime import datetime


# 對應的 DCOPConfModel
class DCOPConfModel(base.DefaultBase):
    """維護 dicom tool 對照設定，用於將狀態碼映射到 ope 資訊。"""

    __tablename__ = "dcop_conf_bt"

    #: 工具代碼 + ope_no 是唯一索引，每筆紀錄只會對應到一種狀態
    tool_id = Column(String(32), primary_key=True, index=True)
    ope_no = Column(String(7), primary_key=True)
    ope_name = Column(String(36))  # 人類可讀的操作名稱
    status_code = Column(String(32), index=True)  # 對應的流程狀態碼
    description = Column(String)  # UI 顯示或後台說明
    active = Column(Integer, default=1)  # 是否仍啟用
    rec_time = Column(DateTime)
    create_time = Column(DateTime, default=datetime.utcnow)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DCOPEventModel(base.DefaultBase):
    """儲存 DICOM 同步過程中發生的每一筆事件。"""

    __tablename__ = "dcop_event_bt"

    # 主鍵和基本識別欄位
    VsPrimaryKey = Column(
        String(128), primary_key=True, index=True, name="vsprimarykey"
    )
    tool_id = Column(String(32), nullable=False, index=True)
    study_uid = Column(String(128), nullable=False, index=True)
    series_uid = Column(String(128), nullable=True, index=True)
    study_id = Column(String(128), nullable=True, index=True)

    # 事件相關欄位
    event_cate = Column(Integer, nullable=True)  # 事件類別
    code_name = Column(String(32), nullable=False)
    code_desc = Column(String(64), nullable=True)

    # 數值和文本欄位
    params_data = Column(JSON, nullable=True)
    result_data = Column(JSON, nullable=True)

    # 操作相關欄位
    ope_no = Column(String(7), nullable=False, index=True)
    ope_name = Column(String(36), nullable=True)

    # 時間欄位
    claim_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    rec_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    create_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    update_time = Column(
        DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # 額外方法
    def __repr__(self):
        """提供更易讀的 log 內容。"""
        return f"<DicomEvent(VsPrimaryKey = {self.VsPrimaryKey},study_uid='{self.study_uid}', status='{self.code_name}')>"

    @classmethod
    async def create_event(
        cls,
        study_uid: str,
        status: str,
        tool_id: str = "DICOM_TOOL",
        series_uid: str = None,
        session: Session | AsyncSession = None,
    ):
        """
        使用狀態碼建立新的事件紀錄。

        Args:
            study_uid: DICOM study UID。
            status: 目標狀態碼，會去 lookup 取得對應的 ope_no。
            tool_id: 觸發此事件的來源工具。
            series_uid: 指定 series 時帶入，否則以 study 維度建立。
            session: Active SQLAlchemy session（必要，否則無法查詢設定）。
        """
        # 如果沒有提供 session，拋出異常
        if session is None:
            raise ValueError("Database session is required to create event")

        # 查詢配置表，獲取 ope_no 和 ope_name
        conf_query = select(DCOPConfModel.ope_no, DCOPConfModel.ope_name).where(
            DCOPConfModel.tool_id == tool_id,
            DCOPConfModel.status_code == status,
            DCOPConfModel.active == 1,
        )

        # 執行查詢
        result = await session.execute(conf_query)
        conf_result = result.first()
        # await db.execute(query)
        # 如果沒有找到對應的配置
        if conf_result is None:
            raise ValueError(
                f"No configuration found for tool_id: {tool_id}, status: {status}"
            )
        ope_no, ope_name = conf_result

        if series_uid is not None:
            obj = cls(
                VsPrimaryKey=f"{tool_id}_{status}_{series_uid}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                tool_id=tool_id,
                study_uid=study_uid,
                series_uid=series_uid,
                code_name=status,
                ope_no=ope_no,
                ope_name=ope_name,
                claim_time=datetime.utcnow(),
                rec_time=datetime.utcnow(),
            )
        else:
            obj = cls(
                VsPrimaryKey=f"{tool_id}_{status}_{study_uid}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                tool_id=tool_id,
                study_uid=study_uid,
                series_uid=series_uid,
                code_name=status,
                ope_no=ope_no,
                ope_name=ope_name,
                claim_time=datetime.utcnow(),
                rec_time=datetime.utcnow(),
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
        session: Session | AsyncSession = None,
    ):
    ):
        """
        根據 ope_no/工具資訊建立事件，通常用於外部回報。

        Args:
            tool_id: 來源工具代號。
            study_uid: 事件所屬的 study。
            series_uid: 事件所屬的 series。
            study_id: 醫院或 PACS 的 study identifier。
            ope_no: 已知的操作狀態碼（xxx.xxx 格式）。
            result_data: 流程結果 payload。
            params_data: 呼叫參數 payload。
            session: Active session，查詢設定與寫入資料必須共用。
        """
        # 如果沒有提供 session，拋出異常
        if session is None:
            raise ValueError("Database session is required to create event")

        # 查詢配置表，獲取 ope_no 和 ope_name
        conf_query = select(
            DCOPConfModel.status_code,
            DCOPConfModel.ope_name,
        ).where(
            DCOPConfModel.tool_id == tool_id,
            DCOPConfModel.ope_no == ope_no,
            DCOPConfModel.active == 1,
        )
        result = await session.execute(conf_query)
        conf_result = result.first()
        # 如果沒有找到對應的配置
        if conf_result is None:
            raise ValueError(
                f"No configuration found for tool_id: {tool_id}, ope_no: {ope_no}"
            )
        status, ope_name = conf_result

        if series_uid is not None:
            obj = cls(
                VsPrimaryKey=f"{tool_id}_{status}_{series_uid}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                tool_id=tool_id,
                study_uid=study_uid,
                series_uid=series_uid,
                code_name=status,
                ope_no=ope_no,
                ope_name=ope_name,
                study_id=study_id,
                result_data=result_data,
                params_data=params_data,
                claim_time=datetime.utcnow(),
                rec_time=datetime.utcnow(),
            )
        else:
            obj = cls(
                VsPrimaryKey=f"{tool_id}_{status}_{study_uid}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                tool_id=tool_id,
                study_uid=study_uid,
                series_uid=series_uid,
                code_name=status,
                ope_no=ope_no,
                ope_name=ope_name,
                study_id=study_id,
                result_data=result_data,
                params_data=params_data,
                claim_time=datetime.utcnow(),
                rec_time=datetime.utcnow(),
            )
        return obj


class StudyPrevLinkModel(base.DefaultBase):
    """維護 study 與前一個 study 的鏈結關係。"""

    __tablename__ = "study_prev_link"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_uid = Column(String(128), nullable=False, unique=True, index=True)
    prev_study_uid = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<StudyPrevLink(study_uid='{self.study_uid}', prev='{self.prev_study_uid}')>"