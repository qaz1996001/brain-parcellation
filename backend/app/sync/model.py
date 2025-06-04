from typing import Dict

from advanced_alchemy.extensions.fastapi import base
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON
from datetime import datetime


# 對應的 DCOPConfModel
class DCOPConfModel(base.DefaultBase):
    __tablename__ = 'dcop_conf_bt'

    tool_id     = Column(String(32), primary_key=True, index=True)
    ope_no      = Column(String(7), primary_key=True)
    ope_name    = Column(String(36))
    status_code = Column(String(32), index=True)
    description = Column(String)
    active      = Column(Integer, default=1)
    rec_time    = Column(DateTime)
    create_time = Column(DateTime, default=datetime.utcnow)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)



class DCOPEventModel(base.DefaultBase):
    __tablename__ = 'dcop_event_bt'

    # 主鍵和基本識別欄位
    VsPrimaryKey = Column(String(128), primary_key=True, index=True,name='vsprimarykey')
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
    claim_time  = Column(DateTime, nullable=False, default=datetime.utcnow)
    rec_time    = Column(DateTime, nullable=False, default=datetime.utcnow)
    create_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    update_time = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)



    # 額外方法
    def __repr__(self):
        return f"<DicomEvent(VsPrimaryKey = {self.VsPrimaryKey},study_uid='{self.study_uid}', status='{self.code_name}')>"

    @classmethod
    async def create_event(
        cls,
        study_uid: str,
        status: str,
        tool_id: str = 'DICOM_TOOL',
        series_uid: str = None,
        session: Session | AsyncSession = None
    ):
        """
        快速創建事件的類方法
        """
        # 如果沒有提供 session，拋出異常
        if session is None:
            raise ValueError("Database session is required to create event")

        # 查詢配置表，獲取 ope_no 和 ope_name
        conf_query = select(
            DCOPConfModel.ope_no,
            DCOPConfModel.ope_name
        ).where(
            DCOPConfModel.tool_id == tool_id,
            DCOPConfModel.status_code == status,
            DCOPConfModel.active == 1
        )

        # 執行查詢
        result = await session.execute(conf_query)
        conf_result = result.first()
        # await db.execute(query)
        # 如果沒有找到對應的配置
        if conf_result is None:
            raise ValueError(f"No configuration found for tool_id: {tool_id}, status: {status}")
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
                rec_time=datetime.utcnow())
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
                rec_time=datetime.utcnow())

        return obj

    @classmethod
    async def create_event_ope_no(
            cls,
            tool_id    :str,
            study_uid  :str,
            series_uid :str,
            study_id   :str,
            ope_no     :str ,
            result_data:Dict[str,str],
            params_data:Dict[str,str],
            session: Session | AsyncSession = None
    ):
        """
        快速創建事件的類方法
        """
        # 如果沒有提供 session，拋出異常
        if session is None:
            raise ValueError("Database session is required to create event")

        # 查詢配置表，獲取 ope_no 和 ope_name
        conf_query = select(
            DCOPConfModel.status_code,DCOPConfModel.ope_name,
        ).where(
            DCOPConfModel.tool_id == tool_id,
            DCOPConfModel.ope_no==ope_no,
            DCOPConfModel.active == 1
        )

        # 執行查詢
        result = await session.execute(conf_query)
        conf_result = result.first()
        # await db.execute(query)
        # 如果沒有找到對應的配置
        if conf_result is None:
            raise ValueError(f"No configuration found for tool_id: {tool_id}, ope_no: {ope_no}")
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
                study_id = study_id,
                result_data=result_data,
                params_data=params_data,
                claim_time=datetime.utcnow(),
                rec_time=datetime.utcnow())
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
                rec_time=datetime.utcnow())

        return obj
