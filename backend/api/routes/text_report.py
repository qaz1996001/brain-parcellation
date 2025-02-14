import io
import re
from typing import List

import httpx
import orjson
import pandas as pd
from fastapi import APIRouter, UploadFile, Request, HTTPException
from fastapi_pagination.api import create_page
from sqlalchemy_filters import apply_filters
from starlette import status

from app.core import SessionDep
from app.core.paginate import paginate_items
from app.model.patient import PatientModel
from app.model.study import TextReportModel, StudyModel,TextReportRawModel
from app.schema import base as base_schema
from app.schema import text_report as text_report_schema
from ..utils import get_model_by_field, get_regexp, get_regexp_filter, get_orther_filter

router = APIRouter()

pattern_impression_str = re.compile(r'(?i:impression\s?:?|imp:?|conclusions?:?)')
pattern_depiction = re.compile(r'Summary :\n(.*)(\n.+)醫師', re.DOTALL)

pattern_impression_str = re.compile(r'(?i:impression\s?:?|imp:?|conclusions?:?)')
pattern_depiction = re.compile(r'Summary :\n(.*)(\n.+)醫師', re.DOTALL)


def get_impression_by_text(x):
    global pattern_depiction, pattern_impression_str
    depiction_match = pattern_depiction.search(x)
    if depiction_match:
        depiction = depiction_match.group(1)
        result_impression_str = pattern_impression_str.split(depiction)
        if len(result_impression_str) > 0:
            return result_impression_str[-1]
        else:
            return ''


def get_text_report(text_report_items_list: List):
    response_list = []
    for text_report_items in text_report_items_list:
        # text_report = text_report_items[0]
        text = text_report_items[-1]
        if text:
            impression = get_impression_by_text(text)
        else:
            text = ""
            impression = ""
        response_list.append(text_report_schema.TextReportOut(patient_id=text_report_items[0],
                                                              gender=text_report_items[1],
                                                              accession_number=text_report_items[2],
                                                              study_description=text_report_items[3],
                                                              text=text,
                                                              impression=impression
                                                              ))
    return response_list

def get_text_report2(text_report_items_list: List):
    response_list = []
    for text_report_items in text_report_items_list:
        # text_report = text_report_items[0]
        text = text_report_items[-1]
        if text:
            impression = get_impression_by_text(text)
        else:
            text = ""
            impression = ""
        response_list.append(text_report_schema.TextReportOut2(accession_number=text_report_items[0],
                                                               text=text,
                                                               impression=impression
                                                              ))
    return response_list


@router.post("/query")
async def post_text_report_query(filter_schema: base_schema.FilterSchema,
                                 session: SessionDep) -> text_report_schema.TextReportPage[
    text_report_schema.TextReportOut]:
    filter_ = filter_schema.dict()['filter_']
    filter_ = get_model_by_field(filter_, text_report_schema.field_model)
    query = (session.query(PatientModel.patient_id, PatientModel.gender,
                           TextReportModel.accession_number,
                           StudyModel.study_description,
                           TextReportModel.text, ).
             join(StudyModel, TextReportModel.accession_number == StudyModel.accession_number).
             join(PatientModel, StudyModel.patient_uid == PatientModel.uid))
    if len(filter_) > 0:
        orther_filter = list(filter(get_orther_filter, filter_))
        regexp_filter = list(filter(get_regexp_filter, filter_))
        regexp_list = get_regexp(regexp_filter)
        filtered_query = apply_filters(query, orther_filter)
        filtered_query = filtered_query.filter(*regexp_list)
    else:
        filtered_query = query
    filtered_query = filtered_query.filter(PatientModel.deleted_at.is_(None),
                                           StudyModel.deleted_at.is_(None),
                                           TextReportModel.deleted_at.is_(None),
                                           )
    text_report_items_list, total, raw_params, params = paginate_items(session, filtered_query)
    response_list = get_text_report(text_report_items_list)

    page: text_report_schema.TextReportPage[text_report_schema.TextReportOut] = create_page(response_list, total,
                                                                                            params)
    return page


@router.post("/query111")
async def post_text_report_query111(filter_schema: base_schema.FilterSchema,
                                 session: SessionDep) -> text_report_schema.TextReportPage[
    text_report_schema.TextReportOut2]:
    filter_ = filter_schema.dict()['filter_']
    filter_ = get_model_by_field(filter_, text_report_schema.raw_field_model)
    query = (session.query(TextReportRawModel.accession_number,
                           TextReportRawModel.text, ))
    if len(filter_) > 0:
        orther_filter = list(filter(get_orther_filter, filter_))
        regexp_filter = list(filter(get_regexp_filter, filter_))
        regexp_list = get_regexp(regexp_filter)
        filtered_query = apply_filters(query, orther_filter)
        filtered_query = filtered_query.filter(*regexp_list)
    else:
        filtered_query = query
    filtered_query = filtered_query.filter(
                                           TextReportRawModel.deleted_at.is_(None),
                                           )
    text_report_items_list, total, raw_params, params = paginate_items(session, filtered_query)
    response_list = get_text_report2(text_report_items_list)

    page: text_report_schema.TextReportPage[text_report_schema.TextReportOut2] = create_page(response_list, total,
                                                                                            params)
    return page



@router.post("/query_text_report_by_excel")
async def query_text_report_by_excel(file: UploadFile,
                                    request: Request,
                                    session: SessionDep):
    file_path = file.filename
    if file_path.endswith('xlsx') or file_path.endswith('xls'):
        file_obj = io.BytesIO(await file.read())
        file_obj.seek(0)
        df = pd.read_excel(file_obj, dtype=str)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,)

    query_list = session.execute(session.query(TextReportRawModel.accession_number,TextReportRawModel.text).
             filter(TextReportRawModel.deleted_at.is_(None),
                    TextReportRawModel.accession_number.in_(df['AccessionNumber'].to_list()),
                    )).all()
    print(query_list)
    return ""
    # return FileResponse(file.file)





@router.post("/unischedule_test", description="""

file:是unischedule匯出的excel <br>

欄位要有：【狀態,病歷號,姓名,性別,年齡,生日,來源,檢查室,申請單號,檢查項目,檢查描述,開單日期/時間,儀器,檢查儀器,報到時間,檢查流水號,排程日期,報告認證時間,認證醫師】

""")
async def post_text_report(file: UploadFile,
                           request: Request,
                           session: SessionDep):
    file_path = file.filename
    if file_path.endswith('xlsx'):
        file_obj = io.BytesIO(await file.read())
        file_obj.seek(0)
        df = pd.read_excel(file_obj, dtype={'病歷號': str, '申請單號': str})
        df['年齡'] = df['年齡'].astype(int)
        # df1 = df[['病歷號','性別','生日','申請單號','檢查描述','報到時間']].copy()
        # df1.columns = ['patient_id', 'gender', 'birth_date', 'accession_number', 'study_description','study_datetime']
        df1 = df[['病歷號', '性別', '申請單號', '檢查描述', '報到時間']].copy()
        df1.columns = ['patient_id', 'gender', 'accession_number', 'study_description', 'study_datetime']
        df1['study_datetime'] = pd.to_datetime(df1['study_datetime'])
        df1['study_date'] = df1['study_datetime'].dt.date
        df1['study_time'] = df1['study_datetime'].dt.time
        df1['birth_date'] = df1['study_datetime'] - df['年齡'].map(lambda x: pd.Timedelta(x * 365, 'days'))
        df2 = df1[
            ['patient_id', 'gender', 'birth_date', 'accession_number', 'study_description', 'study_date', 'study_time']]
        df2['orthanc_patient_ID'] = None
        df2['birth_date'] = df1['birth_date'].dt.date
        json_data = orjson.dumps(df2.to_dict('records'))
        async with httpx.AsyncClient() as client:
            res = await client.post('http://127.0.0.1:8800/api/v1/study/patient', content=json_data, timeout=20)
        print('status_code', res.status_code)
        return {}
    else:
        return {}
