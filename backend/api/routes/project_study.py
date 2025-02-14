import datetime
import io
import uuid
from typing import List,Dict
from copy import copy

import calendar

import pandas as pd
from fastapi import APIRouter, Query, UploadFile
from fastapi_pagination.api import create_page
import sqlalchemy

from sqlalchemy.dialects.postgresql import NUMERIC
from sqlalchemy import and_
from sqlalchemy.sql.expression import func,cast
from sqlalchemy_filters import apply_filters


from app.core.paginate import paginate_items
from app.core import SessionDep
from app.model.patient import PatientModel
from app.model.project import ProjectModel,ProjectStudyModel
from app.model.study import StudyModel
from app.schema.project import ProjectInput
from app.schema.project_study import ProjectStudyOutput,ProjectStudyPost,ProjectStudyAccessionNumberPost
from app.schema.base import CodePage,FilterSchema,get_model_by_field,get_group_key_by_series

router = APIRouter()

@router.get("/hello")
async def hello_world():
    return "Hello World!"


@router.post("/query")
async def post_query(filter_schema : FilterSchema,
                   session: SessionDep) -> CodePage[ProjectStudyOutput]:
    filter_ = filter_schema.dict()['filter_']
    model_field_list = get_model_by_field(filter_)
    print('model_field_list')
    print(model_field_list)
    print(filter_)
    query: Query = session.query(ProjectStudyModel, StudyModel, PatientModel) \
        .join(StudyModel, ProjectStudyModel.study_uid == StudyModel.uid) \
        .join(PatientModel, StudyModel.patient_uid == PatientModel.uid)


    extra_data_filter = list(filter(lambda x: 'extra_data.' in x['field'], filter_))
    extra_data_filter = list(map(convert_extra_data_filter_type, extra_data_filter))
    orther_filter = list(filter(lambda x: 'extra_data.' not in x['field'], filter_))

    orther_filter = get_model_by_field(orther_filter)
    extra_data_filter_sqlaichemy_not_na = list(
        map(lambda x: and_(ProjectStudyModel.extra_data.op("->>")(x['field']).op('!=')('Na')),
            extra_data_filter))

    extra_data_filter_sqlaichemy = list(
        map(lambda x: and_(cast(ProjectStudyModel.extra_data.op("->")(x['field']), NUMERIC).op(x['op'])(x['value'])),
            extra_data_filter))
    extra_data_filter_sqlaichemy_not_na.insert(0, func.jsonb_typeof(ProjectStudyModel.extra_data) == sqlalchemy.text(
        "\'object\'"))
    if filter_:
        filtered_query = apply_filters(query, orther_filter)
        filtered_query = filtered_query.filter(*extra_data_filter_sqlaichemy_not_na).filter(
            *extra_data_filter_sqlaichemy)
    else:
        filtered_query = query

    filtered_query = filtered_query.filter(PatientModel.deleted_at.is_(None),
                                           StudyModel.deleted_at.is_(None),
                                           ProjectStudyModel.deleted_at.is_(None),
                                           )
    print(filtered_query)
    items_list, total, raw_params, params = paginate_items(session, filtered_query)
    response_list = []
    for item in items_list:
        project_study = item[0]
        study = project_study.study
        study_dict = {'project_study_uid': project_study.uid,
                      'project_name': project_study.project.name}
        study_dict['patient_id'] = study.patient.patient_id
        study_dict['gender'] = study.patient.gender
        study_dict['age'] = get_age_by_study_date(study.patient.birth_date, study.study_date)
        study_dict.update(study.to_dict_view())
        study_dict['extra_data'] = project_study.extra_data
        project_study_output = ProjectStudyOutput(**study_dict)
        response_list.append(project_study_output.dict())
    df = pd.json_normalize(response_list, max_level=1)
    columns = df.columns.to_list()
    group_key = get_group_key_by_series(columns)
    group_key.update(get_extra_data_key(columns=columns))

    additional_data = {'code': 2000,
                       'key': columns,
                       'group_key': group_key,
                       'op_list': filter_op_list
                       }
    page: CodePage[ProjectStudyOutput] = create_page(response_list,
                                                     total=total,
                                                     params=params,
                                                     **additional_data)
    return page


@router.post("/study")
async def post_study(project_study_post : ProjectStudyPost,
                     session: SessionDep) -> Dict[str, str]:
    project_model = session.execute(session.query(ProjectModel).filter(ProjectModel.uid==project_study_post.project_uid)).first()[0]
    if project_model:
        result_list = session.execute(session.query(StudyModel.uid).filter(StudyModel.uid.in_(project_study_post.study_uid_list))).all()
        for result in result_list:
            if result is not None:
                project_study_model = session.execute(session.query(ProjectStudyModel).filter(ProjectStudyModel.project_uid==project_model.uid,
                                                                                              ProjectStudyModel.study_uid == result[0],
                                                                                              )).first()[0]
                if project_study_model:
                    continue
                else:
                    datetime_now = datetime.datetime.now()
                    project_study_model = ProjectStudyModel(study_uid=result[0],
                                                            project_uid=project_model.uid,
                                                            extra_data={},
                                                            created_at=datetime_now)
                    session.add(project_study_model)
            else:
                continue
        session.commit()
        return {'code': '2000'}
    else:
        return {'code': '400'}

# @router.post("/study")
@router.post("/study/uid")
async def post_study_uid(project_study_post : ProjectStudyPost,
                         session: SessionDep) -> Dict[str, str]:
    project_model = session.execute(session.query(ProjectModel).filter(ProjectModel.uid==project_study_post.project_uid)).first()[0]
    if project_model:
        result_list = session.execute(session.query(StudyModel.uid).filter(StudyModel.uid.in_(project_study_post.study_uid_list))).all()
        for result in result_list:
            if result is not None:
                project_study_model = session.execute(session.query(ProjectStudyModel).filter(ProjectStudyModel.project_uid==project_model.uid,
                                                                                              ProjectStudyModel.study_uid == result[0],
                                                                                              )).first()[0]
                if project_study_model:
                    continue
                else:
                    datetime_now = datetime.datetime.now()
                    project_study_model = ProjectStudyModel(study_uid=result[0],
                                                            project_uid=project_model.uid,
                                                            extra_data={},
                                                            created_at=datetime_now)
                    session.add(project_study_model)
            else:
                continue
        session.commit()
        return {'code': '2000'}
    else:
        return {'code': '400'}


@router.post("/study/accession_number")
async def post_study_accession_number(project_study_post : ProjectStudyAccessionNumberPost,
                                      session: SessionDep) -> Dict[str, str]:
    project_model = session.execute(session.query(ProjectModel).filter(ProjectModel.uid==project_study_post.project_uid)).first()[0]
    if project_model:
        result_list = session.execute(session.query(StudyModel.uid).filter(StudyModel.uid.in_(project_study_post.study_uid_list))).all()
        for result in result_list:
            if result is not None:
                project_study_model = session.execute(session.query(ProjectStudyModel).filter(ProjectStudyModel.project_uid==project_model.uid,
                                                                                              ProjectStudyModel.study_uid == result[0],
                                                                                              )).first()[0]
                if project_study_model:
                    continue
                else:
                    datetime_now = datetime.datetime.now()
                    project_study_model = ProjectStudyModel(study_uid=result[0],
                                                            project_uid=project_model.uid,
                                                            extra_data={},
                                                            created_at=datetime_now)
                    session.add(project_study_model)
            else:
                continue
        session.commit()
        return {'code': '2000'}
    else:
        return {'code': '400'}



@router.post("/upload/excel")
async def post_study_excel(file:UploadFile ,
                           session: SessionDep) -> List[str]:
    file_obj = io.BytesIO(await file.read())

    df = pd.read_excel(file_obj, dtype={'patient_id': str,
                                        'accession_number': str,
                                        'study_date': str,})
    df['study_date'] = pd.to_datetime(df['study_date'])
    df['project_study_uid'] = df['project_study_uid'].map(lambda x: uuid.UUID(x))
    df.fillna('', inplace=True)
    project_study_uid_list = df['project_study_uid'].to_list()
    result_list = session.execute(session.query(ProjectStudyModel).filter(ProjectStudyModel.uid.in_(project_study_uid_list))).all()
    for result in result_list:
        project_study_model = result[0]
        project_study_uid = project_study_model.uid
        temp_data = df[df['project_study_uid'] == project_study_uid]
        new_extra_data = temp_data.iloc()[0, 6:].to_dict()
        column_order = temp_data.iloc()[0, 6:].index.to_list()
        new_extra_data.update({'column_order':column_order})
        if project_study_model.extra_data:
            raw_extra_data = copy(project_study_model.extra_data)
            raw_extra_data.update(new_extra_data)
            project_study_model.extra_data = raw_extra_data
        else:
            project_study_model.extra_data = new_extra_data
        project_study_model.updated_at = datetime.datetime.now()
    session.commit()
    return df.columns.to_list()


@router.post("/upload/json")
async def post_study_json(
                           session: SessionDep) -> Dict[str, str]:
    return {'code': '2000'}


@router.post("/download")
async def post_download_study(
        session: SessionDep) -> Dict[str, str]:
    return {'code': '2000'}


@router.get("/download/infinitt/{project_name:str}")
async def get_download_infinitt(project_name:str,
                                 session: SessionDep) -> Dict[str, List[str]]:
    project             = session.query(ProjectModel).where(ProjectModel.name == project_name).first()
    if project:
        project_study_list  = session.query(ProjectStudyModel).where(ProjectStudyModel.project_uid==project.uid).all()
        accession_number_list = list(map(lambda x: x.study.accession_number, project_study_list))
        result = []
        accession_number_len = len(accession_number_list)
        for index in range((accession_number_len//50)+1):
            start_index = index*50
            end_index = start_index + 50
            result.append(','.join(accession_number_list[start_index:end_index]))
        return {'code': result}
    return {}


@router.post("/download/infinitt/{project_name:str}")
async def post_download_infinitt(project_name:str,
                                 filter_schema : FilterSchema,
                                 session: SessionDep) -> Dict[str, List[str]]:
    project = session.query(ProjectModel).where(ProjectModel.name == project_name).first()
    if project:
        project_study_list  = session.query(ProjectStudyModel).where(ProjectStudyModel.project_uid==project.uid).all()
        accession_number_list = list(map(lambda x: x.study.accession_number, project_study_list))
        result = []
        accession_number_len = len(accession_number_list)
        for index in range((accession_number_len//50)+1):
            start_index = index*50
            end_index = start_index + 50
            result.append(','.join(accession_number_list[start_index:end_index]))
        return {'code': result}
    return {}

def convert_extra_data_filter_type(extra_data_filter: Dict):
        new_extra_data_filter = extra_data_filter.copy()
        new_extra_data_filter['field'] = new_extra_data_filter['field'].replace('extra_data.','')
        try:
            new_value = float(new_extra_data_filter['value'])
            new_extra_data_filter['value'] = new_value
        except ValueError :
            pass
        return new_extra_data_filter


def get_age_by_study_date(birth_date, study_date):
    year = study_date.year - birth_date.year
    month = study_date.month - birth_date.month
    if month < 0:
        year = year - 1
        month = 12 + month
    day_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if calendar.isleap(study_date.year):  # 判斷如果是閏年
        day_list[1] = 29  # 就將二月份的天數改成 29 天
    day = study_date.day - birth_date.day
    if day < 0:
        month = month - 1
        if month < 0:
            year = year - 1
            month = 12 + month
        day = day_list[month] + day
    return year


def get_extra_data_key(columns: List[str]):
    extra_data_key_list = list(filter(lambda x: 'extra_data' in x, columns))
    return {"extra_data_keys": extra_data_key_list}


filter_op_list = [
    #'is_null',
    'like',
    '==',
    '>', '<',
    '>=', '<=', '!=',
    'regexp',
    'is_not_null', 'in',
    #'not_in',
    #'ilike', 'not_ilike',
    #'any', 'not_any'
]