import pandas as pd
from fastapi import APIRouter,Query,Form,Depends,Response
from fastapi_pagination import Page,LimitOffsetPage,paginate
from fastapi_pagination.api import create_page
from sqlalchemy.sql import Select
from pydantic import BaseModel, Field

from app.core.paginate import paginate_items
from app.core import SessionDep
from app.model.project import ProjectModel,ProjectStudyModel,ProjectSeriesModel
from app.model.study import StudyModel,TextReportModel
from app.schema.base import CodePage
from app.schema.project import StudyOutput

router = APIRouter()


class ProjectOut(BaseModel):  # define your model
    project_uid: str = Field(..., example="Project Name")
    project_name: str = Field(..., example="Project Name")


@router.get("/")
async def get_project_model(session: SessionDep) -> CodePage[ProjectOut]:
    project_model_list , total, raw_params, params = paginate_items(session, Select(ProjectModel))
    response_list = []
    for project_model in project_model_list:
        response_list.append(ProjectOut(project_uid=project_model[0].uid.hex,
                                        project_name=project_model[0].name))
    additional_data = {'code': 2000,
                       'key': list(ProjectOut.__fields__.keys()),
                       'group_key': {},
                       'op_list': []
                       }
    page: CodePage[ProjectOut] = create_page(response_list,
                                             total=total,
                                             params=params,
                                             **additional_data)
    return page


@router.get("/get_project_study/")
def get_project_study(session: SessionDep) -> Page[StudyOutput]:
    response_list = []
    project_model_list      : ProjectModel = session.query(ProjectModel).all()
    for project_model in project_model_list:
        project_study_items,total,raw_params,params  = paginate_items(session,
                                                                      Select(ProjectStudyModel).where(ProjectStudyModel.project_uid==project_model.uid)
                                                                      )

        for project_study in project_study_items:
            study = project_study[0].study
            text = study.text.text if study.text else ""
            response_list.append(StudyOutput(study_uid=study.uid.hex,
                                             patient_uid=study.patient_uid.hex,
                                             patient_id=study.patient.patient_id,
                                             study_date = study.study_date,
                                             study_time=study.study_time,
                                             study_description=study.study_description,
                                             accession_number=study.accession_number,
                                             text=text))
    page:Page[StudyOutput] = create_page(response_list, total, params)
    return page