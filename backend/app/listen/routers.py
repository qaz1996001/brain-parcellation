# app/listen/routers.py

import logging
from fastapi import APIRouter, Depends, Response
from fastapi_cache import FastAPICache
from advanced_alchemy.extensions.fastapi import service, filters

from backend.app.listen import urls

from backend.app.config.deps import provide_filters
from ..database import alchemy

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(urls.LISTEN_GET_,
            status_code=200,
            summary="",
            )
async def get_():
    # 1. listen 1 new study
    # {
    # "ChangeType": "UpdatedAttachment",
    # "Date": "20250707T054525",
    # "ID": "e79deea8-9d5084ca-c87237bb-e8b276fe-2cbcce6b",
    # "Path": "/series/e79deea8-9d5084ca-c87237bb-e8b276fe-2cbcce6b",
    # "ResourceType": "Series",
    # "Seq": 42112
    # }
    # "Done": true,
    # "Last": 42112
    # # http://127.0.0.1:8042/changes
    # # current_listen_seq
    # # current_listen_data
    # # prev_listen_seq
    # # prev_listen_seq_data
    # # get http://127.0.0.1:8042/changes?last -> last_seq last_seq_data　Changes　Done　Last
    # #　if　Done　and Last == current_listen_seq:
    # #     pass
    # # if current_listen_seq < last_seq
    # # prev_listen_seq, prev_listen_seq_data   = current_listen_seq, current_listen_data
    # # current_listen_seq, current_listen_data = last_seq, last_seq_data
    # # # get http://127.0.0.1:8042/changes?limit=100&since=current_listen_seq
    # # #

    # 2. listen 2 new study
    # http://127.0.0.1:8042/studies

    return Response("get_")
