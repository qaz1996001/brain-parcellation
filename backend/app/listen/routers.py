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

    return Response("get_")

