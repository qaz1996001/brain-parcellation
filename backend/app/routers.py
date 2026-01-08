# app/routers.py

from typing import TYPE_CHECKING
from fastapi import APIRouter,Request
if TYPE_CHECKING:
    pass

from backend.app import series
from backend.app import sync
from backend.app import rerun
from backend.app import inference


router = APIRouter()
router.include_router(series.router, tags=["series"])
router.include_router(rerun.router, tags=["rerun"])
router.include_router(sync.router, tags=["sync"])
router.include_router(inference.router, tags=["inference"])
# router.include_router(study.router, tags=["study"])
# router.include_router(listen.router, tags=["listen"])