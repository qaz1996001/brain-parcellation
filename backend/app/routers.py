# app/routers.py

from typing import TYPE_CHECKING
from fastapi import APIRouter
if TYPE_CHECKING:
    import pathlib

from backend.app import series,sync


router = APIRouter()
router.include_router(series.router, prefix="/series", tags=["series"])
router.include_router(sync.router, prefix="/sync", tags=["sync"])
