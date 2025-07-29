# app/routers.py

from typing import TYPE_CHECKING
from fastapi import APIRouter,Request
if TYPE_CHECKING:
    import pathlib

from backend.app import series,sync,rerun, study, listen


router = APIRouter()
router.include_router(series.router, tags=["series"])
router.include_router(rerun.router, tags=["rerun"])
router.include_router(sync.router, tags=["sync"])
router.include_router(study.router, tags=["study"])
router.include_router(listen.router, tags=["listen"])



@router.post('/upload_json')
async def upload_json(request:Request):
    json_data = await request.json()
    print(json_data)
