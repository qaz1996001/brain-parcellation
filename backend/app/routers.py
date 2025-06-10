# app/routers.py

from typing import TYPE_CHECKING
from fastapi import APIRouter,Request
if TYPE_CHECKING:
    import pathlib

from backend.app import series,sync,find,rerun


router = APIRouter()
router.include_router(series.router, prefix="/series", tags=["series"])
router.include_router(find.router, tags=["find"])
router.include_router(rerun.router, tags=["rerun"])
router.include_router(sync.router, tags=["sync"])




@router.post('/upload_json')
async def upload_json(request:Request):
    json_data = await request.json()
    print(json_data)
