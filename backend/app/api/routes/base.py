from fastapi import APIRouter



router = APIRouter()


@router.get("/hello")
def hello_world():
    return "Hello World!"
