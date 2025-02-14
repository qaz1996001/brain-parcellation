import sentry_sdk
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute
from fastapi_pagination import add_pagination
from starlette.middleware.cors import CORSMiddleware
import uvicorn



def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


def create_app() -> FastAPI:
    from app.api import api_router
    from app.core.config import settings

    if settings.SENTRY_DSN and settings.ENVIRONMENT != "local":
        sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)

    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        generate_unique_id_function=custom_generate_unique_id,
        default_response_class=ORJSONResponse,
        version = settings.VERSION,
    )


    # Set all CORS enabled origins
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "*"
                #str(origin).strip("/") for origin in settings.BACKEND_CORS_ORIGINS
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(api_router, prefix=settings.API_V1_STR)
    add_pagination(app)
    return app

app = create_app()


if __name__ == '__main__':
    uvicorn.run(
        app="main:app",
        host='0.0.0.0',
        port=8800)