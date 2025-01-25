from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import route_notes, route_topics, route_query, route_healthcheck
from ..db import SqlAlchemyBase
from .. import services
from ..settings import Settings
from .deps import get_config


settings = Settings()  # type: ignore # https://github.com/pydantic/pydantic-settings/issues/201


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for database initialization"""
    engine = services.get_db(get_config())
    SqlAlchemyBase.metadata.create_all(engine)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(route_notes.router)
app.include_router(route_topics.router)
app.include_router(route_query.router)
app.include_router(route_healthcheck.router)
