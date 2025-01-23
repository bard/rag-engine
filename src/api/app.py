from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import route_notes, route_topics, route_query, route_healthcheck
from ..settings import Settings


app = FastAPI()
settings = Settings()  # type: ignore # https://github.com/pydantic/pydantic-settings/issues/201

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(route_notes.router, tags=["notes"])
app.include_router(route_topics.router, tags=["topics"])
app.include_router(route_query.router, tags=["query"])
app.include_router(route_healthcheck.router, tags=["system"])
