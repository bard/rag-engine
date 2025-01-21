import click
import logging
from os import path
from langchain_core.runnables.config import RunnableConfig

from src import ingest, services, data, config
from src.settings import Settings


SETTINGS = Settings()  # pyright: ignore[reportCallIssue] https://github.com/pydantic/pydantic-settings/issues/201


@click.group()
def cli():
    pass


@click.command(name="ingest")
@click.option("--data", help="Path or URL of resource to ingest.")
def cmd_ingest(data):
    if data.startswith(("http://", "https://", "file://", "data://")):
        input_url = data
    else:
        abs_path = path.abspath(data)
        input_url = f"file://{abs_path}"

    initial_agent_state = ingest.AgentState(
        url=input_url, source_content=None, extracted_data=[]
    )

    ingest.get_graph().invoke(initial_agent_state, get_config())


@click.command(name="initdb")
def cmd_initdb():
    engine = services.get_db(config.AgentConfig.from_runnable_config(get_config()))
    data.SqlAlchemyBase.metadata.create_all(engine)
    click.echo("Database initialized successfully")


def get_config() -> RunnableConfig:
    return RunnableConfig(
        configurable={
            "llm": {
                "type": "openai",
                "model": "gpt-4o",
                "api_key": SETTINGS.openai_api_key.get_secret_value(),
            },
            "weather": {
                "api_key": SETTINGS.openweathermap_api_key.get_secret_value(),
            },
            "db": {
                "url": SETTINGS.db_url.get_secret_value(),
            },
            "embeddings": {
                "type": "chroma-internal",
            },
            "vector_store": {
                "type": "chroma",
                "collection_name": "documents",
                "path": SETTINGS.chroma_db_path,
            },
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, SETTINGS.log_level))

    cli.add_command(cmd_ingest)
    cli.add_command(cmd_initdb)
    cli()
