import click
import os
import logging
from dotenv import load_dotenv
from langchain_core.runnables.config import RunnableConfig

from src import ingest, services, data, config


@click.group()
def cli():
    pass


@click.command(name="ingest")
@click.option("--data", help="Path or URL of resource to ingest.")
def cmd_ingest(data):
    if data.startswith(("http://", "https://", "file://", "data://")):
        input_url = data
    else:
        abs_path = os.path.abspath(data)
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
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "weather": {
                "api_key": os.getenv("OPENWEATHERMAP_API_KEY"),
            },
            "db": {
                "url": os.getenv("DB_URL"),
            },
            "embeddings": {
                "type": "chroma-internal",
            },
            "vector_store": {
                "type": "chroma",
                "collection_name": "documents",
                "path": os.getenv("CHROMA_DB_PATH"),
            },
        }
    )


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.DEBUG)

    cli.add_command(cmd_ingest)
    cli.add_command(cmd_initdb)
    cli()
