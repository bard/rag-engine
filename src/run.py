import click
import logging
from os import path
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage


from src import services, data, workflow_query, workflow_ingest
from src.settings import Settings
from src.config import Config


SETTINGS = Settings()  # type: ignore # https://github.com/pydantic/pydantic-settings/issues/201


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
    initial_agent_state = workflow_ingest.AgentState(
        url=input_url, source_content=None, extracted_data=[]
    )

    workflow_ingest.get_graph().invoke(
        initial_agent_state,
        RunnableConfig(configurable=get_config().model_dump()),
    )

    click.echo("Data ingested successfully")


@click.command(name="initdb")
def cmd_initdb():
    engine = services.get_db(get_config())

    data.SqlAlchemyBase.metadata.create_all(engine)

    click.echo("Database initialized successfully")


@click.command(name="query")
@click.argument("user_query")
def cmd_query(user_query: str):
    initial_agent_state = workflow_query.AgentState(
        messages=[HumanMessage(content=user_query)],
        documents=[],
        weather_info=None,
        is_weather_query=False,
        query=None,  # TODO rename to clarify that it's a rewritten query
        location=None,
    )

    response = workflow_query.get_graph().invoke(
        initial_agent_state,
        RunnableConfig(configurable=get_config().model_dump()),
    )

    ai_messages = [msg for msg in response["messages"] if isinstance(msg, AIMessage)]
    if ai_messages:
        click.echo(ai_messages[-1].content)
    else:
        click.echo("No AI response found")


def get_config() -> Config:
    return Config(
        **{
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
            # ignoring type errors here due to https://docs.pydantic.dev/latest/integrations/visual_studio_code/#strict-errors
        }  # type: ignore
    )


if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, SETTINGS.log_level))

    cli.add_command(cmd_ingest)
    cli.add_command(cmd_initdb)
    cli.add_command(cmd_query)
    cli()
