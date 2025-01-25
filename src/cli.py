import click
import logging
from os import path
from langchain_core.messages import HumanMessage, AIMessage

from src import services, workflow_query, workflow_ingest, db
from src.config import Config


CONFIG = Config.from_env()

# how to add a top-level option --log ai?


@click.group()
def cli():
    pass


@click.command(name="ingest")
@click.argument("path_or_url")
def cmd_ingest(path_or_url: str):
    if path_or_url.startswith(("http://", "https://", "file://", "data://")):
        input_url = path_or_url
    else:
        abs_path = path.abspath(path_or_url)
        input_url = f"file://{abs_path}"
    initial_agent_state = workflow_ingest.AgentState(
        url=input_url, source_content=None, extracted_data=[], topic_id=None
    )

    workflow_ingest.get_graph().invoke(initial_agent_state, CONFIG.to_runnable_config())

    click.echo()
    click.echo("Data ingested successfully")
    click.echo()


@click.command(name="initdb")
def cmd_initdb():
    engine = services.get_db(CONFIG)

    db.SqlAlchemyBase.metadata.create_all(engine)

    click.echo()
    click.echo("Database initialized successfully")
    click.echo()


@click.command(name="query")
@click.argument("user_query")
def cmd_query(user_query: str):
    initial_agent_state = workflow_query.AgentState(
        messages=[HumanMessage(content=user_query)],
        retrieved_knowledge=[],
        query=None,
        topic_id=None,
        external_knowledge_sources=[],
    )

    response = workflow_query.get_graph().invoke(
        initial_agent_state, CONFIG.to_runnable_config()
    )

    ai_messages = [msg for msg in response["messages"] if isinstance(msg, AIMessage)]
    click.echo()
    if ai_messages:
        click.echo(ai_messages[-1].content)
    else:
        click.echo("No AI response found")
    click.echo()


if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, CONFIG.log_level))

    cli.add_command(cmd_ingest)
    cli.add_command(cmd_initdb)
    cli.add_command(cmd_query)
    cli()
