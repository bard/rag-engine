import click
import logging
from os import path
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy.orm import Session

from src import services, workflow_query, workflow_ingest, db
from src.config import Config
from src.db import SqlTopic


CONFIG = Config.from_env()


@click.group()
@click.option(
    "--log",
    help="Set logging level",
    default="INFO",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
)
def cli(log):
    logging.basicConfig(level=getattr(logging, log.upper()))


@click.command(name="create_topic")
@click.option("--id", required=False, help="Optional topic ID")
@click.option("--name", required=True, help="Topic name")
def cmd_create_topic(id: str | None, name: str):
    db = services.get_db(CONFIG)
    with Session(db) as session:
        topic = SqlTopic(id=id, name=name)
        session.add(topic)
        session.commit()
        session.refresh(topic)

    click.echo()
    click.echo(f"Created topic '{name}' with ID: {topic.id}")
    click.echo()


@click.command(name="ingest")
@click.argument("path_or_url")
@click.option("--topic_id", required=False, help="Optional topic ID")
def cmd_ingest(path_or_url: str, topic_id: str | None):
    if path_or_url.startswith(("http://", "https://", "file://", "data://")):
        input_url = path_or_url
    else:
        abs_path = path.abspath(path_or_url)
        input_url = f"file://{abs_path}"
    initial_agent_state = workflow_ingest.AgentState(
        url=input_url, source_content=None, extracted_data=[], topic_id=topic_id
    )

    workflow_ingest.get_graph().invoke(initial_agent_state, CONFIG.to_runnable_config())

    click.echo()
    click.echo("Data ingested successfully")
    click.echo()


@click.command(name="list_topics")
def cmd_list_topics():
    """List all available topics."""
    db_engine = services.get_db(CONFIG)
    with Session(db_engine) as session:
        topics = session.query(SqlTopic).order_by(SqlTopic.name).all()

    click.echo()
    if topics:
        click.echo("Available topics:")
        for topic in topics:
            click.echo(f"  {topic.id}: {topic.name}")
    else:
        click.echo("No topics found")
    click.echo()


@click.command(name="info")
def cmd_info():
    """Display information about the database and vector store."""
    # Get database stats
    db_engine = services.get_db(CONFIG)
    with Session(db_engine) as session:
        topic_count = session.query(SqlTopic).count()
        doc_count = session.query(db.SqlKnowledgeBaseDocument).count()

    # Get sanitized DB URL (remove credentials)
    db_url = str(CONFIG.db.url)
    if "@" in db_url:
        db_url = db_url.split("@")[1]
        db_url = f"***:***@{db_url}"

    click.echo()
    click.echo("Database Information:")
    click.echo(f"  URL: {db_url}")
    click.echo(f"  Topics: {topic_count}")
    click.echo(f"  Documents: {doc_count}")
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
@click.option("--topic_id", required=False, help="Optional topic ID")
def cmd_query(user_query: str, topic_id: str | None):
    initial_agent_state = workflow_query.AgentState(
        messages=[HumanMessage(content=user_query)],
        retrieved_knowledge=[],
        query=None,
        topic_id=topic_id,
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
    cli.add_command(cmd_create_topic)
    cli.add_command(cmd_ingest)
    cli.add_command(cmd_info)
    cli.add_command(cmd_initdb)
    cli.add_command(cmd_list_topics)
    cli.add_command(cmd_query)
    cli()
