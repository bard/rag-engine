import pprint
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel
from enum import Enum
from data_uri_parser import DataURI  # type: ignore[import-untyped]

from .. import services, workflow_ingest
from ..db import SqlTopic, SqlKnowledgeBaseDocument
from ..config import Config
from .deps import get_config


router = APIRouter()


class NoteCreate(BaseModel):
    """Pydantic model for note creation request"""

    class NoteContentType(str, Enum):
        PLAIN = "text/plain"
        HTML = "text/html"

    content: str
    content_type: NoteContentType
    topic_id: Optional[str] = None


@router.post("/notes")
def create_note(note: NoteCreate, config: Config = Depends(get_config)):
    """Add a new note to the knowledge base"""

    if note.topic_id:
        db = services.get_db(config)
        with Session(db) as session:
            topic = session.query(SqlTopic).filter(SqlTopic.id == note.topic_id).first()
            if topic is None:
                raise HTTPException(status_code=400, detail="Invalid topic")

    data_uri = DataURI.make(
        "text/plain",
        charset="utf8",
        base64=True,
        data=note.content.encode("utf-8"),
    )

    initial_state = workflow_ingest.AgentState(
        url=data_uri,
        source_content=None,
        extracted_data=[],
        topic_id=note.topic_id,
    )

    try:
        result = workflow_ingest.get_graph().invoke(
            initial_state, config.to_runnable_config()
        )

        # Get the ID of the first extracted document
        if result["extracted_data"]:
            note_id = result["extracted_data"][0].id()
            return {"id": note_id}
        else:
            raise HTTPException(status_code=500, detail="No note was created")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notes")
def get_notes(
    topic_id: str = Query(
        description="Topic ID to filter notes by. Use 'default' to get notes without a topic."
    ),
    config: Config = Depends(get_config),
):
    """Get notes filtered by topic"""
    db = services.get_db(config)

    with Session(db) as session:
        query = session.query(SqlKnowledgeBaseDocument)

        if topic_id == "default":
            query = query.filter(SqlKnowledgeBaseDocument.topic_id.is_(None))
        elif topic_id is not None:
            db_topic = session.query(SqlTopic).filter(SqlTopic.id == topic_id).first()
            if db_topic is None:
                raise HTTPException(status_code=404, detail="Topic not found")
            query = query.filter(SqlKnowledgeBaseDocument.topic_id == db_topic.id)

        notes = query.all()
        return notes


@router.delete("/notes/{note_id}")
def delete_note(note_id: str, config: Config = Depends(get_config)):
    """Delete a note from both database and vector store"""
    db = services.get_db(config)
    vector_store = services.get_vector_store(config)

    with Session(db) as session:
        # First check if note exists
        note = session.query(SqlKnowledgeBaseDocument).filter(
            SqlKnowledgeBaseDocument.id == note_id
        ).first()
        if note is None:
            raise HTTPException(status_code=404, detail="Note not found")

        try:
            # Delete from database
            session.delete(note)
            session.commit()

            # Delete all chunks from vector store
            # The vector store contains chunks with IDs in format "{note_id}-{chunk_id}"
            vector_store.delete([f"{note_id}-{i}" for i in range(100)])  # Assuming max 100 chunks per note

            return {"message": f"Note {note_id} deleted successfully"}
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=str(e))
