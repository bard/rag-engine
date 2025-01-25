from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel

from .. import services
from ..config import Config
from ..db import SqlTopic
from .deps import get_config


router = APIRouter()


class TopicCreate(BaseModel):
    """Pydantic model for topic creation request"""

    name: str


class TopicResponse(BaseModel):
    """Pydantic model for topic response"""

    id: str
    name: str
    created_at: datetime

    class ConfigDict:
        from_attributes = True


@router.post("/topics", response_model=TopicResponse, operation_id="create_topic")
def create_topic(topic: TopicCreate, config: Config = Depends(get_config)):
    """Add a new topic to the database"""
    db = services.get_db(config)

    # Normalize the topic name
    normalized_name = topic.name.strip()

    with Session(db) as session:
        # Check for existing topic (case-insensitive)
        existing_topic = (
            session.query(SqlTopic).filter(SqlTopic.name.ilike(normalized_name)).first()
        )

        if existing_topic:
            raise HTTPException(
                status_code=409, detail=f"Topic '{normalized_name}' already exists"
            )

        db_topic = SqlTopic(name=normalized_name, created_at=datetime.now(timezone.utc))
        try:
            session.add(db_topic)
            session.commit()
            # Refresh to ensure we have the latest data including the generated ID
            session.refresh(db_topic)
            return db_topic
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/topics", response_model=List[TopicResponse], operation_id="list_topics")
def list_topics(config: Config = Depends(get_config)):
    """Get all topics from the database"""
    db = services.get_db(config)

    with Session(db) as session:
        topics = session.query(SqlTopic).all()
        return topics


@router.get(
    "/topics/{topic_id}", response_model=TopicResponse, operation_id="get_topic"
)
def get_topic(topic_id: str, config: Config = Depends(get_config)):
    """Get a topic by ID"""
    db = services.get_db(config)

    with Session(db) as session:
        topic = session.query(SqlTopic).filter(SqlTopic.id == topic_id).first()
        if topic is None:
            raise HTTPException(status_code=404, detail="Topic not found")
        return topic


@router.delete("/topics/{topic_id}", operation_id="delete_topic")
def delete_topic(topic_id: str, config: Config = Depends(get_config)):
    """Delete a topic by ID"""
    db = services.get_db(config)

    with Session(db) as session:
        topic = session.query(SqlTopic).filter(SqlTopic.id == topic_id).first()
        if topic is None:
            raise HTTPException(status_code=404, detail="Topic not found")

        try:
            session.delete(topic)
            session.commit()
            return {"message": f"Topic {topic_id} deleted successfully"}
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=str(e))
