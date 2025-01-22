import uuid
from typing import List
from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase
from sqlalchemy import inspect, DateTime, ForeignKey


class SqlAlchemyBase(DeclarativeBase):
    pass


class SqlTopic(SqlAlchemyBase):
    """SqlAlchemy model for stored topic"""

    __tablename__ = "topics"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationship to KnowledgeBase
    knowledge_documents: Mapped[List["SqlKnowledgeBaseDocument"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"SqlTopic(id={self.id}, name={self.name})"


class SqlKnowledgeBaseDocument(SqlAlchemyBase):
    """SqlAlchemy model for stored document"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    data: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    topic_id: Mapped[str] = mapped_column(
        ForeignKey("topics.id"),
        # allow NULL to signify "default topic"
        nullable=True,
    )
    topic: Mapped[SqlTopic] = relationship(back_populates="knowledge_documents")

    def __repr__(self) -> str:
        return f"SqlKnowledgeBaseDocument(id={self.id}, data={self.data}, content={self.content}"

    def to_dict(self):
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}
