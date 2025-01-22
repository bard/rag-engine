from datetime import datetime, timezone
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy import inspect, DateTime


class SqlAlchemyBase(DeclarativeBase):
    pass


class SqlKnowledgeBaseDocument(SqlAlchemyBase):
    """SqlAlchemy model for stored document"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    data: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"SqlKnowledgeBaseDocument(id={self.id}, data={self.data}, content={self.content}"

    def to_dict(self):
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}
