from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy import inspect


class SqlAlchemyBase(DeclarativeBase):
    pass


class SqlKnowledgeBaseDocument(SqlAlchemyBase):
    """SqlAlchemy model for stored document"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    data: Mapped[str] = mapped_column(nullable=False)
    readable: Mapped[str] = mapped_column(nullable=False)
    # TODO add title and source url fields

    def __repr__(self) -> str:
        return f"SqlKnowledgeBaseDocument(id={self.id}, data={self.data}, readable={self.readable}"

    def to_dict(self):
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}
