import json
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, Enum, Integer, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TEXT, TypeDecorator

from app.core.database.base import Base


class JsonList(TypeDecorator):
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None

    def __repr__(self):
        return f"JsonList({self.impl})"

    def __str__(self):
        return f"JsonList({self.impl})"


class BaseModel(Base):
    __abstract__ = True

    def __repr__(self, indent: int = 2):
        space = " " * indent

        attributes = "\n".join(
            f"{space}{k}={repr(v)}"
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        )

        return f"{self.__tablename__}(\n{attributes}\n)"

    def __str__(self, indent: int = 2):
        return self.__repr__(indent)


class FileDetails(BaseModel):
    __tablename__ = "file_details"

    id = Column(Integer, primary_key=True, autoincrement=True)

    flag_id = Column(Uuid, nullable=False)
    supplementary_id = Column(Text, nullable=True)
    project_id = Column(Text, nullable=False)
    user_id = Column(Text, nullable=False)

    file_extension = Column(Text, nullable=False)

    title = Column(Text, nullable=True)
    doi = Column(Text, nullable=True)
    doi_url = Column(Text, nullable=True)
    publication_type = Column(Text, nullable=True)

    summary = Column(Text, nullable=True)
    authors = Column(JSONB, nullable=True)
    funder = Column(JSONB, nullable=True)
    paper_summary = Column(Text, nullable=True)

    publisher = Column(Text, nullable=True)
    published_date = Column(JSONB, nullable=True)
    issue = Column(Text, nullable=True)
    issue_date = Column(JSONB, nullable=True)
    volume = Column(Text, nullable=True)
    page = Column(Text, nullable=True)
    license = Column(JSONB, nullable=True)

    container_title = Column(Text, nullable=True)
    ISSN = Column(JSONB, nullable=True)

    reference_count = Column(Integer, nullable=True)
    references = Column(JSONB, nullable=True)
    referenced_by_count = Column(Integer, nullable=True)

    raw_metadata = Column(JSONB, nullable=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())


class FigureDetails(BaseModel):
    __tablename__ = "figure_details"

    id = Column(Integer, primary_key=True, autoincrement=True)

    flag_id = Column(Uuid, nullable=False)
    supplementary_id = Column(Text, nullable=True)
    project_id = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)

    figure_id = Column(Uuid, default=uuid4, nullable=False, unique=True)
    parent_figure_id = Column(Uuid, nullable=True)
    figure_number = Column(Text, nullable=True)

    caption = Column(Text, nullable=True)
    summary = Column(Text, nullable=False)
    footnote = Column(Text, nullable=True)
    normalized_bbox_coordinates = Column(JsonList, nullable=False)
    converted_bbox_coordinates = Column(JsonList, nullable=False)

    bucket_path = Column(Text, nullable=False)
    legend_paths = Column(JsonList, nullable=True)
    legend_converted_bbox_coordinates = Column(JsonList, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())


class TableDetails(BaseModel):
    __tablename__ = "table_details"

    id = Column(Integer, primary_key=True, autoincrement=True)

    flag_id = Column(Uuid, nullable=False)
    supplementary_id = Column(Text, nullable=True)
    project_id = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)

    table_id = Column(Uuid, default=uuid4, nullable=False, unique=True)
    table_number = Column(Text, nullable=True)

    caption = Column(Text, nullable=True)
    summary = Column(Text, nullable=False)
    digitized = Column(Text, nullable=False)
    footnote = Column(Text, nullable=True)
    normalized_bbox_coordinates = Column(JsonList, nullable=False)
    converted_bbox_coordinates = Column(JsonList, nullable=False)

    bucket_path = Column(Text, nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())


class PageDetails(BaseModel):
    __tablename__ = "page_details"

    id = Column(Integer, primary_key=True, autoincrement=True)

    flag_id = Column(Uuid, nullable=False)
    supplementary_id = Column(Text, nullable=True)
    project_id = Column(Text, nullable=False)

    page_number = Column(Integer, nullable=False)
    bucket_path = Column(Text, nullable=False)

    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())


class MinerUResponses(BaseModel):
    __tablename__ = "mineru_responses"

    id = Column(Integer, primary_key=True, autoincrement=True)

    flag_id = Column(Uuid, nullable=False)
    supplementary_id = Column(Text, nullable=True)
    response_type = Column(
        Enum("initial", "final", name="response_type_enum"), nullable=False
    )
    response = Column(JSON, nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())
