from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from app.v3.endpoints.extraction_templates.agent_services.schemas import (
    ExtractionOutputRow,
    UserInputRow,
)


class CreateExtractionTemplateRequest(BaseModel):
    project_id: UUID
    thread_id: UUID
    user_id: UUID
    user_query: str
    task_type: Literal["template_creation", "template_editing"] = "template_creation"

    table_name: str | None = None
    table_description: str | None = None
    input_rows: list[UserInputRow] | None = None
    output_rows: list[ExtractionOutputRow] | None = None


class StreamChunkMetadata(BaseModel):
    chunk_order: int
    step_id: int
    active_time_seconds: float
    action: str | None = None
    template_schema: dict | None = None


class StreamChunk(BaseModel):
    type: str
    header: str
    body_message: str | None
    metadata: StreamChunkMetadata | None = None
