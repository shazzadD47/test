import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, Field


class ReportContextDocs(BaseModel):
    page_content: str
    flag_id: str
    title: str | None = None


class ReportState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

    tool_request_message: AIMessage | None = None
    tool_calls: list[dict] = []
    tool_results: list[ToolMessage] = []

    user_query: Annotated[str, "The original query from the user"]
    flag_id: Annotated[str | list[str] | None, "The flag_id of the research paper"]
    project_id: Annotated[str | None, "The project_id of the research paper"]
    sources: list[ReportContextDocs] = []
    sender: Annotated[str, "The sender of the message"]

    files_for_flag_ids: set[str] = set()
    failed_flag_ids: set[str] = set()

    file_contents: str | list[dict[str, Any]] | None = None
    user_selected_context: bool = False

    generation_type: Literal["ai_assistant", "ai_edit"] = "ai_assistant"
    report_context: str | None = None
    selected_report_context: str | None = None
    selected_text: str | None = None


class ReportGenerationRequest(BaseModel):
    query: str = Field(
        description="The query/prompt for report generation", min_length=2
    )
    project_id: str = Field(..., description="project ID to reference")
    generation_type: Literal["ai_assistant", "ai_edit"] = Field(
        default="ai_assistant",
        description="Type of generation: ai_assistant or ai_edit",
    )
    flag_ids: list[str] | None = Field(
        None, description="Optional list of flag IDs to reference"
    )
    file_contents: str | list[dict[str, Any]] | None = Field(
        None,
        description="Optional list of file contents for context (can be compressed)",
    )
    report_context: str | None = Field(
        None, description="Current report content for context"
    )
    selected_text: str | None = Field(
        None, description="Selected text for AI Edit mode"
    )
    user_selected_context: bool = Field(
        False, description="Optional flag indicating if user selected specific context"
    )


class ReportGenerationResponse(BaseModel):
    node: str = Field(description="The name of the node")
    node_type: Literal["agent", "tool", "error"] = Field(
        description="The type of the node"
    )
    generation_type: Literal["ai_assistant", "ai_edit", "ai_insights"] = Field(
        description="The type of generation being performed"
    )
    header: str | None = Field(None, description="A descriptive name for the node")
    body_message: str | None = Field(
        None, description="The generated content to send to the client"
    )
    process_description: str | None = Field(
        None, description="A description of the process"
    )
    metadata: dict | None = Field(None, description="Metadata about the response")
    sources: list[ReportContextDocs] = Field(
        [], description="Sources used in generation"
    )
    is_complete: bool = Field(False, description="Whether generation is complete")
