import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, Field


class ContextDocs(BaseModel):
    page_content: str
    flag_id: str
    title: str | None = None


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

    tool_request_message: AIMessage | None = None
    tool_calls: list[dict] = []
    tool_results: list[ToolMessage] = []

    user_query: Annotated[str, "The original query from the user"]
    flag_id: Annotated[str | list[str] | None, "The flag_id of the research paper"]
    project_id: Annotated[str | None, "The project_id of the research paper"]
    sources: list[ContextDocs] = []
    sender: Annotated[str, "The sender of the message"]
    next_agent: Annotated[str | None, "The next agent to send the message to"]

    files_for_flag_ids: set[str] = set()
    failed_flag_ids: set[str] = set()

    current_notebook_path: str | None = None
    file_contents: list[dict] | None = None
    user_selected_context: bool = False

    using_jupyter: bool = False

    language: str = "python3"
    language_version: str = "3.12"

    agent_chat_type: Literal["chat", "agent"] = "chat"

    # Notebook agent integration
    notebook_url: str | None = None
    notebook_token: str | None = None
    session_id: str | None = None
    current_kernel: str | None = None
    use_notebook_agent_streaming: bool = False


class AgentChatRequest(BaseModel):
    query: str = Field(description="The query to send to the agent", min_length=2)
    project_id: str | None = Field(None, description="project ID to reference")
    flag_ids: list[str] | None = Field(
        None, description="Optional list of flag IDs to reference"
    )
    file_contents: list[dict] | None = Field(
        None, description="Optional list of file contents for notebook context"
    )
    current_notebook_path: str | None = Field(
        None, description="Optional path to the current notebook"
    )
    user_selected_context: bool = Field(
        False, description="Optional flag indicating if user selected specific context"
    )
    notebook_url: str | None = Field(
        None, description="URL to the Jupyter notebook server"
    )
    notebook_token: str | None = Field(
        None, description="Token for the Jupyter notebook server"
    )


class AgentChatResponse(BaseModel):
    node: str = Field(description="The name of the node")
    node_type: Literal["agent", "tool", "error"] = Field(
        description="The type of the node"
    )
    line_message: str | None = Field(
        None, description="A temporary message to show in UI"
    )
    header: str | None = Field(None, description="A descriptive name for the node")
    body_message: str | None = Field(
        None, description="The message/output to send to the client"
    )
    process_description: str | None = Field(
        None, description="A description of the process"
    )
    metadata: dict | None = Field(None, description="Metadata about the node")
    sources: list[ContextDocs] = Field([], description="Sources used in the node")


class routeResponse(BaseModel):
    next_agent: Literal["code_generator"] = Field(
        ..., description="The next agent to pass the query to"
    )


class GenerateTitleRequest(BaseModel):
    query: str = Field(
        ..., description="The user query to generate a title from", min_length=1
    )


class GenerateTitleResponse(BaseModel):
    title: str = Field(..., description="Generated title for the chat")
