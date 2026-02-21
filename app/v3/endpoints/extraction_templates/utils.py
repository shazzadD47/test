import hashlib
import json

from langchain_core.messages import BaseMessageChunk
from pydantic import BaseModel

from app.v3.endpoints.extraction_templates.agent_services.schemas import AgentState


def prepare_tool_response(
    tool_name: str, content: str, message_chunk: BaseMessageChunk
) -> str:
    """Prepare the response from the tool.

    Args:
        tool_name (str): The name of the tool.
        content (str): The content of the message chunk.
        message_chunk (BaseMessageChunk): The message chunk from the tool.

    Returns:
        str: The prepared response from the tool.
    """
    if tool_name == "read_file":
        if hasattr(message_chunk, "artifact") and message_chunk.artifact:
            file_path = message_chunk.artifact["file_path"]
            content = f"Reading from file `{file_path}`"
        else:
            lines = content.split("\n")
            if len(lines) > 5:
                content = (
                    "\n".join(lines[:5])
                    + f"\n...({len(lines) - 5} more lines were read...)"
                )
            else:
                content = "\n".join(lines)

    return content


def prepare_template_schema(state: AgentState) -> dict:
    schema = {
        "name": state.get("table_name"),
        "description": state.get("table_description"),
    }

    input_schema = state.get("input_schema", [])
    output_schema = state.get("output_schema", [])

    schema["input_schema"] = []
    for item in input_schema:
        if isinstance(item, dict) and "__reducer_action" not in item:
            schema["input_schema"].append(item)
        elif isinstance(item, BaseModel):
            schema["input_schema"].append(item.model_dump())

    schema["output_schema"] = []
    for item in output_schema:
        if isinstance(item, dict) and "__reducer_action" not in item:
            schema["output_schema"].append(item)
        elif isinstance(item, BaseModel):
            schema["output_schema"].append(item.model_dump())

    return schema


def compute_hash(data: str | dict | list) -> str:
    """Compute a SHA-256 hash of the data."""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True, default=str)

    return hashlib.sha256(data.encode()).hexdigest()
