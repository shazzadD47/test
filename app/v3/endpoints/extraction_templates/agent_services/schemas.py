from typing import Annotated, Literal
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def get_name(item):
    """Extract name from either dict or object."""
    if isinstance(item, dict):
        return item.get("name")
    return getattr(item, "name", None)


def get_id(item):
    """Extract id from either dict or object."""
    if isinstance(item, dict):
        return item.get("id")
    return getattr(item, "id", None)


def _get_action(item):
    """Get the reducer action from a dict marker or class instance."""
    if isinstance(item, dict):
        return item.get("__reducer_action")
    return None


def list_reducer(old, new):
    """Reducer that handles additions, deletions, and updates in lists.

    Supports dict-based markers for serialization safety:
    - {"__reducer_action": "remove", "name": "..."} to remove by name
    - {"__reducer_action": "update", "id": "...", "item": <model>} to update by id
    """
    result = list(old)

    for item in new:
        action = _get_action(item)

        if action == "remove":
            item_name = item["name"]
            result = [
                old_item for old_item in result if get_name(old_item) != item_name
            ]
        elif action == "update":
            item_id = str(item["id"])
            new_item = item["item"]
            for i, old_item in enumerate(result):
                if str(get_id(old_item)) == item_id:
                    result[i] = new_item
                    break
        else:
            # Check for duplicates before adding
            item_name = get_name(item)
            if not any(get_name(old_item) == item_name for old_item in result):
                result.append(item)

    return result


class BaseRow(BaseModel):
    id: UUID = Field(
        default_factory=uuid4,
        description="The unique identifier of the row.",
    )
    name: str = Field(
        description="The name of the row.",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_]+$",
    )
    description: str = Field(
        description="The description of the row.",
        min_length=1,
    )


class UserInputRow(BaseRow):
    name: str = Field(
        description="The name of the input row.",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_ ]+$",
    )
    input_type: Literal["chart", "image", "text", "table", "equation"] = Field(
        description="The type of the input data."
    )
    is_required: bool = Field(
        default=False,
        description="Whether the input is required for extraction.",
    )


class ExtractionOutputRow(BaseRow):
    d_type: Literal["string", "number"] = Field(
        description="The data type of the column to extract the data from."
    )
    is_root: bool = Field(
        default=False,
        description="Whether the output is a root output for extraction.",
    )


class SuggestedAction(BaseModel):
    action: Literal["suggest_user_to_finish", "mark_as_finished"]
    reason: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] = []
    suggested_action: SuggestedAction | None = None

    project_id: str
    user_token: str
    task_type: str

    flag_to_file_id_map: dict[str, str] = {}
    file_id_to_flag_map: dict[str, str] = {}
    file_details_list: list[dict] = []

    table_name: str | None = None
    table_description: str | None = None
    input_schema: Annotated[list[UserInputRow], list_reducer] = []
    output_schema: Annotated[list[ExtractionOutputRow], list_reducer] = []

    incoming_input_rows: list[UserInputRow]
    incoming_output_rows: list[ExtractionOutputRow]

    info_hash: str | None
    input_rows_hash: str | None
    output_rows_hash: str | None


def RemoveItem(item: UserInputRow | ExtractionOutputRow) -> dict:
    """Create a remove marker dict. Survives checkpoint serialization."""
    return {"__reducer_action": "remove", "name": get_name(item)}


def UpdateItem(item: UserInputRow | ExtractionOutputRow) -> dict:
    """Create an update marker dict. Survives checkpoint serialization."""
    return {"__reducer_action": "update", "id": str(get_id(item)), "item": item}
