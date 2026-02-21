from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from app.logging import logger
from app.utils.llms import get_message_text
from app.v3.endpoints.extraction_templates.agent_services.agents import (
    agent_tools,
    get_main_agent,
)
from app.v3.endpoints.extraction_templates.agent_services.schemas import (
    AgentState,
    RemoveItem,
    SuggestedAction,
    UpdateItem,
    get_id,
)
from app.v3.endpoints.extraction_templates.utils import compute_hash

tool_node = ToolNode(tools=agent_tools)


def _to_dict(row):
    """Normalize a row to dict for comparison."""
    if isinstance(row, dict):
        return row
    if hasattr(row, "model_dump"):
        return row.model_dump()
    return row


def reconcile_rows(existing, incoming):
    """Produce reducer operations to sync existing rows with incoming rows."""
    operations = []

    existing_by_id = {str(get_id(row)): row for row in existing}
    incoming_by_id = {str(get_id(row)): row for row in incoming}

    for row_id, row in existing_by_id.items():
        if row_id not in incoming_by_id:
            operations.append(RemoveItem(row))

    for row_id, row in incoming_by_id.items():
        if row_id in existing_by_id:
            if _to_dict(row) != _to_dict(existing_by_id[row_id]):
                operations.append(UpdateItem(row))
        else:
            operations.append(row)

    return operations


async def setup_node(state: AgentState) -> Command[Literal["main_agent"]]:
    flag_to_file_id_map = state.get("flag_to_file_id_map", {})
    file_id_to_flag_map = state.get("file_id_to_flag_map", {})
    file_details_list = state.get("file_details_list", [])

    table_name = state.get("table_name")
    table_description = state.get("table_description")
    task_type = state.get("task_type", "template_creation")

    info_hash = state.get("info_hash")
    input_rows_hash = state.get("input_rows_hash")
    output_rows_hash = state.get("output_rows_hash")

    incoming_input_rows = state.get("incoming_input_rows", [])
    incoming_output_rows = state.get("incoming_output_rows", [])

    existing_input_rows = state.get("input_schema", [])
    existing_output_rows = state.get("output_schema", [])

    input_ops = reconcile_rows(existing_input_rows, incoming_input_rows)
    output_ops = reconcile_rows(existing_output_rows, incoming_output_rows)

    return_messages = []
    if input_rows_hash and compute_hash(incoming_input_rows) != input_rows_hash:
        return_messages.append(
            SystemMessage(
                content="The input rows have changed. Must review the changes "
                "before continuing using the `read_current_extraction_schema` tool."
            )
        )

    if output_rows_hash and compute_hash(incoming_output_rows) != output_rows_hash:
        return_messages.append(
            SystemMessage(
                content=(
                    "The output rows have changed. Must review the changes "
                    "before continuing using the `read_current_extraction_schema` tool."
                )
            )
        )

    return Command(
        goto="main_agent",
        update={
            "messages": return_messages,
            "flag_to_file_id_map": flag_to_file_id_map,
            "file_id_to_flag_map": file_id_to_flag_map,
            "file_details_list": file_details_list,
            "table_name": table_name,
            "table_description": table_description,
            "task_type": task_type,
            "input_schema": input_ops,
            "output_schema": output_ops,
            "incoming_input_rows": [],
            "incoming_output_rows": [],
            "info_hash": info_hash,
            "input_rows_hash": input_rows_hash,
            "output_rows_hash": output_rows_hash,
        },
    )


async def main_agent_node(state: AgentState) -> Command[Literal[END]]:
    """Main agent node."""
    agent = get_main_agent()

    should_retry = True
    retry_count = 0
    retry_messages = []

    while should_retry and retry_count < 3:
        response = await agent.ainvoke(
            {
                "messages": state["messages"] + retry_messages,
                "task_type": state.get("task_type", "template_creation"),
            }
        )

        if (response.tool_calls and len(response.tool_calls) > 0) or (
            get_message_text(response).strip()
        ):
            should_retry = False
        else:
            logger.warning(
                f"Agent produced a null response (attempt {retry_count + 1}/3). "
                f"Message count: {len(state['messages'])}"
            )

        retry_count += 1
        retry_messages = [
            SystemMessage(
                content="In last attempt you produced a null response. Make sure that "
                "you produce a valid response of the previous messages. Try again."
            )
        ]

    suggested_action = None

    if response.tool_calls:
        remaining_tool_calls = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "suggest_actions_to_user":
                logger.info(
                    "Agent called suggest_actions_to_user, extracting from tool calls"
                )
                suggested_action = SuggestedAction(**tool_call["args"])
            else:
                remaining_tool_calls.append(tool_call)

        response.tool_calls = remaining_tool_calls

    goto = "tools" if response.tool_calls and len(response.tool_calls) > 0 else "exit"

    return Command(
        goto=goto, update={"messages": [response], "suggested_action": suggested_action}
    )


async def exit_node(state: AgentState) -> Command[Literal[END]]:
    table_name = state.get("table_name") or ""
    table_description = state.get("table_description") or ""
    table_info = table_name + table_description

    info_hash = compute_hash(table_info) if table_info else None
    input_rows_hash = compute_hash(state.get("input_schema", []))
    output_rows_hash = compute_hash(state.get("output_schema", []))

    return Command(
        goto=END,
        update={
            "messages": [],
            "input_schema": [],
            "output_schema": [],
            "info_hash": info_hash,
            "input_rows_hash": input_rows_hash,
            "output_rows_hash": output_rows_hash,
        },
    )
