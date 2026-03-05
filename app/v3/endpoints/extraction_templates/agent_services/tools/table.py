from copy import deepcopy
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel

from app.logging import logger
from app.v3.endpoints.extraction_templates.agent_services.schemas import (
    AgentState,
    ExtractionOutputRow,
    RemoveItem,
    UpdateItem,
    UserInputRow,
)


def _clean_schema(rows: list) -> list:
    """Filter out any reducer marker dicts that may have leaked into the state."""
    return [row for row in rows if isinstance(row, BaseModel)]


READ_CURRENT_SCHEMA_TOOL_DESCRIPTION = """Read the current extraction schema."""
ADD_INPUT_ROW_TOOL_DESCRIPTION = """Add a new input row to the extraction schema."""
ADD_OUTPUT_ROW_TOOL_DESCRIPTION = """Add a new output row to the extraction schema."""
DELETE_INPUT_ROW_TOOL_DESCRIPTION = (
    """Delete an input row from the extraction schema."""
)
DELETE_OUTPUT_ROW_TOOL_DESCRIPTION = (
    """Delete an output row from the extraction schema."""
)
UPDATE_INPUT_ROW_TOOL_DESCRIPTION = """Update an input row in the extraction schema."""
UPDATE_OUTPUT_ROW_TOOL_DESCRIPTION = (
    """Update an output row in the extraction schema."""
)
UPDATE_TABLE_INFO_TOOL_DESCRIPTION = (
    """Update the table name and/or description for the extraction template."""
)


@tool(description=READ_CURRENT_SCHEMA_TOOL_DESCRIPTION)
def read_current_extraction_schema(state: Annotated[AgentState, InjectedState]) -> str:
    table_name = state.get("table_name")
    table_description = state.get("table_description")
    input_schema = _clean_schema(state.get("input_schema", []))
    output_schema = _clean_schema(state.get("output_schema", []))

    if not table_name:
        table_name = "[Table Name Not Set Yet]"
    if not table_description:
        table_description = "[Table Description Not Set Yet]"

    schema_str = f"Table Name: {table_name}\nTable Description: {table_description}\n\n"

    if input_schema:
        schema_str += "Input Schema:\n"
        for row in input_schema:
            required = "Yes" if row.is_required else "No"
            schema_str += (
                f"    - [id: {row.id}] {row.name}: {row.description} "
                f"(Type: {row.input_type}, Required: {required})\n"
            )
    else:
        schema_str += "Input Schema: [Not Set Yet or No Inputs Needed]\n"

    schema_str += "\n"
    if output_schema:
        schema_str += "Output Schema:\n"
        for row in output_schema:
            root = "Yes" if row.is_root else "No"
            schema_str += (
                f"    - [id: {row.id}] {row.name}: {row.description} "
                f"(Type: {row.d_type}, Root: {root})\n"
            )
    else:
        schema_str += "Output Schema: [Not Set Yet]\n"

    return schema_str


@tool(description=ADD_INPUT_ROW_TOOL_DESCRIPTION)
def add_input_row(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    name: Annotated[
        str,
        (
            "The name of the input for extraction "
            "(alphanumeric and underscores only, max 64 chars)"
        ),
    ],
    description: Annotated[str, "The description of the input for extraction"],
    input_type: Annotated[
        str,
        (
            "The type of the input data: "
            "'chart', 'image', 'text', 'table', or 'equation'"
        ),
    ],
    is_required: Annotated[
        bool, "Whether the input is required for extraction"
    ] = False,
) -> str | Command:
    existing_input_schema = _clean_schema(state.get("input_schema", []))
    new_items = []

    if not name or len(name) > 64:
        return (
            f"ERROR: Invalid name '{name}'. "
            "Name must be 1-64 characters. "
            "TROUBLESHOOT: Use a shorter name."
        )

    if not name.replace("_", "").isalnum():
        return (
            f"ERROR: Invalid name '{name}'. "
            "Name must contain only alphanumeric characters and underscores. "
            "TROUBLESHOOT: Remove special characters from the name."
        )

    if name in [r.name for r in existing_input_schema]:
        return (
            f"ERROR: Input row '{name}' already exists. "
            "TROUBLESHOOT: Use read_current_extraction_schema to see "
            "existing inputs, or use update_input_row to modify it."
        )

    if input_type not in ["chart", "image", "text", "table", "equation"]:
        return (
            f"ERROR: Invalid input_type '{input_type}'. "
            "TROUBLESHOOT: Use one of: "
            "'chart', 'image', 'text', 'table', 'equation'."
        )

    if input_type == "chart":
        existing_chart_inputs = [
            r for r in existing_input_schema if r.input_type == "chart"
        ]
        if existing_chart_inputs:
            return (
                "ERROR: A chart input already exists "
                f"('{existing_chart_inputs[0].name}'). "
                "Only ONE chart input is allowed per extraction template. "
                "TROUBLESHOOT: Use 'image' type if you need additional visual inputs, "
                "or update the existing chart input using update_input_row."
            )

        # Check if any output rows have is_root=True
        existing_output_schema = _clean_schema(state.get("output_schema", []))
        root_outputs = [r for r in existing_output_schema if r.is_root]
        if root_outputs:
            root_names = ", ".join([f"'{r.name}'" for r in root_outputs])
            return (
                "ERROR: Cannot add a chart input while output rows have is_root=True. "  # nosec B608 # noqa E501
                "Chart inputs require all output columns to be general "
                "(is_root=False). TROUBLESHOOT: First update these root output rows "
                f"to general: {root_names}. Use update_output_row to set is_root=False "
                "for each."
            )

        is_required = True

    if not description or not description.strip():
        return (
            "ERROR: Description cannot be empty. "
            "TROUBLESHOOT: Provide a meaningful description for the input."
        )

    try:
        row = UserInputRow(
            name=name,
            description=description,
            input_type=input_type,
            is_required=is_required,
        )
    except Exception as e:
        return (
            f"ERROR: Failed to create input row. {str(e)}. "
            "TROUBLESHOOT: Check that all parameters are valid."
        )

    new_items.append(row)

    # Auto-add additional inputs for chart type
    additional_info = ""
    if input_type == "chart":
        figure_number_row = UserInputRow(
            name="Figure Number",
            description=(
                "The figure number or label identifying this chart in the document "
                "(e.g., 'Figure 1', 'Fig. 2A', '3B'). This helps locate the chart "
                "in the source material."
            ),
            input_type="text",
            is_required=False,
        )
        new_items.append(figure_number_row)

        chart_legend_row = UserInputRow(
            name="Legend Image",
            description=(
                "The legend or key image associated with this chart, "
                "containing labels, colors, symbols, and their explanations. "
                "This is essential for interpreting the digitized chart data correctly."
            ),
            input_type="image",
            is_required=False,
        )
        new_items.append(chart_legend_row)

        additional_info = (
            " Additionally, 'Figure Number' (text) and 'Legend Image' (image) "
            "inputs were automatically added."
        )

    message = ToolMessage(
        content=(
            f"Input row '{name}' added successfully "
            f"(type: {input_type}, required: {is_required}).{additional_info}"
        ),
        tool_call_id=tool_call_id,
        name="add_input_row",
    )

    return Command(
        update={
            "messages": [message],
            "input_schema": new_items,
        }
    )


def _check_root_column_count(
    output_schema: list, excluding_name: str | None = None
) -> str:
    """Return a warning string if adding a root column exceeds safe limits."""
    existing_roots = [
        r
        for r in output_schema
        if r.is_root and (excluding_name is None or r.name != excluding_name)
    ]
    count = len(existing_roots)
    if count >= 2:
        root_names = ", ".join(f"'{r.name}'" for r in existing_roots)
        return (
            f" WARNING: This schema now has {count + 1} root columns "
            f"({root_names}, plus this one). "
            "Multiple root columns produce a Cartesian product of rows — "
            f"for example, {count + 1} root columns with 4 values each "
            f"would generate {4 ** (count + 1)} rows. "
            "This is almost never the intended behavior. "
            "STRONGLY CONSIDER: Set this column to is_root=False and keep only "
            "1-2 root columns that best define the extraction granularity."
        )
    if count == 1:
        existing_name = existing_roots[0].name
        return (
            f" NOTE: This schema now has 2 root columns "
            f"('{existing_name}' and this one). "
            "Two root columns create a Cartesian product — "
            "e.g., 3 arms × 4 timepoints = 12 rows. "
            "Only proceed if each combination should be a distinct row. "
            "If not, set this column to is_root=False."
        )
    return ""


@tool(description=ADD_OUTPUT_ROW_TOOL_DESCRIPTION)
def add_output_row(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    name: Annotated[
        str,
        (
            "The name of the output column "
            "(alphanumeric and underscores only, max 64 chars)"
        ),
    ],
    description: Annotated[str, "The description of the output column"],
    d_type: Annotated[str, "The data type of the output column: 'string' or 'number'"],
    is_root: Annotated[
        bool, "Whether the output is a root output for extraction"
    ] = False,
) -> str | Command:
    existing_output_schema = _clean_schema(state.get("output_schema", []))
    new_items = []

    if not name or len(name) > 64:
        return (
            f"ERROR: Invalid name '{name}'. "
            "Name must be 1-64 characters. "
            "TROUBLESHOOT: Use a shorter name."
        )

    if not name.replace("_", "").isalnum():
        return (
            f"ERROR: Invalid name '{name}'. "
            "Name must contain only alphanumeric characters and underscores. "
            "TROUBLESHOOT: Remove special characters from the name."
        )

    if name in [r.name for r in existing_output_schema]:
        return (
            f"ERROR: Output row '{name}' already exists. "
            "TROUBLESHOOT: Use read_current_extraction_schema to see "
            "existing columns, or use update_output_row to modify it."
        )

    if d_type not in ["string", "number"]:
        return (
            f"ERROR: Invalid d_type '{d_type}'. "
            "TROUBLESHOOT: Use either 'string' or 'number' as the data type."
        )

    # Check: if any input has chart type, root outputs are not allowed
    if is_root:
        existing_input_schema = state.get("input_schema", [])
        chart_inputs = [r for r in existing_input_schema if r.input_type == "chart"]
        if chart_inputs:
            return (
                "ERROR: Cannot set is_root=True when a chart input exists "
                f"('{chart_inputs[0].name}'). "
                "Chart inputs require all output columns to be general "
                "(is_root=False). TROUBLESHOOT: Set is_root=False for this output "
                "column, or remove the chart input first using delete_input_row."
            )

    # Check: warn if too many root columns
    root_warning = ""
    if is_root:
        root_warning = _check_root_column_count(existing_output_schema)

    if not description or not description.strip():
        return (
            "ERROR: Description cannot be empty. "
            "TROUBLESHOOT: Provide a meaningful description for the output."
        )

    try:
        row = ExtractionOutputRow(
            name=name, description=description, d_type=d_type, is_root=is_root
        )
    except Exception as e:
        return (
            f"ERROR: Failed to create output row. {str(e)}. "
            "TROUBLESHOOT: Check that all parameters are valid."
        )

    new_items.append(row)

    # Auto-add unit column for number type (right after the main column)
    additional_info = ""
    if d_type == "number":
        unit_column_name = f"{name}_unit"

        if len(unit_column_name) > 64:
            additional_info = (
                f" Note: Unit column '{unit_column_name}' was NOT auto-added "
                "because the name would exceed 64 characters. "
                "Add it manually with a shorter name if needed."
            )
        elif unit_column_name not in [r.name for r in existing_output_schema]:
            unit_row = ExtractionOutputRow(
                name=unit_column_name,
                description=(
                    f"The unit of measurement for {name}. Extract the exact unit as "
                    f"written in the source (e.g., 'mg', 'kg', 'mg/kg', '%', 'days', "
                    f"'months'). If no unit is specified, enter 'NA'."
                ),
                d_type="string",
                is_root=False,
            )

            new_items.append(unit_row)
            additional_info = (
                f" Additionally, '{unit_column_name}' (string) column was "
                f"automatically added for the unit."
            )

    message = ToolMessage(
        content=(
            f"Output row '{name}' added successfully "
            f"(type: {d_type}, root: {is_root}).{additional_info}{root_warning}"
        ),
        tool_call_id=tool_call_id,
        name="add_output_row",
    )

    return Command(
        update={
            "messages": [message],
            "output_schema": new_items,
        }
    )


@tool(description=DELETE_INPUT_ROW_TOOL_DESCRIPTION)
def delete_input_row(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    id: Annotated[str, "The UUID of the input row to delete"],
    name: Annotated[str, "The name of the input row to delete"],
) -> str | Command:
    input_schema = _clean_schema(state.get("input_schema", []))
    items_to_remove = []

    if not input_schema:
        return (
            "ERROR: No input schema exists. "
            "TROUBLESHOOT: Use read_current_extraction_schema to verify, "
            "or add input rows first using add_input_row."
        )

    row_to_delete = None
    for row in input_schema:
        if str(row.id) == id and row.name == name:
            row_to_delete = row
            break

    if not row_to_delete:
        return (
            f"ERROR: Input row with name '{name}' and id '{id}' not found. "
            "The id and name must both match an existing row. "
            f"TROUBLESHOOT: Use read_current_extraction_schema to see the "
            "current schema before another try."
        )

    items_to_remove.append(row_to_delete)

    additional_deletions = []
    if row_to_delete.input_type == "chart":
        figure_number_row = None
        for row in input_schema:
            if row.name == "Figure Number":
                figure_number_row = row
                break
        if figure_number_row:
            items_to_remove.append(figure_number_row)
            additional_deletions.append("Figure Number")

        chart_legend_row = None
        for row in input_schema:
            if row.name == "Legend Image":
                chart_legend_row = row
                break
        if chart_legend_row:
            items_to_remove.append(chart_legend_row)
            additional_deletions.append("Legend Image")

    additional_info = ""
    if additional_deletions:
        deleted_list = ", ".join([f"`{d}`" for d in additional_deletions])
        additional_info = f" The associated inputs {deleted_list} were also deleted."

    message = ToolMessage(
        content=f"Input row '{name}' deleted successfully.{additional_info}",
        tool_call_id=tool_call_id,
        name="delete_input_row",
    )

    return Command(
        update={
            "messages": [message],
            "input_schema": [RemoveItem(item) for item in items_to_remove],
        }
    )


@tool(description=DELETE_OUTPUT_ROW_TOOL_DESCRIPTION)
def delete_output_row(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    id: Annotated[str, "The UUID of the output row to delete"],
    name: Annotated[str, "The name of the output row to delete"],
) -> str | Command:
    output_schema = _clean_schema(state.get("output_schema", []))
    items_to_remove = []

    if not output_schema:
        return (
            "ERROR: No output schema exists. "
            "TROUBLESHOOT: Use read_current_extraction_schema to verify, "
            "or add output rows first using add_output_row."
        )

    row_to_delete = None
    for row in output_schema:
        if str(row.id) == id and row.name == name:
            row_to_delete = row
            break

    if not row_to_delete:
        return (
            f"ERROR: Output row with name '{name}' and id '{id}' not found. "
            "The id and name must both match an existing row. "
            f"TROUBLESHOOT: Use read_current_extraction_schema to see the "
            "current schema before another try."
        )

    items_to_remove.append(row_to_delete)

    unit_column_name = f"{name}_unit"
    unit_row_to_delete = None
    for row in output_schema:
        if row.name == unit_column_name:
            unit_row_to_delete = row
            break

    additional_info = ""
    if unit_row_to_delete:
        items_to_remove.append(unit_row_to_delete)
        additional_info = (
            f" The associated unit column '{unit_column_name}' was also deleted."
        )

    message = ToolMessage(
        content=f"Output row '{name}' deleted successfully.{additional_info}",
        tool_call_id=tool_call_id,
        name="delete_output_row",
    )

    return Command(
        update={
            "messages": [message],
            "output_schema": [RemoveItem(item) for item in items_to_remove],
        }
    )


@tool(description=UPDATE_INPUT_ROW_TOOL_DESCRIPTION)
def update_input_row(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    id: Annotated[str, "The UUID of the input row to update"],
    name: Annotated[str, "The current name of the input row (for confirmation)"],
    updated_name: Annotated[
        str | None,
        (
            "The new name for the input row "
            "(alphanumeric and underscores only, max 64 chars). "
            "Leave None to keep current."
        ),
    ] = None,
    description: Annotated[
        str | None,
        "The new description (leave None to keep current)",
    ] = None,
    input_type: Annotated[
        str | None,
        (
            "The new input type: 'chart', 'image', 'text', 'table', "
            "or 'equation' (leave None to keep current)"
        ),
    ] = None,
    is_required: Annotated[
        bool | None,
        "Whether the input is required (leave None to keep current)",
    ] = None,
) -> str | Command:
    input_schema = _clean_schema(state.get("input_schema", []))
    items_to_update = []

    if not input_schema:
        return (
            "ERROR: No input schema exists. "
            "TROUBLESHOOT: Use read_current_extraction_schema to verify, "
            "or add input rows first using add_input_row."
        )

    row_to_update = None
    for row in input_schema:
        if str(row.id) == id and row.name == name:
            row_to_update = row
            break

    if not row_to_update:
        return (
            f"ERROR: Input row with name '{name}' and id '{id}' not found. "
            "The id and name must both match an existing row. "
            "TROUBLESHOOT: Use read_current_extraction_schema to see the "
            "current schema before another try."
        )

    if all(v is None for v in [updated_name, description, input_type, is_required]):
        return (
            "ERROR: No fields to update. "
            "TROUBLESHOOT: Provide at least one field to update "
            "(updated_name, description, input_type, or is_required)."
        )

    if updated_name is not None:
        if not updated_name or len(updated_name) > 64:
            return (
                f"ERROR: Invalid updated_name '{updated_name}'. "
                "Name must be 1-64 characters. "
                "TROUBLESHOOT: Use a shorter name."
            )
        if not updated_name.replace("_", "").isalnum():
            return (
                f"ERROR: Invalid updated_name '{updated_name}'. "
                "Name must contain only alphanumeric characters and underscores. "
                "TROUBLESHOOT: Remove special characters from the name."
            )
        if updated_name != name and updated_name in [r.name for r in input_schema]:
            return (
                f"ERROR: Input row '{updated_name}' already exists. "
                "TROUBLESHOOT: Choose a different name."
            )

    if description is not None and not description.strip():
        return (
            "ERROR: Description cannot be empty. "
            "TROUBLESHOOT: Provide a meaningful description, "
            "or omit the description parameter to keep the current one."
        )

    if input_type is not None and input_type not in [
        "chart",
        "image",
        "text",
        "table",
        "equation",
    ]:
        return (
            f"ERROR: Invalid input_type '{input_type}'. "
            "TROUBLESHOOT: Use one of: "
            "'chart', 'image', 'text', 'table', 'equation'."
        )

    # Validate: only one chart input allowed
    if input_type == "chart" and row_to_update.input_type != "chart":
        existing_chart_inputs = [
            r for r in input_schema if r.input_type == "chart" and r.name != name
        ]
        if existing_chart_inputs:
            return (
                "ERROR: A chart input already exists "
                f"('{existing_chart_inputs[0].name}'). "
                "Only ONE chart input is allowed per extraction template. "
                "TROUBLESHOOT: Use 'image' type instead, "
                "or update the existing chart input using update_input_row."
            )

    # Validate: no root output rows when changing to chart type
    if input_type == "chart" and row_to_update.input_type != "chart":
        output_schema = _clean_schema(state.get("output_schema", []))
        root_outputs = [r for r in output_schema if r.is_root]
        if root_outputs:
            root_names = ", ".join([f"'{r.name}'" for r in root_outputs])
            return (
                "ERROR: Cannot change input type to 'chart' while output rows have "  # nosec B608 # noqa E501
                "is_root=True. Chart inputs require all output columns to be general "
                "(is_root=False). "
                "TROUBLESHOOT: First update these root output rows to general: "
                f"{root_names}. Use update_output_row to set is_root=False for each."
            )

    updated_fields = []
    additional_info = ""
    new_companion_items = []
    updated_item = deepcopy(row_to_update)
    try:
        if updated_name is not None:
            updated_item.name = updated_name
            updated_fields.append("name")

        if description is not None:
            updated_item.description = description
            updated_fields.append("description")

        if input_type is not None:
            updated_item.input_type = input_type
            updated_fields.append("input_type")

        if is_required is not None:
            updated_item.is_required = is_required
            updated_fields.append("is_required")

        # Auto-add companions when changing to chart type
        if input_type == "chart" and row_to_update.input_type != "chart":
            updated_item.is_required = True
            if "is_required" not in updated_fields:
                updated_fields.append("is_required")

            figure_number_row = UserInputRow(
                name="Figure Number",
                description=(
                    "The figure number or label identifying this chart in the "
                    "document (e.g., 'Figure 1', 'Fig. 2A', '3B'). This helps "
                    "locate the chart in the source material."
                ),
                input_type="text",
                is_required=False,
            )
            new_companion_items.append(figure_number_row)

            chart_legend_row = UserInputRow(
                name="Legend Image",
                description=(
                    "The legend or key image associated with this chart, "
                    "containing labels, colors, symbols, and their explanations. "
                    "This is essential for interpreting the digitized chart data "
                    "correctly."
                ),
                input_type="image",
                is_required=False,
            )
            new_companion_items.append(chart_legend_row)

            additional_info = (
                " Additionally, 'Figure Number' (text) and 'Legend Image' (image) "
                "inputs were automatically added, and is_required was set to True."
            )

        items_to_update.append(updated_item)

    except Exception as e:
        logger.error(f"Failed to update input row: {e}", exc_info=True)
        return (
            f"ERROR: Failed to update input row. {str(e)}. "
            "TROUBLESHOOT: Check that all parameter values are valid."
        )

    display_name = updated_name if updated_name else name
    message = ToolMessage(
        content=(
            f"Input row '{display_name}' updated successfully. "
            f"Updated fields: {', '.join(updated_fields)}.{additional_info}"
        ),
        tool_call_id=tool_call_id,
        name="update_input_row",
    )

    return Command(
        update={
            "messages": [message],
            "input_schema": [
                *(UpdateItem(item) for item in items_to_update),
                *new_companion_items,
            ],
        }
    )


@tool(description=UPDATE_OUTPUT_ROW_TOOL_DESCRIPTION)
def update_output_row(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    id: Annotated[str, "The UUID of the output row to update"],
    name: Annotated[str, "The current name of the output row (for confirmation)"],
    updated_name: Annotated[
        str | None,
        (
            "The new name for the output row "
            "(alphanumeric and underscores only, max 64 chars). "
            "Leave None to keep current."
        ),
    ] = None,
    description: Annotated[
        str | None,
        "The new description (leave None to keep current)",
    ] = None,
    d_type: Annotated[
        str | None,
        "The new data type: 'string' or 'number' (leave None to keep current)",
    ] = None,
    is_root: Annotated[
        bool | None,
        "Whether the output is a root output (leave None to keep current)",
    ] = None,
) -> str | Command:
    output_schema = _clean_schema(state.get("output_schema", []))
    items_to_return = []

    if not output_schema:
        return (
            "ERROR: No output schema exists. "
            "TROUBLESHOOT: Use read_current_extraction_schema to verify, "
            "or add output rows first using add_output_row."
        )

    row_to_update = None
    for row in output_schema:
        if str(row.id) == id and row.name == name:
            row_to_update = row
            break

    if not row_to_update:
        return (
            f"ERROR: Output row with name '{name}' and id '{id}' not found. "
            "The id and name must both match an existing row. "
            "TROUBLESHOOT: Use read_current_extraction_schema to see the "
            "current schema before another try."
        )

    if all(v is None for v in [updated_name, description, d_type, is_root]):
        return (
            "ERROR: No fields to update. "
            "TROUBLESHOOT: Provide at least one field to update "
            "(updated_name, description, d_type, or is_root)."
        )

    if updated_name is not None:
        if not updated_name or len(updated_name) > 64:
            return (
                f"ERROR: Invalid updated_name '{updated_name}'. "
                "Name must be 1-64 characters. "
                "TROUBLESHOOT: Use a shorter name."
            )
        if not updated_name.replace("_", "").isalnum():
            return (
                f"ERROR: Invalid updated_name '{updated_name}'. "
                "Name must contain only alphanumeric characters and underscores. "
                "TROUBLESHOOT: Remove special characters from the name."
            )
        if updated_name != name and updated_name in [r.name for r in output_schema]:
            return (
                f"ERROR: Output row '{updated_name}' already exists. "
                "TROUBLESHOOT: Choose a different name."
            )

    if description is not None and not description.strip():
        return (
            "ERROR: Description cannot be empty. "
            "TROUBLESHOOT: Provide a meaningful description, "
            "or omit the description parameter to keep the current one."
        )

    if d_type is not None and d_type not in ["string", "number"]:
        return (
            f"ERROR: Invalid d_type '{d_type}'. "
            "TROUBLESHOOT: Use either 'string' or 'number' as the data type."
        )

    # Validate: no root outputs when chart input exists
    if is_root is True and not row_to_update.is_root:
        input_schema = state.get("input_schema", [])
        chart_inputs = [r for r in input_schema if r.input_type == "chart"]
        if chart_inputs:
            return (
                "ERROR: Cannot set is_root=True when a chart input exists "
                f"('{chart_inputs[0].name}'). "
                "Chart inputs require all output columns to be general "
                "(is_root=False). TROUBLESHOOT: Keep is_root=False for this output "
                "column, or remove the chart input first using delete_input_row."
            )

    # Check: warn if too many root columns
    root_warning = ""
    if is_root is True and not row_to_update.is_root:
        root_warning = _check_root_column_count(output_schema, excluding_name=name)

    updated_fields = []
    old_dtype = row_to_update.d_type
    updated_item = deepcopy(row_to_update)

    try:
        if updated_name is not None:
            updated_item.name = updated_name
            updated_fields.append("name")

        if description is not None:
            updated_item.description = description
            updated_fields.append("description")

        if d_type is not None:
            updated_item.d_type = d_type
            updated_fields.append("d_type")

        if is_root is not None:
            updated_item.is_root = is_root
            updated_fields.append("is_root")

        items_to_return.append(UpdateItem(updated_item))

    except Exception as e:
        logger.error(f"Failed to update output row: {e}", exc_info=True)
        return (
            f"ERROR: Failed to update output row. {str(e)}. "
            "TROUBLESHOOT: Check that all parameter values are valid."
        )

    # Handle unit column when d_type changes
    display_name = updated_name if updated_name else name
    additional_info = ""
    old_unit_column_name = f"{name}_unit"
    new_unit_column_name = f"{display_name}_unit"

    unit_row_index = -1
    for i, row in enumerate(output_schema):
        if row.name == old_unit_column_name:
            unit_row_index = i
            break

    if d_type == "number" and old_dtype == "string" and unit_row_index == -1:
        if len(new_unit_column_name) > 64:
            additional_info = (
                f" Note: Unit column '{new_unit_column_name}' was NOT auto-added "
                "because the name would exceed 64 characters. "
                "Add it manually with a shorter name if needed."
            )
        else:
            unit_row = ExtractionOutputRow(
                name=new_unit_column_name,
                description=(
                    f"The unit of measurement for {display_name}. Extract the exact "
                    f"unit as written in the source (e.g., 'mg', 'kg', 'mg/kg', '%', "
                    f"'days', 'months'). If no unit is specified, enter 'NA'."
                ),
                d_type="string",
                is_root=False,
            )
            items_to_return.append(unit_row)
            additional_info = (
                f" Unit column '{new_unit_column_name}' was automatically added."
            )

    elif d_type == "string" and old_dtype == "number" and unit_row_index != -1:
        items_to_return.append(RemoveItem(output_schema[unit_row_index]))
        additional_info = (
            f" Unit column '{old_unit_column_name}' was automatically removed."
        )

    # Handle unit column rename when output column is renamed (type stays number)
    elif (
        updated_name is not None
        and updated_name != name
        and (d_type or old_dtype) == "number"
        and unit_row_index != -1
    ):
        old_unit = output_schema[unit_row_index]
        if len(new_unit_column_name) > 64:
            additional_info = (
                f" Note: Unit column '{old_unit_column_name}' was NOT renamed "
                f"to '{new_unit_column_name}' because the new name would exceed "
                "64 characters. Rename it manually with a shorter name if needed."
            )
        else:
            renamed_unit = ExtractionOutputRow(
                name=new_unit_column_name,
                description=old_unit.description.replace(name, display_name),
                d_type="string",
                is_root=False,
            )
            items_to_return.append(RemoveItem(old_unit))
            items_to_return.append(renamed_unit)
            additional_info = (
                f" Unit column renamed from '{old_unit_column_name}' "
                f"to '{new_unit_column_name}'."
            )

    message = ToolMessage(
        content=(
            f"Output row '{display_name}' updated successfully. "
            f"Updated fields: {', '.join(updated_fields)}.{additional_info}"
            f"{root_warning}"
        ),
        tool_call_id=tool_call_id,
        name="update_output_row",
    )

    return Command(
        update={
            "messages": [message],
            "output_schema": items_to_return,
        }
    )


@tool(description=UPDATE_TABLE_INFO_TOOL_DESCRIPTION)
def update_table_info(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    table_name: Annotated[
        str | None,
        "The name for the extraction template table (None to keep current)",
    ] = None,
    table_description: Annotated[
        str | None,
        "The description for the extraction template table (None to keep current)",
    ] = None,
) -> str | Command:
    if table_name is None and table_description is None:
        return (
            "ERROR: No fields to update. "
            "TROUBLESHOOT: Provide at least one field to update "
            "(table_name or table_description)."
        )

    if table_name is not None:
        if not table_name.strip():
            return (
                "ERROR: Table name cannot be empty. "
                "TROUBLESHOOT: Provide a meaningful table name."
            )
        if len(table_name) > 128:
            return (
                f"ERROR: Table name is too long ({len(table_name)} characters). "
                "TROUBLESHOOT: Use a table name with 128 characters or less."
            )

    if (table_description is not None) and (not table_description.strip()):
        return (
            "ERROR: Table description cannot be empty. "
            "TROUBLESHOOT: Provide a meaningful table description."
        )

    updated_fields = []
    updates = {}

    if table_name is not None:
        updates["table_name"] = table_name.strip()
        updated_fields.append("table_name")

    if table_description is not None:
        updates["table_description"] = table_description.strip()
        updated_fields.append("table_description")

    current_name = updates.get("table_name", state.get("table_name", "[Not Set]"))
    current_desc = updates.get(
        "table_description", state.get("table_description", "[Not Set]")
    )

    message = ToolMessage(
        content=(
            f"Table info updated successfully. "
            f"Updated fields: {', '.join(updated_fields)}. "
            f"Current table name: '{current_name}', "
            f"description: '{current_desc}'"
        ),
        tool_call_id=tool_call_id,
        name="update_table_info",
    )

    return Command(
        update={
            "messages": [message],
            **updates,
        }
    )
