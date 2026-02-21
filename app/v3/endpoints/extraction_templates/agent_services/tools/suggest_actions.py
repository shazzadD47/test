from typing import Literal

from langchain_core.tools import tool
from pydantic import Field

SUGGEST_ACTIONS_TOOL_DESCRIPTION = """Use this tool to suggest an action to the user when appropriate.

IMPORTANT GUIDELINES:
- ONLY call this tool alone - do NOT combine it with other tool calls
- Use 'suggest_user_to_finish' when the template is ready and user should review/complete it
- Use 'mark_as_finished' when the template creation process is complete

Examples of when to use:
- User has defined all required inputs and outputs for the extraction template
- Template schema is complete and ready for use
- User asks "are we done?" or "is this ready?"

Returns a structured response with the action and reason.
"""  # noqa: E501


@tool(description=SUGGEST_ACTIONS_TOOL_DESCRIPTION)
def suggest_actions_to_user(
    action: Literal["suggest_user_to_finish", "mark_as_finished"] = Field(
        description=(
            "The action to suggest:\n"
            "- 'suggest_user_to_finish': Suggest to the user that they should "
            "finish/complete the template\n"
            "- 'mark_as_finished': Mark the template creation as finished"
        )
    ),
    reason: str = Field(
        description="Explanation for why you're suggesting this action to the user"
    ),
) -> dict:
    return {
        "action": action,
        "reason": reason,
    }
