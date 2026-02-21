from contextlib import asynccontextmanager

from langgraph.graph import END, START, StateGraph

from app.v3.endpoints.report_generator.constants import (
    ReportAgents,
    ReportGenerationType,
)
from app.v3.endpoints.report_generator.schema import ReportState
from app.v3.endpoints.report_generator.services.nodes import (
    report_assistant_node,
    report_edit_node,
    report_insights_node,
    report_tool_node,
)
from app.v3.endpoints.report_generator.utils.memory import get_shared_checkpointer


async def report_router(state: ReportState):
    """Router function to determine the next node based on state."""
    if state.get("tool_calls"):
        return "tool"

    return END


async def entry_router(state: ReportState):
    """Router to determine which agent to start with based on generation type."""
    generation_type = state.get("generation_type", ReportGenerationType.AI_ASSISTANT)

    if generation_type == ReportGenerationType.AI_ASSISTANT:
        return "assistant_agent"
    elif generation_type == ReportGenerationType.AI_EDIT:
        return "edit_agent"
    elif generation_type == ReportGenerationType.AI_INSIGHTS:
        return "insight_agent"

    else:
        raise ValueError(f"Unknown generation type: {generation_type}")


async def tool_router(state: ReportState):
    """Router to determine which agent to return to after tool execution."""
    sender = state.get("sender")

    if sender == ReportAgents.ASSISTANT_AGENT.value:
        return "assistant_agent"
    elif sender == ReportAgents.EDIT_AGENT.value:
        return "edit_agent"
    else:
        return "assistant_agent"


@asynccontextmanager
async def get_report_graph():
    """Create and configure the report generation workflow graph."""
    checkpointer = await get_shared_checkpointer()

    try:
        workflow = StateGraph(ReportState)

        workflow.add_node("assistant_agent", report_assistant_node)
        workflow.add_node("edit_agent", report_edit_node)
        workflow.add_node("insight_agent", report_insights_node)
        workflow.add_node("tool", report_tool_node)

        workflow.add_conditional_edges(
            "assistant_agent",
            report_router,
            {
                "tool": "tool",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "edit_agent",
            report_router,
            {
                "tool": "tool",
                END: END,
            },
        )

        workflow.add_edge(
            "insight_agent",
            END,
        )

        workflow.add_conditional_edges(
            "tool",
            tool_router,
            {
                "assistant_agent": "assistant_agent",
                "edit_agent": "edit_agent",
            },
        )

        workflow.add_conditional_edges(
            START,
            entry_router,
            {
                "assistant_agent": "assistant_agent",
                "edit_agent": "edit_agent",
                "insight_agent": "insight_agent",
            },
        )

        compiled_graph = workflow.compile(checkpointer=checkpointer)
        yield compiled_graph

    except Exception:
        raise
