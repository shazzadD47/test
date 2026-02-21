from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.v3.endpoints.agent_chat.utils.memory import get_shared_checkpointer
from app.v3.endpoints.extraction_templates.agent_services.nodes import (
    exit_node,
    main_agent_node,
    setup_node,
    tool_node,
)
from app.v3.endpoints.extraction_templates.agent_services.schemas import AgentState


async def get_graph() -> CompiledStateGraph[AgentState]:
    workflow = StateGraph(AgentState)

    workflow.add_node("setup", setup_node)
    workflow.add_node("main_agent", main_agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("exit", exit_node)

    workflow.add_edge(START, "setup")
    workflow.add_edge("tools", "main_agent")

    checkpointer = await get_shared_checkpointer()

    return workflow.compile(checkpointer=checkpointer)
