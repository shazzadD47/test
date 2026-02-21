from contextlib import asynccontextmanager

from langgraph.graph import END, START, StateGraph

from app.v3.endpoints.agent_chat.schema import AgentState
from app.v3.endpoints.agent_chat.services.nodes import (
    code_generator_node,
    deep_agent_node,
    rag_agent_node,
    setup_node,
    tool_node,
)
from app.v3.endpoints.agent_chat.utils.memory import get_shared_checkpointer

from ..constants import Agents


async def router(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if state.get("tool_calls"):
        calls = state.get("tool_calls")
        filtered_calls = []

        for call in calls:
            if call.get("name") == "routeResponse":
                return call.get("args").get("next_agent")
            else:
                filtered_calls.append(call)

        if filtered_calls:
            state["tool_calls"] = filtered_calls
            return "tool"

    if "FINISHED" in last_message.content:
        return END

    return END


@asynccontextmanager
async def get_graph():
    """Create and configure the agent workflow graph."""
    checkpointer = await get_shared_checkpointer()

    try:
        workflow = StateGraph(AgentState)
        workflow.add_node("setup", setup_node)
        workflow.add_node(Agents.DEEP_AGENT, deep_agent_node)
        workflow.add_node(Agents.MAIN_AGENT, rag_agent_node)
        workflow.add_node(Agents.CODE_GENERATOR, code_generator_node)
        workflow.add_node("tool", tool_node)

        workflow.add_conditional_edges(
            Agents.MAIN_AGENT,
            router,
            {
                "tool": "tool",
                "code_generator": "code_generator",
                END: END,
            },
        )
        workflow.add_conditional_edges(
            Agents.CODE_GENERATOR,
            router,
            {"tool": "tool", END: END},
        )

        workflow.add_conditional_edges(
            Agents.DEEP_AGENT,
            router,
            {"tool": "tool", END: END},
        )

        workflow.add_conditional_edges(
            "tool",
            lambda x: x["sender"],
            {
                Agents.MAIN_AGENT: Agents.MAIN_AGENT,
                Agents.CODE_GENERATOR: Agents.CODE_GENERATOR,
                Agents.DEEP_AGENT: Agents.DEEP_AGENT,
            },
        )

        workflow.add_conditional_edges(
            "setup",
            lambda x: x["next_agent"],
            {
                Agents.MAIN_AGENT: Agents.MAIN_AGENT,
                Agents.CODE_GENERATOR: Agents.CODE_GENERATOR,
                Agents.DEEP_AGENT: Agents.DEEP_AGENT,
            },
        )

        workflow.add_edge(START, "setup")

        compiled_graph = workflow.compile(checkpointer=checkpointer)
        yield compiled_graph
    except Exception:
        raise
