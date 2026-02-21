from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.v3.endpoints.general_extraction.constants import run_name
from app.v3.endpoints.general_extraction.schemas import AgentState
from app.v3.endpoints.general_extraction.services.helpers import (
    check_if_all_labels_extracted,
)
from app.v3.endpoints.general_extraction.services.nodes import (
    input_preprocessing_node,
    label_context_generator_node,
    label_query_generator_node,
    relationship_analyzer_node,
    table_finalization_node,
)


def router(state: AgentState):
    inputs = state["workflow_input"]
    if check_if_all_labels_extracted(inputs["table_structure"]):
        return "table_finalization"
    else:
        return "input_preprocessing"


def get_graph() -> CompiledStateGraph:
    workflow = StateGraph(AgentState)
    langfuse_handler = CallbackHandler()

    workflow.add_node("input_preprocessing", input_preprocessing_node)
    workflow.add_node("relationship_analyzer", relationship_analyzer_node)
    workflow.add_node("label_query_generator", label_query_generator_node)
    workflow.add_node("label_context_generator", label_context_generator_node)
    workflow.add_node("table_finalization", table_finalization_node)

    workflow.add_edge(START, "input_preprocessing")
    workflow.add_edge("input_preprocessing", "relationship_analyzer")
    workflow.add_edge("relationship_analyzer", "label_query_generator")
    workflow.add_edge("label_query_generator", "label_context_generator")
    workflow.add_conditional_edges(
        "label_context_generator",
        router,
        {
            "input_preprocessing": "input_preprocessing",
            "table_finalization": "table_finalization",
        },
    )
    workflow.add_edge("table_finalization", END)

    workflow = workflow.compile().with_config(
        {"run_name": run_name, "callbacks": [langfuse_handler]}
    )

    return workflow
