from collections import defaultdict
from typing import Literal

from app.core.vector_store import VectorStore
from app.v3.endpoints.agent_chat.schema import ContextDocs


async def retrieve_context(
    query: str,
    flag_id: str = None,
    project_id: str = None,
    file_type: Literal["document", "code"] = None,
    k: int = 20,
) -> list[ContextDocs]:
    filter_params = {}
    if flag_id:
        filter_params["flag_id"] = flag_id
    if project_id:
        filter_params["project_id"] = project_id
    if file_type:
        filter_params["file_type"] = file_type

    retriever = VectorStore.get_retriever(
        search_kwargs={"filter": filter_params, "k": k}
    )
    contexts = await retriever.ainvoke(query)

    return [
        ContextDocs(
            page_content=context.page_content,
            flag_id=context.metadata.get("flag_id"),
            title=context.metadata.get("title"),
        )
        for context in contexts
    ]


def group_contexts_by_flag_id(
    contexts: list[ContextDocs],
) -> dict[str, list[ContextDocs]]:
    grouped_contexts = defaultdict(list)
    for context in contexts:
        grouped_contexts[context.flag_id].append(context)

    return grouped_contexts
