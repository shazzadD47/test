from typing import Literal

from langchain_core.documents import Document

from app.core.vector_store import VectorStore
from app.utils.retrievers import (
    ainvoke_retriever_with_retry,
    batch_invoke_retriever_with_retry,
    invoke_retriever_with_retry,
)


async def get_context_docs(
    query: str,
    top_k: int = 20,
    flag_id: str = None,
    project_id: str = None,
) -> list[Document]:
    filter_params = {}
    if flag_id:
        filter_params["flag_id"] = flag_id
    if project_id:
        filter_params["project_id"] = project_id

    filter_params["file_type"] = "document"

    retriever = VectorStore.get_retriever(
        search_kwargs={"filter": filter_params, "k": top_k}
    )

    contexts = await ainvoke_retriever_with_retry(retriever=retriever, query=query)

    return contexts


def sync_get_context_docs(
    query: str,
    top_k: int = 20,
    flag_id: str = None,
    project_id: str = None,
) -> list[Document]:
    filter_params = {}
    if flag_id:
        filter_params["flag_id"] = flag_id
    if project_id:
        filter_params["project_id"] = project_id

    filter_params["file_type"] = "document"

    retriever = VectorStore.get_retriever(
        search_kwargs={"filter": filter_params, "k": top_k}
    )

    contexts = invoke_retriever_with_retry(retriever=retriever, query=query)

    return contexts


def sync_batch_get_context_docs(
    queries: list[str],
    top_k: int = 20,
    flag_id: str = None,
    project_id: str = None,
    retriever_kwargs: dict = None,
) -> list[Document]:
    filter_params = {}
    if flag_id:
        filter_params["flag_id"] = flag_id
    if project_id:
        filter_params["project_id"] = project_id

    filter_params["file_type"] = "document"

    retriever = VectorStore.get_retriever(
        search_kwargs={"filter": filter_params, "k": top_k}
    )

    return batch_invoke_retriever_with_retry(
        retriever, queries, retriever_kwargs=retriever_kwargs
    )


def select_contexts(
    contexts: list[list[str]],
    top_k: int = 20,
) -> list[str]:
    if len(contexts) == 1:
        return contexts[0]

    selected_contexts = []
    if len(contexts) == 2:
        count_limit = len(contexts[0]) // 2
        selected_contexts = contexts[0] + contexts[1][: count_limit // 2]
        unique_contexts = []
        for context in selected_contexts:
            if context not in unique_contexts:
                unique_contexts.append(context)
        return unique_contexts

    for _count, context in enumerate(zip(*contexts)):
        for single_context in context:
            if single_context not in selected_contexts:
                selected_contexts.append(single_context)
        if len(set(selected_contexts)) >= top_k:
            break

    unique_contexts = []
    for context in selected_contexts:
        if context not in unique_contexts:
            unique_contexts.append(context)

    return unique_contexts


def _get_chunk_id_sort_key(chunk_id):
    """Get sort key for chunk_id that handles both numeric and string values.

    Returns a tuple for sorting where:
    - First element: True if chunk_id is None (puts None values at the end)
    - Second element: numeric value for proper numeric sorting, or inf for non-numeric
    - Third element: string value for alphabetic sorting of non-numeric chunk_ids
    """
    if chunk_id is None:
        return (True, float("inf"), "")
    # Try numeric conversion first for proper numeric sorting
    try:
        return (False, int(chunk_id), "")
    except (ValueError, TypeError):
        # Non-numeric chunk_id: sort alphabetically after numeric ones
        return (False, float("inf"), str(chunk_id))


def retrieve_all_contexts(
    flag_id: str = None,
    project_id: str = None,
    file_type: Literal["document", "code"] = None,
) -> str:
    """Retrieve all chunks for a given flag_id without similarity search."""
    filter_params = {}
    if flag_id:
        filter_params["flag_id"] = flag_id
    if project_id:
        filter_params["project_id"] = project_id
    if file_type:
        filter_params["file_type"] = file_type

    documents = VectorStore.get_all_by_filter(filters=filter_params)
    # if chunk_id is in the metadata, then sort chunks by chunk_id
    # in ascending order. If some chunks do not have chunk_id, then put them at the end
    sorted_documents = sorted(
        documents,
        key=lambda x: _get_chunk_id_sort_key(x.metadata.get("chunk_id")),
    )
    return "\n".join(
        [f"{i+1}) {doc.page_content}" for i, doc in enumerate(sorted_documents)]
    )
