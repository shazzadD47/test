import asyncio

from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.vector_store import VectorStore
from app.utils.texts import combine_langchain_contexts, get_unique_langchain_contexts
from app.v3.endpoints.autofill.prompts import ARMS_GIVEN_QUESTION_PROMPT


async def get_context_documents_from_question(
    context_chain: RunnableSerializable,
    retriever: VectorStoreRetriever,
    original_question: str,
    questions: list[str],
    table_contents: str = None,
    return_sources: bool = False,
):
    invokes = []
    for question in questions:
        context = retriever.ainvoke(question)

        invokes.append(context)

    contexts = await asyncio.gather(*invokes)
    contexts = [
        context for question_contexts in contexts for context in question_contexts
    ]
    contexts = get_unique_langchain_contexts(contexts)

    summarized_contexts = await context_chain.ainvoke(
        {
            "question": original_question,
            "tables": table_contents,
            "contexts": combine_langchain_contexts(contexts),
        }
    )

    if return_sources:
        sources = []
        for context_doc in contexts:
            source_details = {
                "page_content": context_doc.page_content,
                "source": context_doc.metadata.get("source"),
                "page": context_doc.metadata.get("page"),
                "flag_id": context_doc.metadata.get("flag_id"),
                "title": context_doc.metadata.get("title"),
            }
            sources.append(source_details)

        return summarized_contexts, sources

    return summarized_contexts


async def get_contexts_from_generated_questions(
    context_chain: RunnableSerializable,
    questions: list[list[str]],
    project_id: str,
    paper_id: str,
    is_root: bool,
    table_contents: str = None,
    return_sources: bool = False,
    tags: list[str] = None,
):
    retriever = VectorStore.get_retriever(
        search_kwargs={
            "k": 25 if is_root else 15,
            "filter": {
                "flag_id": paper_id.strip(),
                "project_id": project_id.strip(),
            },
        },
    )

    if tags:
        retriever = retriever.with_config({"tags": tags})

    invokes = []
    for question_list in questions:
        context = get_context_documents_from_question(
            context_chain,
            retriever,
            question_list[0],
            question_list,
            table_contents,
            return_sources=return_sources,
        )
        invokes.append(context)

    contexts = await asyncio.gather(*invokes)

    if return_sources:
        sources = [context[1] for context in contexts if context[1]]
        contexts = [context[0] for context in contexts]

        return contexts, sources

    return contexts


async def get_contexts_from_question(
    context_retriever_chain: RunnableSerializable,
    questions: list[str],
    is_root: bool,
    choice_tuples: list[tuple[str, str]] = None,
    table_contents: str = None,
):
    invokes = []
    for question in questions:
        if not is_root:
            question = ARMS_GIVEN_QUESTION_PROMPT.format(
                question=question,
                arms=choice_tuples,
            )

        context = context_retriever_chain.ainvoke(
            {
                "question": question,
                "tables": table_contents,
            }
        )

        invokes.append(context)

    contexts = await asyncio.gather(*invokes)
    return contexts
