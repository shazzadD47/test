from typing import Annotated

from langchain_core.tools import tool

from app.v3.endpoints.agent_chat.utils.retriever import retrieve_context
from app.v3.endpoints.report_generator.schema import ReportContextDocs


@tool
async def report_paper_context_retrieval(
    query: Annotated[
        str, "The query to search context from research papers for report writing"
    ],
    flag_id: Annotated[
        str | None, "Analogous to a file id. It is a unique id for a research paper."
    ] = None,
    project_id: Annotated[
        str | None,
        "The id of the project in which the research paper belongs to.",
    ] = None,
) -> list[ReportContextDocs]:
    """Get information from research papers for report writing purposes.
    Optimized for retrieving content suitable for professional report generation.

    flag_id is the id of the research paper like a file id.
    project_id is the id of the project in which the research paper belongs to.

    Provide a specific query to search for the most relevant information
    for your report.
    """
    contexts = await retrieve_context(query, flag_id, project_id, "document", k=15)
    return [
        ReportContextDocs(
            page_content=ctx.page_content, flag_id=ctx.flag_id, title=ctx.title
        )
        for ctx in contexts
    ]


@tool
async def report_code_context_retrieval(
    query: Annotated[
        str, "The query to search context from code files for report writing"
    ],
    flag_id: Annotated[
        str | None, "Analogous to a file id. It is a unique id for a code file."
    ] = None,
    project_id: Annotated[
        str | None, "The id of the project in which the code file belongs to."
    ] = None,
) -> list[ReportContextDocs]:
    """Get information from code files for report writing purposes.
    Optimized for retrieving code-related content suitable for professional
    report generation.

    flag_id is the id of the code file like a file id.
    project_id is the id of the project in which the code file belongs to.

    Provide a specific query to search for the most relevant code information
    for your report.
    """
    contexts = await retrieve_context(query, flag_id, project_id, "code", k=10)
    return [
        ReportContextDocs(
            page_content=ctx.page_content, flag_id=ctx.flag_id, title=ctx.title
        )
        for ctx in contexts
    ]
