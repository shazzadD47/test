from app.v3.endpoints.covariate_extraction.helpers.helpers import (
    extract_contexts_for_query_from_pdf,
)
from app.v3.endpoints.covariate_extraction.rag_tasks import get_context_docs_task


def extract_contexts_of_labels(
    flag_id: str,
    project_id: str,
    questions: list[str],
    file_details: dict,
    supplementary_file_details: dict = None,
    supplementary_id: str = None,
    langfuse_handler=None,
) -> tuple[list[str], list[str]] | list[str]:
    if file_details["pdf_path"] != "N/A":
        all_contexts = extract_contexts_for_query_from_pdf(
            flag_id,
            file_details,
            supplementary_file_details,
            questions,
            langfuse_handler,
        )
        return all_contexts
    else:
        contexts_from_main_doc = get_context_docs_task(
            queries=questions,
            flag_id=flag_id,
            project_id=project_id,
            top_k=20,
        )
        if supplementary_id:
            contexts_from_supp = get_context_docs_task(
                queries=questions,
                flag_id=flag_id,
                project_id=project_id,
                supplementary_id=supplementary_id,
                top_k=20,
            )
        return contexts_from_main_doc, contexts_from_supp
