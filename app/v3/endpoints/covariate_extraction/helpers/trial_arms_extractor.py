from typing import Any

from app.v3.endpoints.covariate_extraction.helpers.helpers import (
    extract_contexts_for_query_from_pdf,
    get_summarized_contexts,
)
from app.v3.endpoints.covariate_extraction.logging import celery_logger as logger


def get_arms_doses_contexts(
    project_id: str,
    flag_id: str,
    file_details: dict,
    supplementary_file_details: dict = None,
    supplementary_id: str = None,
    langfuse_handler=None,
) -> list[dict[str, Any]]:
    logger.info(f"Getting trial arms contexts for flag id: {flag_id}")
    arms_question = [
        "What are the arms/treatment groups in this study?",
        "What are the amounts of doses for each arm/treatment group?",
    ]

    summarization_question = (
        "What are the arms/treatment groups and their doses in this study?"
    )

    if file_details["pdf_path"] == "N/A":
        arms = get_summarized_contexts(
            arms_question,
            summarization_question,
            flag_id,
            project_id,
            supplementary_id,
            langfuse_handler,
        )

    else:
        arms = extract_contexts_for_query_from_pdf(
            flag_id=flag_id,
            file_details=file_details,
            supplementary_file_details=supplementary_file_details,
            questions=summarization_question,
            langfuse_handler=langfuse_handler,
        )

    return arms
