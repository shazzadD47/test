import datetime
import os
import time
import traceback
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.auto.chat_model import AutoChatModel
from app.utils import check_if_null
from app.utils.cache import create_gemini_cache
from app.utils.download import download_files_from_flag_id
from app.utils.files import (
    create_file_input,
)
from app.utils.gemini import upload_file_to_gemini
from app.utils.llms import (
    get_message_text,
    invoke_chain_with_retry,
    invoke_llm_with_retry,
)
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints.covariate_extraction.chains import (
    prepare_context_summarization_chain,
)
from app.v3.endpoints.covariate_extraction.configs import settings as cov_settings
from app.v3.endpoints.covariate_extraction.helpers.utils import (
    extract_contexts_from_paper,
)
from app.v3.endpoints.covariate_extraction.logging import celery_logger as logger
from app.v3.endpoints.covariate_extraction.prompts import (
    PAPER_CONTEXT_RAG_PROMPT,
    SYSTEM_INSTRUCTION,
)


def process_pdf_file(
    flag_id: str,
) -> dict:
    file_details = {}
    try:
        pdf_path = download_files_from_flag_id(flag_id)
        if pdf_path and os.path.exists(pdf_path):
            file_details["pdf_path"] = pdf_path
        else:
            file_details["pdf_path"] = "N/A"
    except Exception as e:
        logger.error(f"Error getting file from flag id: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        file_details["pdf_path"] = "N/A"

    return file_details


def cache_pdf_file(
    pdf_paths: list[str] | str,
    flag_id: str,
    cache_name: str,
) -> dict:
    try:
        all_pdf_files = []
        if isinstance(pdf_paths, list):
            for pdf_path in pdf_paths:
                pdf_file = upload_file_to_gemini(pdf_path, flag_id)
                if pdf_file:
                    all_pdf_files.append(pdf_file)
        else:
            pdf_file = upload_file_to_gemini(pdf_paths, flag_id)
            if pdf_file:
                all_pdf_files.append(pdf_file)
    except Exception as e:
        logger.error(f"Error uploading pdf file to gemini: {e}")
        return None
    try:
        if len(all_pdf_files) > 0:
            system_instruction = SYSTEM_INSTRUCTION.format(
                date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            cache = create_gemini_cache(
                messages=all_pdf_files,
                model_name=cov_settings.CONTEXT_EXTRACTOR_LLM,
                cache_name=cache_name,
                system_instruction=system_instruction,
            )
            return cache
        return None
    except Exception as e:
        logger.error(f"Error creating gemini cache: {e}")
        return None


def get_summarized_contexts(
    project_id: str,
    flag_id: str,
    langfuse_session_id: str = None,
) -> list[dict[str, Any]]:
    start_time = time.time()
    file_details = process_pdf_file(flag_id)
    if (
        isinstance(file_details, dict)
        and "pdf_path" in file_details
        and not check_if_null(file_details["pdf_path"])
        and os.path.exists(file_details["pdf_path"])
    ):
        try:
            return {
                "summarized_contexts": _get_summarized_contexts_from_pdf(
                    file_details=file_details,
                    langfuse_session_id=langfuse_session_id,
                ),
                "total_time": time.time() - start_time,
            }
        except Exception as e:
            logger.error(f"Error getting summarized contexts for inputs from pdf: {e}")
            return {
                "summarized_contexts": _get_summarized_contexts_from_rag(
                    project_id=project_id,
                    flag_id=flag_id,
                ),
                "total_time": time.time() - start_time,
            }

    else:
        try:
            return {
                "summarized_contexts": _get_summarized_contexts_from_rag(
                    project_id=project_id,
                    flag_id=flag_id,
                    langfuse_session_id=langfuse_session_id,
                ),
                "total_time": time.time() - start_time,
            }
        except Exception as e:
            logger.error(f"Error getting summarized contexts for inputs from rag: {e}")
            return {
                "summarized_contexts": "N/A",
                "total_time": time.time() - start_time,
            }


def _get_summarized_contexts_from_pdf(
    file_details: dict | None = None,
    langfuse_session_id: str = None,
) -> dict[str, Any]:
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    model_name = cov_settings.CONTEXT_EXTRACTOR_LLM
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_instruction = SYSTEM_INSTRUCTION.format(date=current_time)
    pdf_input = create_file_input(file_details["pdf_path"], model_name)

    agent = AutoChatModel.from_model_name(model_name=model_name, temperature=0.2)
    result = invoke_llm_with_retry(
        llm=agent,
        config={"callbacks": [langfuse_handler]},
        messages=[
            SystemMessage(content=[{"type": "text", "text": system_instruction}]),
            HumanMessage(content=[pdf_input]),
            HumanMessage(
                content=[{"type": "text", "text": (PAPER_CONTEXT_RAG_PROMPT)}]
            ),
        ],
    )
    return get_message_text(result)


def _get_summarized_contexts_from_rag(
    project_id: str,
    flag_id: str,
    langfuse_session_id: str = None,
) -> dict[str, Any]:
    contexts = extract_contexts_from_paper(
        paper_id=flag_id,
        project_id=project_id,
        langfuse_session_id=langfuse_session_id,
    )

    context_summarization_chain = prepare_context_summarization_chain()
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)

    summarized_contexts = invoke_chain_with_retry(
        chain=context_summarization_chain,
        input={"contexts": contexts},
        config={"callbacks": [langfuse_handler]},
        max_retries=cov_settings.MAX_RETRIES,
        reraise=True,
    )
    return summarized_contexts
