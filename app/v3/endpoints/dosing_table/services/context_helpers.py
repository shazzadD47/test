import base64
import os
import traceback
import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from app.core.auto.chat_model import AutoChatModel, AutoSdkChatModel
from app.utils.cache import create_gemini_cache, retrieve_gemini_cache
from app.utils.download import download_files_from_flag_id
from app.utils.gemini import upload_file_to_gemini
from app.utils.llms import (
    batch_invoke_chain_with_retry,
    get_message_text,
    invoke_llm_with_retry,
)
from app.utils.texts import (
    combine_langchain_contexts,
    combine_string_contexts,
    get_unique_langchain_contexts,
)
from app.v3.endpoints.dosing_table.chains import (
    context_retrieve_and_summarization_chain,
    study_design_understanding_chain,
)
from app.v3.endpoints.dosing_table.configs import settings as dosing_settings
from app.v3.endpoints.dosing_table.constants import (
    NO_FIGURE_INITIAL_RAG_QUESTIONS,
    context_agent,
)
from app.v3.endpoints.dosing_table.logging import celery_logger as logger
from app.v3.endpoints.dosing_table.prompts.with_figure import (
    CONTEXT_PROMPT_TEMPLATE_DOC,
)
from app.v3.endpoints.dosing_table.prompts.without_figure import (
    STUDY_DESIGN_UNDERSTANDING_PROMPT_DOC,
)
from app.v3.endpoints.dosing_table.utils import (
    get_context_retriever,
)


def process_pdf_file(
    flag_id: str,
    file_details: dict | None = None,
    cache_name: str = None,
) -> dict:
    model_name = dosing_settings.CONTEXT_GENERATOR_LLM
    model = AutoSdkChatModel.from_model_name(model_name)
    pdf_uploaded, pdf_saved = False, False
    if (
        file_details
        and isinstance(file_details, dict)
        and "gemini_file_name" in file_details
        and file_details["gemini_file_name"] != "N/A"
        and "pdf_path" in file_details
        and os.path.exists(file_details["pdf_path"])
    ):
        pdf_file = model.client.files.get(name=file_details["gemini_file_name"])
        pdf_uploaded = pdf_file is not None
        pdf_saved = os.path.exists(file_details["pdf_path"])
    else:
        file_details = {}

    if pdf_uploaded and pdf_saved:
        return file_details

    if not pdf_saved:
        try:
            pdf_path = download_files_from_flag_id(flag_id, return_type="path")
            if pdf_path and os.path.exists(pdf_path):
                pdf_saved = True
                file_details["pdf_path"] = pdf_path
        except Exception as e:
            logger.error(f"Error getting file from flag id: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            file_details["pdf_path"] = "N/A"
            return file_details

    if not pdf_uploaded:
        if pdf_saved:
            try:
                if model_name.startswith("gemini"):
                    pdf_file = upload_file_to_gemini(pdf_path, flag_id)
                    if pdf_file:
                        file_details["gemini_file_name"] = pdf_file.name
                        pdf_uploaded = True
                    else:
                        file_details["gemini_file_name"] = "N/A"
                        pdf_uploaded = False
            except Exception as e:
                logger.error(f"Error uploading file to gemini: {e}")
                file_details["gemini_file_name"] = "N/A"
                return file_details
        else:
            file_details["gemini_file_name"] = "N/A"
            return file_details

    if pdf_uploaded and pdf_saved:
        # cache the pdf file in gemini
        pdf_file = model.client.files.get(name=file_details["gemini_file_name"])
        try:
            logger.info(f"Caching pdf file: {pdf_path}")
            create_gemini_cache(
                messages=[pdf_file],
                model_name=model_name,
                cache_name=cache_name,
            )
            file_details["cache_name"] = cache_name
        except Exception as e:
            logger.error(f"Error caching pdf file: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            file_details["cache_name"] = "N/A"
    return file_details


def get_study_design_contexts(
    project_id: str,
    flag_id: str,
    file_details: dict,
    chain_configs: dict,
) -> list[dict[str, Any]]:
    logger.info(f"Getting study design contexts for flag id: {flag_id}")

    if file_details["pdf_path"] == "N/A":
        # use rag contexts
        logger.info(f"Getting rag contexts for flag id: {flag_id}")
        context_chain = get_context_retriever(flag_id, project_id)

        invoke_results = batch_invoke_chain_with_retry(
            chain=context_chain,
            questions=NO_FIGURE_INITIAL_RAG_QUESTIONS,
            config=chain_configs,
        )
        contexts = []
        for invoke_result in invoke_results:
            contexts.extend(invoke_result)

        logger.info(f"Got {len(contexts)} contexts")

        unique_contexts = get_unique_langchain_contexts(contexts)
        combined_contexts = combine_langchain_contexts(unique_contexts)

        logger.info(f"Got {len(unique_contexts)} unique contexts")
        study_design_chain = study_design_understanding_chain()
        study_design = study_design_chain.invoke(
            {"contexts": combined_contexts}, config=chain_configs
        )

        logger.info(f"Study design: {study_design}")
        return study_design

    else:
        if file_details["cache_name"] == "N/A":
            logger.info(f"Creating uncached inputs for flag id: {flag_id}")
            uncached_inputs = create_inputs_for_study_design_understanding(
                pdf_path=file_details["pdf_path"],
            )
            result = invoke_llm_with_retry(
                context_agent,
                messages=uncached_inputs,
                config=chain_configs,
            )
        else:
            logger.info(f"Getting cached inputs for flag id: {flag_id}")
            messages = [HumanMessage(content=STUDY_DESIGN_UNDERSTANDING_PROMPT_DOC)]
            cache = retrieve_gemini_cache(file_details["cache_name"])
            agent = AutoChatModel.from_model_name(
                model_name=dosing_settings.CONTEXT_GENERATOR_LLM,
                temperature=0.2,
                cached_content=cache.name,
            )

            result = invoke_llm_with_retry(
                agent,
                messages=messages,
                config=chain_configs,
            )

        return get_message_text(result)


def create_inputs_for_study_design_understanding(
    pdf_path: str = None,
) -> list[str]:
    with open(pdf_path, "rb") as f:
        pdf_data = base64.b64encode(f.read()).decode("utf-8")
    all_messages = [
        HumanMessage(
            content=[
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": pdf_data,
                    "mime_type": "application/pdf",
                    "filename": (f"{uuid.uuid4()}.pdf"),
                }
            ]
        ),
        HumanMessage(content=STUDY_DESIGN_UNDERSTANDING_PROMPT_DOC),
    ]
    return all_messages


def get_arms_routes_and_doses_contexts(
    project_id: str,
    flag_id: str,
    file_details: dict,
    chain_configs: dict,
) -> list[dict[str, Any]]:
    logger.info(f"Getting trial arms contexts for flag id: {flag_id}")
    arms_question = (
        "What are the arms/treatments in the study? "
        "What are the standard treatment names for each arm?"
    )
    routes_question = (
        "What are the route of administration of the following arms:" "\n\n{arms}"
    )
    dose_intervals_question = (
        "What is the interval between the doses for the following treatments:"
        "\n\n{arms}"
    )

    if file_details["pdf_path"] == "N/A":
        logger.info("Getting rag contexts for flag id: {flag_id}")
        context_chain = context_retrieve_and_summarization_chain(flag_id, project_id)

        arms = context_chain.invoke(
            {"question": arms_question},
            config=chain_configs,
        )

        routes = context_chain.invoke(
            {"question": routes_question.format(arms=arms)},
            config=chain_configs,
        )
        dose_intervals = context_chain.invoke(
            {"question": dose_intervals_question.format(arms=arms)},
            config=chain_configs,
        )

        contexts = [routes, dose_intervals]

        contexts.insert(0, arms)
        contexts = combine_string_contexts(contexts)

        logger.debug(f"Contexts: {contexts}\n{'=' * 88}")

    else:
        arms = extract_contexts_from_pdf(
            flag_id=flag_id,
            file_details=file_details,
            prompt=CONTEXT_PROMPT_TEMPLATE_DOC,
            question=arms_question,
            chain_configs=chain_configs,
        )
        routes = extract_contexts_from_pdf(
            flag_id=flag_id,
            file_details=file_details,
            prompt=CONTEXT_PROMPT_TEMPLATE_DOC,
            question=routes_question.format(arms=arms),
            chain_configs=chain_configs,
        )
        dose_intervals = extract_contexts_from_pdf(
            flag_id=flag_id,
            file_details=file_details,
            prompt=CONTEXT_PROMPT_TEMPLATE_DOC,
            question=dose_intervals_question.format(arms=arms),
            chain_configs=chain_configs,
        )
        contexts = [routes, dose_intervals]
        contexts.insert(0, arms)
        contexts = combine_string_contexts(contexts)

        logger.debug(f"Contexts: {contexts}\n{'=' * 88}")

    return contexts


def extract_contexts_from_pdf(
    flag_id: str,
    file_details: dict,
    prompt: str,
    question: str,
    chain_configs: dict,
) -> list[str]:
    if file_details["cache_name"] == "N/A":
        logger.info(f"Creating uncached inputs for flag id: {flag_id}")
        inputs = create_inputs_to_extract_with_figure_contexts(
            pdf_path=file_details["pdf_path"],
            prompt=prompt,
            question=question,
        )
        result = invoke_llm_with_retry(
            llm=context_agent,
            messages=inputs,
            config=chain_configs,
        )
    else:
        logger.info(f"Getting cached inputs for flag id: {flag_id}")
        messages = [
            HumanMessage(
                content=[{"type": "text", "text": prompt.format(question=question)}]
            )
        ]
        cache = retrieve_gemini_cache(file_details["cache_name"])
        agent = AutoChatModel.from_model_name(
            model_name=dosing_settings.CONTEXT_GENERATOR_LLM,
            temperature=0.2,
            cached_content=cache.name,
        )
        result = invoke_llm_with_retry(
            llm=agent,
            messages=messages,
            config=chain_configs,
        )
    return get_message_text(result)


def create_inputs_to_extract_with_figure_contexts(
    pdf_path: str = None,
    prompt: str = None,
    question: str = None,
) -> list[str]:
    with open(pdf_path, "rb") as f:
        pdf_data = base64.b64encode(f.read()).decode("utf-8")
    all_messages = [
        HumanMessage(
            content=[
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": pdf_data,
                    "mime_type": "application/pdf",
                    "filename": (f"{uuid.uuid4()}.pdf"),
                }
            ]
        ),
        HumanMessage(content=prompt.format(question=question)),
    ]
    return all_messages
