import asyncio
import time

from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, create_model

from app.configs import settings
from app.constants import d_type_map
from app.core.auto import AutoCallbackHandler, AutoChatModel
from app.core.callbacks.anthropic import AnthropicTokenUsageCallbackHandler
from app.v3.endpoints.get_paper_labels.chains import (
    prepare_contexts_given_qa_chain,
    prepare_question_rephrasing_chain,
    table_definition_from_contexts_chain,
)
from app.v3.endpoints.get_paper_labels.configs import (
    settings as get_paper_label_settings,
)
from app.v3.endpoints.get_paper_labels.constants import paper_dependent_fields
from app.v3.endpoints.get_paper_labels.helpers import (
    combine_contexts,
    get_contexts_from_rag,
    get_questions_from_table_structure,
    get_summarized_contexts,
)
from app.v3.endpoints.get_paper_labels.logging import logger
from app.v3.endpoints.get_paper_labels.schemas import PaperLabelsTableField

REDIS_LIVE_TIME = settings.CACHE_DAY * 24 * 60 * 60


async def dynamic_paper_metadata_extraction_service(
    paper_id: str,
    project_id: str,
    table_structure: list[PaperLabelsTableField],
):

    start_time = time.time()
    claude_usage = AnthropicTokenUsageCallbackHandler()
    openai_usage = OpenAICallbackHandler()
    usage_handler = AutoCallbackHandler.from_model_name(
        get_paper_label_settings.LLM_NAME
    )

    if isinstance(usage_handler, OpenAICallbackHandler):
        usage_handler = openai_usage
    elif isinstance(usage_handler, AnthropicTokenUsageCallbackHandler):
        usage_handler = claude_usage

    llm = AutoChatModel.from_model_name(
        get_paper_label_settings.LLM_NAME,
        temperature=0.2,
        max_tokens=16384,
        callbacks=[usage_handler],
    )

    context_qa_chain = prepare_contexts_given_qa_chain(llm=llm)
    rephrase_chain = prepare_question_rephrasing_chain(llm=llm)
    logger.debug(f"Chains initialized in {time.time() - start_time} seconds.")

    # get questions from table definition for paper rag
    start_time = time.time()
    paper_labels_in_table = [
        field.name
        for field in table_structure
        if field.c_type in ["paper_label", "paper_labels"]
    ]
    paper_labels_in_table += paper_dependent_fields

    all_questions = await get_questions_from_table_structure(
        rephrase_chain, table_structure, paper_labels_in_table
    )

    logger.debug(f"Questions generated in {time.time() - start_time} seconds.")

    # get contexts from rag
    start_time = time.time()

    invokes = []
    for field in table_structure:
        if field.name in paper_labels_in_table:
            invokes.append(
                get_contexts_from_rag(
                    all_questions[field.name]["retrieval_questions"],
                    paper_id,
                    project_id,
                )
            )
    all_contexts = await asyncio.gather(*invokes)
    for i, field in enumerate(table_structure):
        if field.name in paper_labels_in_table:
            all_questions[field.name]["contexts"] = all_contexts[i]
    logger.debug(f"Contexts retrieved in {time.time() - start_time} seconds.")

    start_time = time.time()
    questions_with_summarized_contexts = await get_summarized_contexts(
        context_qa_chain, all_questions
    )
    logger.debug(f"Contexts summarized in {time.time() - start_time} seconds.")

    paper_contexts = combine_contexts(
        questions_with_summarized_contexts, paper_labels_in_table
    )

    # retrieve all table definition values from RAG
    start_time = time.time()
    paper_labels = {
        field.name: (
            d_type_map[field.d_type] | None,
            Field(..., description=field.description),
        )
        for field in table_structure
        if field.name in paper_labels_in_table
    }
    PaperLineLabels = create_model("PaperLineLabels", **paper_labels)
    parser = PydanticOutputParser(pydantic_object=PaperLineLabels)
    format_instructions = parser.get_format_instructions()
    table_values_from_contexts_chain = table_definition_from_contexts_chain(llm, parser)
    result_from_rag = await table_values_from_contexts_chain.ainvoke(
        {
            "contexts": paper_contexts,
            "output_instructions": format_instructions,
        }
    )
    result_from_rag = result_from_rag.dict()
    logger.debug(
        f"Paper Labels from RAG retrieved in {time.time() - start_time} seconds."
    )

    return {
        "message": "successfully extracted data",
        "data": result_from_rag,
        "usage": {
            "anthropic": {
                "total_tokens": claude_usage.total_tokens,
                "prompt_tokens": claude_usage.prompt_tokens,
                "completion_tokens": claude_usage.completion_tokens,
                "cost": claude_usage.total_cost,
            },
            "openai": {
                "total_tokens": openai_usage.total_tokens,
                "prompt_tokens": openai_usage.prompt_tokens,
                "completion_tokens": openai_usage.completion_tokens,
                "cost": openai_usage.total_cost,
            },
        },
    }
