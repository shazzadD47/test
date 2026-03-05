from datetime import datetime
from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.core.auto.chat_model import AutoChatModel
from app.v3.endpoints.general_extraction.constants import (
    query_generator_llm,
)
from app.v3.endpoints.general_extraction.langchain_schemas import (
    QueryList,
)
from app.v3.endpoints.general_extraction.prompts import (
    LABEL_QUESTION_GENERATION_FROM_ROOT_LABELS_PROMPT,
    LABEL_QUESTION_GENERATION_PROMPT,
    SYSTEM_INSTRUCTION,
)


def label_query_generator_agent(
    generate_for_dependent_labels: bool = False,
    schema: BaseModel = None,
):
    if schema:
        llm = query_generator_llm.with_structured_output(schema)
    else:
        llm = query_generator_llm.with_structured_output(QueryList)

    if generate_for_dependent_labels:
        prompt = PromptTemplate.from_template(
            SYSTEM_INSTRUCTION
            + "\n\n"
            + LABEL_QUESTION_GENERATION_FROM_ROOT_LABELS_PROMPT
        )
        today = datetime.now().strftime("%Y-%m-%d %A")
        prompt = prompt.partial(date=today)
        chain = (
            {
                "label_details": itemgetter("label_details"),
                "combination_of_answers_of_root_label": itemgetter(
                    "combination_of_answers_of_root_label"
                ),
                "total_questions": itemgetter("total_questions"),
            }
            | prompt
            | llm
        )
    else:
        prompt = PromptTemplate.from_template(
            SYSTEM_INSTRUCTION + "\n\n" + LABEL_QUESTION_GENERATION_PROMPT
        )
        today = datetime.now().strftime("%Y-%m-%d %A")
        prompt = prompt.partial(date=today)
        chain = {"label_details": itemgetter("label_details")} | prompt | llm
    return chain


_context_llm_cache: dict[str, object] = {}


def _get_or_create_context_llm(model_name: str):
    """
    Return a cached base LLM instance for context generation.
    """
    if model_name not in _context_llm_cache:
        if model_name.startswith("gemini"):
            _context_llm_cache[model_name] = AutoChatModel.from_model_name(
                model_name=model_name, thinking_level="low"
            )
        else:
            _context_llm_cache[model_name] = AutoChatModel.from_model_name(
                model_name=model_name,
            )
    return _context_llm_cache[model_name]


def label_context_generator_agent(
    model_name: str,
    schema: BaseModel = None,
):
    context_generator_llm = _get_or_create_context_llm(model_name)

    if schema:
        context_generator_llm = context_generator_llm.with_structured_output(
            schema,
        )
    return context_generator_llm
