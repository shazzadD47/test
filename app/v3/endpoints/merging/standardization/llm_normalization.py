from operator import itemgetter
from uuid import uuid4

from langchain_core.prompts import PromptTemplate
from langfuse import observe

from app.core.auto import AutoChatModel
from app.utils.llms import invoke_chain_with_retry
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints.merging.configs import settings as merging_settings
from app.v3.endpoints.merging.langchain_schemas import (
    DVNormalizer,
    RegimenNormalizer,
    UnitNormalizer,
    ValueFixer,
)
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.standardization.constants import (
    MAX_RETRIES,
    REGIMEN_ABBREVIATIONS,
    SIMILAR_REGIMEN_MAPPING,
)
from app.v3.endpoints.merging.standardization.prompts import (
    STANDARDIZE_DV_VALUES_PROMPT,
    STANDARDIZE_REGIMEN_PROMPT,
    STANDARDIZE_VALUES_PROMPT,
    UNIT_NORMALIZATION_PROMPT,
)

logger = logger.getChild("standardization")


@observe()
def standardize_unit_llm(
    values: list[str], langfuse_session_id: str = None
) -> list[str]:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    llm = AutoChatModel.from_model_name(
        merging_settings.LLM_MODEL,
    )
    structured_llm = llm.with_structured_output(schema=UnitNormalizer)
    prompt_template = PromptTemplate.from_template(UNIT_NORMALIZATION_PROMPT)
    chain = (
        {
            "values": itemgetter("values"),
        }
        | prompt_template
        | structured_llm
    )
    result = invoke_chain_with_retry(
        chain,
        {"values": values},
        max_retries=MAX_RETRIES,
        config={"callbacks": [langfuse_handler]},
    )
    return result.model_dump()["normalized_units"]


@observe()
def standardize_values_llm(
    values: list[str], langfuse_session_id: str = None
) -> list[str]:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    llm = AutoChatModel.from_model_name(
        merging_settings.LLM_MODEL,
    )
    structured_llm = llm.with_structured_output(schema=ValueFixer)
    prompt_template = PromptTemplate.from_template(STANDARDIZE_VALUES_PROMPT)
    chain = (
        {
            "values": itemgetter("values"),
        }
        | prompt_template
        | structured_llm
    )
    result = invoke_chain_with_retry(
        chain,
        {"values": values},
        max_retries=MAX_RETRIES,
        config={"callbacks": [langfuse_handler]},
    )
    return result.model_dump()["fixed_values"]


@observe()
def standardize_regimen_llm(
    values: list[str], langfuse_session_id: str = None
) -> list[str]:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    llm = AutoChatModel.from_model_name(
        merging_settings.LLM_MODEL,
    )
    structured_llm = llm.with_structured_output(schema=RegimenNormalizer)
    prompt_template = PromptTemplate.from_template(STANDARDIZE_REGIMEN_PROMPT)
    chain = (
        {
            "values": itemgetter("values"),
            "REGIMEN_ABBREVIATIONS": itemgetter("REGIMEN_ABBREVIATIONS"),
            "SIMILAR_REGIMEN_MAPPING": itemgetter("SIMILAR_REGIMEN_MAPPING"),
        }
        | prompt_template
        | structured_llm
    )
    result = invoke_chain_with_retry(
        chain,
        {
            "values": values,
            "REGIMEN_ABBREVIATIONS": REGIMEN_ABBREVIATIONS,
            "SIMILAR_REGIMEN_MAPPING": SIMILAR_REGIMEN_MAPPING,
        },
        max_retries=MAX_RETRIES,
        config={"callbacks": [langfuse_handler]},
    )
    return result.model_dump()["normalized_regimens"]


@observe()
def standardize_dv_values_llm(
    values: list[str],
    constant_values: list[str],
    langfuse_session_id: str = None,
) -> list[str]:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    llm = AutoChatModel.from_model_name(
        merging_settings.LLM_MODEL,
    )
    structured_llm = llm.with_structured_output(schema=DVNormalizer)
    prompt_template = PromptTemplate.from_template(STANDARDIZE_DV_VALUES_PROMPT)
    chain = (
        {
            "input_values": itemgetter("input_values"),
            "constant_values": itemgetter("constant_values"),
        }
        | prompt_template
        | structured_llm
    )
    result = invoke_chain_with_retry(
        chain,
        {"input_values": values, "constant_values": constant_values},
        max_retries=MAX_RETRIES,
        config={"callbacks": [langfuse_handler]},
    )
    return result.model_dump()["normalized_values"]
