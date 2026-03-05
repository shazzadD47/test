from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.configs import settings
from app.core.auto.chat_model import AutoChatModel
from app.v3.endpoints.covariate_extraction.configs import settings as cov_settings
from app.v3.endpoints.covariate_extraction.fixers import (
    COVTableOutputFixer,
    COVTypeOutputFixer,
)
from app.v3.endpoints.covariate_extraction.prompts import (
    ANSWER_FROM_CONTEXT_PROMPT,
    FIND_SPECIFIC_ANSWER_PROMPT,
    GET_TABLE_COVARIATES_WITH_PAPER_SUMMARY_PROMPT,
    MAP_COVARIATES_PROMPT,
    QA_ON_CONTEXT_PROMPT,
    QUESTION_PARAPHRASING_PROMPT,
    SUMMARIZE_PAPER_CONTEXT_PROMPT,
    SYSTEM_INSTRUCTION,
)

_llm_gpt = None


def _get_llm_gpt():
    global _llm_gpt
    if _llm_gpt is None:
        _llm_gpt = ChatOpenAI(model=cov_settings.LLM_NAME)
    return _llm_gpt


def prepare_context_summarization_chain() -> RunnableSerializable:
    chain = (
        {
            "contexts": itemgetter("contexts"),
        }
        | PromptTemplate.from_template(
            SYSTEM_INSTRUCTION + SUMMARIZE_PAPER_CONTEXT_PROMPT
        )
        | _get_llm_gpt()
        | StrOutputParser()
    )

    return chain


def prepare_table_extraction_chain(schema: type[BaseModel]) -> RunnableSerializable:
    structured_llm = _get_llm_gpt().with_structured_output(schema=schema)
    chain = (
        {
            "example_covariate_table": itemgetter("example_covariate_table"),
            "example_covariate_output": itemgetter("example_covariate_output"),
            "contexts": itemgetter("contexts"),
            "table_contents": itemgetter("table_contents"),
            "format": itemgetter("format"),
            "mandatory_columns": itemgetter("mandatory_columns"),
        }
        | PromptTemplate.from_template(GET_TABLE_COVARIATES_WITH_PAPER_SUMMARY_PROMPT)
        | structured_llm
        | COVTableOutputFixer()
    )
    return chain


def prepare_table_column_map_chain(
    schema: type[BaseModel],
    standard_covariates_csv: str,
) -> RunnableSerializable:
    json_schema = schema.model_json_schema()
    structured_llm = _get_llm_gpt().with_structured_output(schema=json_schema)
    chain = (
        {
            "covariate_list": itemgetter("covariate_list"),
            "cov_list_csv_string": itemgetter("cov_list_csv_string"),
            "output_format": itemgetter("output_format"),
        }
        | PromptTemplate.from_template(MAP_COVARIATES_PROMPT)
        | structured_llm
        | COVTypeOutputFixer(standard_covariates=standard_covariates_csv)
    )

    return chain


def find_specific_answer_chain() -> RunnableSerializable:
    chain = (
        {
            "label_name": itemgetter("label_name"),
            "label_description": itemgetter("label_description"),
            "answer": itemgetter("answer"),
        }
        | PromptTemplate.from_template(FIND_SPECIFIC_ANSWER_PROMPT)
        | _get_llm_gpt()
        | StrOutputParser()
    )
    return chain


def table_definition_from_contexts_chain(llm: BaseChatModel, schema: BaseModel):
    ans_context_prompt = PromptTemplate.from_template(ANSWER_FROM_CONTEXT_PROMPT)

    chain = (
        {
            "contexts": itemgetter("contexts"),
        }
        | ans_context_prompt
        | llm.with_structured_output(schema)
    )
    return chain


def prepare_question_rephrasing_chain(llm: BaseChatModel) -> RunnableSerializable:
    prompt = PromptTemplate.from_template(QUESTION_PARAPHRASING_PROMPT)

    chain = (
        {
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def prepare_contexts_given_qa_chain(llm: BaseChatModel) -> RunnableSerializable:
    chain = (
        {
            "contexts": itemgetter("contexts"),
            "questions": itemgetter("questions"),
        }
        | PromptTemplate.from_template(QA_ON_CONTEXT_PROMPT)
        | llm
        | StrOutputParser()
    )

    return chain


def _get_llm_claude():
    return AutoChatModel.from_model_name(
        settings.CLAUDE_MODEL_ID,
        temperature=0.2,
    )


def _get_llm():
    return AutoChatModel.from_model_name(cov_settings.LLM_NAME, temperature=0.2)


class _LazyChain:
    """Lazy wrapper that defers LLM/chain creation until first use,
    avoiding heavy allocations at module import time in forked workers."""

    def __init__(self, factory):
        self._factory = factory
        self._instance = None

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = self._factory()
        return getattr(self._instance, name)

    def invoke(self, *args, **kwargs):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance.invoke(*args, **kwargs)

    def batch(self, *args, **kwargs):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance.batch(*args, **kwargs)


context_qa_chain = _LazyChain(lambda: prepare_contexts_given_qa_chain(llm=_get_llm()))
context_qa_chain_claude = _LazyChain(
    lambda: prepare_contexts_given_qa_chain(llm=_get_llm_claude())
)
rephrase_chain = _LazyChain(lambda: prepare_question_rephrasing_chain(llm=_get_llm()))
