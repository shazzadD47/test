from operator import itemgetter

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.configs import settings
from app.core.vector_store import VectorStore
from app.utils.texts import combine_langchain_contexts
from app.v3.endpoints.dosing_table.configs import settings as dosing_settings
from app.v3.endpoints.dosing_table.prompts.with_figure import (
    CONTEXT_PROMPT_TEMPLATE_RAG,
    DOSE_CALCULATION_PROMPT,
)
from app.v3.endpoints.dosing_table.prompts.without_figure import (
    STUDY_DESIGN_UNDERSTANDING_PROMPT_RAG,
)

llm_gpt = ChatOpenAI(model=settings.GPT_4_TEXT_MODEL, temperature=0.2)
llm_reasoning = ChatOpenAI(model=dosing_settings.REASONING_LLM)

llm_claude = ChatAnthropic(
    model=settings.CLAUDE_MODEL_ID,
    temperature=0.2,
    max_tokens_to_sample=4096,
    api_key=settings.ANTHROPIC_API_KEY,
)


def context_retrieve_and_summarization_chain(
    paper_id: str, project_id: str
) -> RunnableSerializable:
    retriever = VectorStore.get_retriever(
        search_kwargs={
            "k": 15,
            "filter": {
                "flag_id": paper_id.strip(),
                "project_id": project_id.strip(),
            },
        }
    )

    chain = (
        {
            "contexts": itemgetter("question") | retriever | combine_langchain_contexts,
            "question": itemgetter("question"),
        }
        | PromptTemplate.from_template(CONTEXT_PROMPT_TEMPLATE_RAG)
        | llm_gpt
        | StrOutputParser()
    )

    return chain


def prepare_table_chain(
    prompt_template: str, output_schema: type[BaseModel]
) -> RunnableSerializable:
    prompt = PromptTemplate.from_template(prompt_template)

    chain = (
        {
            "contexts": itemgetter("contexts"),
        }
        | prompt
        | llm_reasoning.with_structured_output(output_schema)
    )

    return chain


def study_design_understanding_chain() -> RunnableSerializable:
    llm = ChatOpenAI(model=dosing_settings.CONTEXT_GENERATOR_LLM)

    chain = (
        {
            "contexts": itemgetter("contexts"),
        }
        | PromptTemplate.from_template(STUDY_DESIGN_UNDERSTANDING_PROMPT_RAG)
        | llm
        | StrOutputParser()
    )

    return chain


def dose_calculation_chain() -> RunnableSerializable:
    chain = (
        {
            "contexts": itemgetter("contexts"),
        }
        | PromptTemplate.from_template(DOSE_CALCULATION_PROMPT)
        | llm_gpt
        | StrOutputParser()
    )

    return chain
