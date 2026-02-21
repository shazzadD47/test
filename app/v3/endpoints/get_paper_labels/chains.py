from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable

from app.v3.endpoints.get_paper_labels.prompts import (
    ANSWER_FROM_CONTEXT_PROMPT,
    QA_ON_CONTEXT_PROMPT,
    QUESTION_PARAPHRASING_PROMPT,
)


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
            "question": itemgetter("question"),
        }
        | PromptTemplate.from_template(QA_ON_CONTEXT_PROMPT)
        | llm
        | StrOutputParser()
    )

    return chain


def table_definition_from_contexts_chain(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
):
    ans_context_prompt = PromptTemplate.from_template(ANSWER_FROM_CONTEXT_PROMPT)

    chain = (
        {
            "contexts": itemgetter("contexts"),
            "output_instructions": itemgetter("output_instructions"),
        }
        | ans_context_prompt
        | llm
        | parser
    )
    return chain
