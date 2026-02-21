from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel

from app.v3.endpoints.iterative_autofill.langchain_schemas import (
    AnswerModel,
)
from app.v3.endpoints.iterative_autofill.prompts import (
    ANSWER_FROM_CONTEXT_PROMPT,
    NESTED_LABEL_CONTEXT_INSTRUCTION,
    QA_ON_CONTEXT_ADDITIONAL_INSTRUCTIONS,
    QA_ON_CONTEXT_PROMPT,
    QUESTION_PARAPHRASING_PROMPT,
)


def prepare_question_rephrasing_chain(
    llm: BaseChatModel,
) -> RunnableSerializable:
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


def prepare_contexts_given_qa_chain(
    llm: BaseChatModel,
    has_parents: bool = False,
    output_schema: type[BaseModel] = None,
) -> RunnableSerializable:
    if output_schema is None:
        output_schema = AnswerModel

    if has_parents:
        context_prompt = (
            QA_ON_CONTEXT_PROMPT
            + "\n"
            + QA_ON_CONTEXT_ADDITIONAL_INSTRUCTIONS
            + "\n"
            + NESTED_LABEL_CONTEXT_INSTRUCTION
        )
    else:
        context_prompt = (
            QA_ON_CONTEXT_PROMPT + "\n" + QA_ON_CONTEXT_ADDITIONAL_INSTRUCTIONS
        )

    prompt = PromptTemplate.from_template(context_prompt)

    chain = {
        "contexts": itemgetter("contexts"),
        "questions": itemgetter("questions"),
        "total_number_of_chunks": itemgetter("total_number_of_chunks"),
    }

    if has_parents:
        chain["parent_label_answers"] = itemgetter("parent_label_answers")

    runnable_chain = chain | prompt
    runnable_chain = runnable_chain | llm.with_structured_output(output_schema)

    return runnable_chain


def table_definition_from_contexts_chain(
    llm: BaseChatModel,
    output_schema: type[BaseModel],
) -> RunnableSerializable:
    prompt = PromptTemplate.from_template(ANSWER_FROM_CONTEXT_PROMPT)

    chain = (
        {
            "contexts": itemgetter("contexts"),
            "labels_with_answers": itemgetter("labels_with_answers"),
        }
        | prompt
        | llm.with_structured_output(output_schema)
    )

    return chain
