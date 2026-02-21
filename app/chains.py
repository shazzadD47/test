from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from app.configs import settings
from app.prompts import (
    CONTEXT_PROMPT_TEMPLATE,
    QUESTION_GENERATION_PROMPT,
    QUESTION_PARAPHRASING_PROMPT,
)

gpt_3_model = ChatOpenAI(model=settings.GPT_3_MODEL, temperature=0.2)
gpt_4_model = ChatOpenAI(model=settings.GPT_4_TEXT_MODEL, temperature=0.2)


def prepare_question_generation_chain(
    parser: PydanticOutputParser,
    run_name: str = "Question Generation Chain",
    tags: list[str] = None,
) -> RunnableSerializable:
    prompt = PromptTemplate.from_template(QUESTION_GENERATION_PROMPT)

    chain = (
        {
            "question": itemgetter("question"),
            "output_format": itemgetter("output_format"),
        }
        | prompt
        | gpt_4_model
        | parser
    )

    chain = chain.with_config({"run_name": run_name, "tags": tags})

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


def prepare_context_given_summarization_chain(
    use_gpt_3: bool = False,
    run_name: str = "Question Generation Chain",
    tags: list[str] = None,
):
    context_prompt = PromptTemplate.from_template(CONTEXT_PROMPT_TEMPLATE)
    model = gpt_3_model if use_gpt_3 else gpt_4_model

    chain = (
        {
            "question": itemgetter("question"),
            "tables": itemgetter("tables"),
            "contexts": itemgetter("contexts"),
        }
        | context_prompt
        | model
        | StrOutputParser()
    )

    chain = chain.with_config({"run_name": run_name, "tags": tags})

    return chain
