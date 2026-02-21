from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable

from app.v3.endpoints.plot_digitizer.prompts import (
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
