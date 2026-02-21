from operator import itemgetter

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_openai import ChatOpenAI

from app.configs import settings
from app.core.output_parsers import RetryOutputParser
from app.core.vector_store import VectorStore
from app.prompts import CONTEXT_PROMPT_TEMPLATE
from app.utils.texts import clean_langchain_contexts, combine_langchain_contexts
from app.v3.endpoints.autofill.constants import LANGSMITH_TAGS
from app.v3.endpoints.autofill.prompts import (
    INFO_EXTRACTION_PROMPT,
    INFO_EXTRACTION_WITH_ROOTS_PROMPT,
)


def prepare_context_retrieval_chain(
    paper_id: str, project_id: str, is_root: bool = False
) -> RunnableSerializable:
    retriever = VectorStore.get_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 25 if is_root else 15,
            "score_threshold": 0.6,
            "filter": {
                "flag_id": paper_id.strip(),
                "project_id": project_id.strip(),
            },
        },
    )

    llm = ChatOpenAI(model=settings.GPT_4_TEXT_MODEL, temperature=0.2)
    prompt = PromptTemplate.from_template(CONTEXT_PROMPT_TEMPLATE)
    retrieve_chain = (lambda x: x["question"]) | retriever | clean_langchain_contexts

    summarization_chain = (
        RunnablePassthrough.assign(
            contexts=(lambda x: combine_langchain_contexts(x["contexts"]))
        )
        .assign(question=lambda x: x["question"])
        .assign(tables=lambda x: x["tables"])
        | prompt
        | llm
        | StrOutputParser()
    )

    chain = (
        RunnablePassthrough.assign(contexts=retrieve_chain)
        .assign(answer=summarization_chain)
        .with_config(
            {
                "run_name": "Context Retrieval with Summarization Chain",
                "tags": LANGSMITH_TAGS,
            }
        )
    )

    return chain


def prepare_information_extraction_chain(
    parser: PydanticOutputParser, is_root: bool = False
) -> RunnableSerializable:
    llm = ChatOpenAI(model=settings.GPT_4_TEXT_MODEL, temperature=0.2)
    prompt = PromptTemplate.from_template(
        INFO_EXTRACTION_PROMPT if is_root else INFO_EXTRACTION_WITH_ROOTS_PROMPT
    )

    item_getters = {
        "contexts": itemgetter("contexts"),
        "output_format": itemgetter("output_format"),
    }

    if not is_root:
        item_getters["choice_tuples"] = itemgetter("choice_tuples")

    retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm, max_retries=2)

    part_chain = item_getters | prompt | llm

    chain = RunnableParallel(
        completion=part_chain, prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

    chain = chain.with_config(
        {"run_name": "Final Information Extraction Chain", "tags": LANGSMITH_TAGS}
    )

    return chain
