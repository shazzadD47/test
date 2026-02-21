import logging
from typing import Any

from langchain_core.messages import BaseMessage
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.vector_store import VectorStoreRetriever
from app.logging import logger

logger = logger.getChild("retry_retriever")


async def ainvoke_retriever_with_retry(
    retriever: VectorStoreRetriever,
    query: str,
    retriever_kwargs: dict[str, Any] = None,
    max_retries: int = 3,
    reraise: bool = True,
    min_wait_seconds: int = 2,
    max_wait_seconds: int = 10,
    wait_exponential_multiplier: float = 0.75,
    retry_if_exception_types: tuple[Exception, ...] = None,
) -> BaseMessage:

    tenacity_kwargs = {
        "stop": stop_after_attempt(max_retries),
        "wait": wait_exponential(
            multiplier=wait_exponential_multiplier,
            min=min_wait_seconds,
            max=max_wait_seconds,
        ),
        "reraise": reraise,
        "before_sleep": before_sleep_log(logger, logging.WARNING),
    }

    if retry_if_exception_types:
        tenacity_kwargs["retry"] = retry_if_exception_type(retry_if_exception_types)

    retry_decorator = retry(**tenacity_kwargs)

    if retriever_kwargs is None:
        retriever_kwargs = {}

    @retry_decorator
    async def _ainvoke_retriever_with_retry() -> BaseMessage:

        return await retriever.ainvoke(query, **retriever_kwargs)

    return await _ainvoke_retriever_with_retry()


def invoke_retriever_with_retry(
    retriever: VectorStoreRetriever,
    query: str,
    retriever_kwargs: dict[str, Any] = None,
    max_retries: int = 3,
    reraise: bool = True,
    min_wait_seconds: int = 2,
    max_wait_seconds: int = 10,
    wait_exponential_multiplier: float = 0.75,
    retry_if_exception_types: tuple[Exception, ...] = None,
) -> BaseMessage:

    tenacity_kwargs = {
        "stop": stop_after_attempt(max_retries),
        "wait": wait_exponential(
            multiplier=wait_exponential_multiplier,
            min=min_wait_seconds,
            max=max_wait_seconds,
        ),
        "reraise": reraise,
        "before_sleep": before_sleep_log(logger, logging.WARNING),
    }

    if retry_if_exception_types:
        tenacity_kwargs["retry"] = retry_if_exception_type(retry_if_exception_types)

    retry_decorator = retry(**tenacity_kwargs)

    if retriever_kwargs is None:
        retriever_kwargs = {}

    @retry_decorator
    def _invoke_retriever_with_retry() -> BaseMessage:

        return retriever.invoke(query, **retriever_kwargs)

    return _invoke_retriever_with_retry()


def batch_invoke_retriever_with_retry(
    retriever: VectorStoreRetriever,
    queries: list[str],
    retriever_kwargs: dict[str, Any] = None,
    max_retries: int = 3,
    reraise: bool = True,
    min_wait_seconds: int = 2,
    max_wait_seconds: int = 10,
    wait_exponential_multiplier: float = 0.75,
    retry_if_exception_types: tuple[Exception, ...] = None,
) -> list[BaseMessage]:

    tenacity_kwargs = {
        "stop": stop_after_attempt(max_retries),
        "wait": wait_exponential(
            multiplier=wait_exponential_multiplier,
            min=min_wait_seconds,
            max=max_wait_seconds,
        ),
        "reraise": reraise,
        "before_sleep": before_sleep_log(logger, logging.WARNING),
    }

    if retry_if_exception_types:
        tenacity_kwargs["retry"] = retry_if_exception_type(retry_if_exception_types)

    retry_decorator = retry(**tenacity_kwargs)

    if retriever_kwargs is None:
        retriever_kwargs = {}

    @retry_decorator
    def _batch_invoke_retriever_with_retry() -> list[BaseMessage]:

        return retriever.batch(queries, **retriever_kwargs)

    return _batch_invoke_retriever_with_retry()
