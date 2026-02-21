import asyncio
import contextvars
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, TypeVar

from google.genai import types
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.configs import settings
from app.constants import tenacity_kwargs
from app.core.auto.chat_model import AutoSdkChatModel
from app.core.auto.token_config import MAX_TOKEN_MAPPING
from app.logging import logger
from app.utils.cache import retrieve_gemini_cache

logger = logger.getChild("retry_llms")

Input = TypeVar("Input", contravariant=True)


async def ainvoke_llm_with_retry(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    config: Any = None,
    llm_kwargs: dict[str, Any] = None,
    max_retries: int = 3,
    reraise: bool = True,
    min_wait_seconds: int = 2,
    max_wait_seconds: int = 10,
    wait_exponential_multiplier: float = 0.75,
    retry_if_exception_types: tuple[Exception, ...] = None,
    timeout: int = None,
) -> BaseMessage:

    if timeout is None:
        timeout = settings.LLM_INVOKE_TIMEOUT

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

    # Only set retry key if retry_if_exception_types is explicitly provided.
    # Otherwise, tenacity uses its default behavior (retry on all exceptions).
    if retry_if_exception_types:
        retry_exceptions = (asyncio.TimeoutError,) + retry_if_exception_types
        tenacity_kwargs["retry"] = retry_if_exception_type(retry_exceptions)

    retry_decorator = retry(**tenacity_kwargs)

    if llm_kwargs is None:
        llm_kwargs = {}

    @retry_decorator
    async def _ainvoke_llm_with_retry() -> BaseMessage:
        try:
            return await asyncio.wait_for(
                llm.ainvoke(messages, config=config, **llm_kwargs), timeout=timeout
            )
        except TimeoutError:
            logger.warning(
                f"LLM invocation timed out after {timeout} seconds, retrying..."
            )
            raise

    return await _ainvoke_llm_with_retry()


def invoke_llm_with_retry(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    config: Any = None,
    llm_kwargs: dict[str, Any] = None,
    max_retries: int = 3,
    reraise: bool = True,
    min_wait_seconds: int = 2,
    max_wait_seconds: int = 10,
    wait_exponential_multiplier: float = 0.75,
    retry_if_exception_types: tuple[Exception, ...] = None,
    timeout: int = None,
) -> BaseMessage:

    if timeout is None:
        timeout = settings.LLM_INVOKE_TIMEOUT

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

    # Only set retry key if retry_if_exception_types is explicitly provided.
    # Otherwise, tenacity uses its default behavior (retry on all exceptions).
    if retry_if_exception_types:
        retry_exceptions = (
            FuturesTimeoutError,
            TimeoutError,
        ) + retry_if_exception_types
        tenacity_kwargs["retry"] = retry_if_exception_type(retry_exceptions)

    retry_decorator = retry(**tenacity_kwargs)

    if llm_kwargs is None:
        llm_kwargs = {}

    @retry_decorator
    def _invoke_llm_with_retry() -> BaseMessage:
        executor = ThreadPoolExecutor(max_workers=1)
        # Copy the current context so ContextVar-based callbacks
        # (e.g. langchain cost tracking) are visible in the worker thread.
        ctx = contextvars.copy_context()
        try:
            future = executor.submit(
                ctx.run, llm.invoke, messages, config, **llm_kwargs
            )
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning(
                f"LLM invocation timed out after {timeout} seconds, retrying..."
            )
            raise
        finally:
            # Use wait=False to avoid blocking on timeout.
            # Running threads can't be cancelled in Python, but we won't block on them.
            executor.shutdown(wait=False, cancel_futures=True)

    return _invoke_llm_with_retry()


async def ainvoke_chain_with_retry(
    chain: RunnableSerializable,
    input: Input,
    config: RunnableConfig | None = None,
    chain_kwargs: Any = None,
    max_retries: int = 3,
    reraise: bool = True,
    min_wait_seconds: int = 2,
    max_wait_seconds: int = 10,
    wait_exponential_multiplier: float = 0.75,
    retry_if_exception_types: tuple[Exception, ...] = None,
    timeout: int = None,
) -> BaseMessage:

    if timeout is None:
        timeout = settings.LLM_INVOKE_TIMEOUT

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

    # Only set retry key if retry_if_exception_types is explicitly provided.
    # Otherwise, tenacity uses its default behavior (retry on all exceptions).
    if retry_if_exception_types:
        retry_exceptions = (asyncio.TimeoutError,) + retry_if_exception_types
        tenacity_kwargs["retry"] = retry_if_exception_type(retry_exceptions)

    retry_decorator = retry(**tenacity_kwargs)

    if chain_kwargs is None:
        chain_kwargs = {}

    @retry_decorator
    async def _ainvoke_chain_with_retry() -> BaseMessage:
        try:
            return await asyncio.wait_for(
                chain.ainvoke(input, config, **chain_kwargs), timeout=timeout
            )
        except TimeoutError:
            logger.warning(
                f"Chain invocation timed out after {timeout} seconds, retrying..."
            )
            raise

    return await _ainvoke_chain_with_retry()


def invoke_chain_with_retry(
    chain: RunnableSerializable,
    input: Input,
    config: RunnableConfig | None = None,
    chain_kwargs: Any = None,
    max_retries: int = 3,
    reraise: bool = True,
    min_wait_seconds: int = 2,
    max_wait_seconds: int = 10,
    wait_exponential_multiplier: float = 0.75,
    retry_if_exception_types: tuple[Exception, ...] = None,
    timeout: int = None,
) -> BaseMessage:

    if timeout is None:
        timeout = settings.LLM_INVOKE_TIMEOUT

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

    # Only set retry key if retry_if_exception_types is explicitly provided.
    # Otherwise, tenacity uses its default behavior (retry on all exceptions).
    if retry_if_exception_types:
        retry_exceptions = (
            FuturesTimeoutError,
            TimeoutError,
        ) + retry_if_exception_types
        tenacity_kwargs["retry"] = retry_if_exception_type(retry_exceptions)

    retry_decorator = retry(**tenacity_kwargs)

    if chain_kwargs is None:
        chain_kwargs = {}

    @retry_decorator
    def _invoke_chain_with_retry() -> BaseMessage:
        executor = ThreadPoolExecutor(max_workers=1)
        # Copy the current context so ContextVar-based callbacks
        # (e.g. langchain cost tracking) are visible in the worker thread.
        ctx = contextvars.copy_context()
        try:
            future = executor.submit(
                ctx.run, chain.invoke, input, config, **chain_kwargs
            )
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning(
                f"Chain invocation timed out after {timeout} seconds, retrying..."
            )
            raise
        finally:
            # Use wait=False to avoid blocking on timeout.
            # Running threads can't be cancelled in Python, but we won't block on them.
            executor.shutdown(wait=False, cancel_futures=True)

    return _invoke_chain_with_retry()


def batch_invoke_chain_with_retry(
    chain: RunnableSerializable,
    input: list[Input],
    config: RunnableConfig | None = None,
    chain_kwargs: Any = None,
    max_retries: int = 3,
    reraise: bool = True,
    min_wait_seconds: int = 2,
    max_wait_seconds: int = 10,
    wait_exponential_multiplier: float = 0.75,
    retry_if_exception_types: tuple[Exception, ...] = None,
    timeout: int = None,
) -> BaseMessage:

    if timeout is None:
        timeout = settings.LLM_BATCH_INVOKE_TIMEOUT

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

    # Only set retry key if retry_if_exception_types is explicitly provided.
    # Otherwise, tenacity uses its default behavior (retry on all exceptions).
    if retry_if_exception_types:
        retry_exceptions = (
            FuturesTimeoutError,
            TimeoutError,
        ) + retry_if_exception_types
        tenacity_kwargs["retry"] = retry_if_exception_type(retry_exceptions)

    retry_decorator = retry(**tenacity_kwargs)

    if chain_kwargs is None:
        chain_kwargs = {}

    @retry_decorator
    def _batch_invoke_chain_with_retry() -> BaseMessage:
        executor = ThreadPoolExecutor(max_workers=1)
        # Copy the current context so ContextVar-based callbacks
        # (e.g. langchain cost tracking) are visible in the worker thread.
        ctx = contextvars.copy_context()
        try:
            future = executor.submit(
                ctx.run, chain.batch, input, config, **chain_kwargs
            )
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning(
                "Batch chain invocation timed out after "
                f"{timeout} seconds, retrying..."
            )
            raise
        finally:
            # Use wait=False to avoid blocking on timeout.
            # Running threads can't be cancelled in Python, but we won't block on them.
            executor.shutdown(wait=False, cancel_futures=True)

    return _batch_invoke_chain_with_retry()


def batch_invoke_llm_with_retry(
    llm: BaseChatModel,
    messages: list[list[BaseMessage]],
    config: Any = None,
    llm_kwargs: dict[str, Any] = None,
    max_retries: int = 3,
    reraise: bool = True,
    min_wait_seconds: int = 2,
    max_wait_seconds: int = 10,
    wait_exponential_multiplier: float = 0.75,
    retry_if_exception_types: tuple[Exception, ...] = None,
    timeout: int = None,
) -> BaseMessage:

    if timeout is None:
        timeout = settings.LLM_BATCH_INVOKE_TIMEOUT

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

    # Only set retry key if retry_if_exception_types is explicitly provided.
    # Otherwise, tenacity uses its default behavior (retry on all exceptions).
    if retry_if_exception_types:
        retry_exceptions = (
            FuturesTimeoutError,
            TimeoutError,
        ) + retry_if_exception_types
        tenacity_kwargs["retry"] = retry_if_exception_type(retry_exceptions)

    retry_decorator = retry(**tenacity_kwargs)

    if llm_kwargs is None:
        llm_kwargs = {}

    @retry_decorator
    def _batch_invoke_llm_with_retry() -> BaseMessage:
        executor = ThreadPoolExecutor(max_workers=1)
        # Copy the current context so ContextVar-based callbacks
        # (e.g. langchain cost tracking) are visible in the worker thread.
        ctx = contextvars.copy_context()
        try:
            future = executor.submit(ctx.run, llm.batch, messages, config, **llm_kwargs)
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning(
                "Batch LLM invocation timed out after "
                f"{timeout} seconds, retrying..."
            )
            raise
        finally:
            # Use wait=False to avoid blocking on timeout.
            # Running threads can't be cancelled in Python, but we won't block on them.
            executor.shutdown(wait=False, cancel_futures=True)

    return _batch_invoke_llm_with_retry()


@retry(**tenacity_kwargs)
def execute_gemini_agent_with_retry(
    model_name: str,
    messages: list[types.Part],
    schema: types.Schema = None,
    cache_name: str = None,
    stream: bool = False,
) -> dict[str, Any]:
    if cache_name:
        cache = retrieve_gemini_cache(
            cache_display_name=cache_name,
        )
    else:
        cache = None

    generation_config = {
        "max_output_tokens": MAX_TOKEN_MAPPING[model_name],
    }
    if cache:
        generation_config["cached_content"] = cache.name
    if schema:
        generation_config["response_schema"] = schema
        generation_config["response_mime_type"] = "application/json"

    if cache:
        logger.info("Using Gemini cache.")
    else:
        logger.info("No Gemini cache found.")

    model_kwargs = {
        "contents": messages,
        "config": types.GenerateContentConfig(
            **generation_config,
        ),
    }
    model = AutoSdkChatModel.from_model_name(model_name, **model_kwargs)
    if stream:
        response = model.invoke_through_stream()
    else:
        response = model.invoke()

    if not stream:
        logger.info(f"Usage: {response.usage_metadata.dict()}")
        if schema:
            response = json.loads(response.text)
        else:
            response = response.text
    return response


def get_message_text(message: AIMessage | AIMessageChunk) -> str:
    """Get the text from the message.

    Args:
        message (AIMessage | AIMessageChunk): The message to get the text from.

    Returns:
        str: The text from the message.
    """
    content = message.content

    if isinstance(content, str):
        return content

    text_blocks = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if text is not None:
                text_blocks.append(text)
        elif isinstance(block, str):
            text_blocks.append(block)

    return "".join(text_blocks)
