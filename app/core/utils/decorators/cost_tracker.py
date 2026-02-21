import asyncio
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from langchain_core.callbacks import get_usage_metadata_callback

from app.configs import settings
from app.core.callbacks.cost_handler import (
    RAGCostCallback,
    SdkUsageMetadataCallbackHandler,
)
from app.core.usage.calculate_cost import (
    calculate_cost,
    calculate_total_cost,
)
from app.core.usage.helpers import (
    convert_to_custom_usage_metadata,
    format_model_names,
)
from app.core.utils.decorators.helpers import (
    combine_with_result_cost_metadata,
    return_result_with_cost_metadata,
)
from app.core.utils.hooks import hook_manager
from app.logging import logger

# ---------------------------------------------------------------------------
# ContextVar-based SDK cost handler (concurrency-safe)
# ---------------------------------------------------------------------------
# A single global ContextVar holds the SdkUsageMetadataCallbackHandler for the
# current execution context (request / asyncio task).  A single global hook is
# registered once at module load; it looks up the handler from the ContextVar
# so that concurrent requests each track their own costs independently.
# ---------------------------------------------------------------------------
_sdk_handler_var: ContextVar[SdkUsageMetadataCallbackHandler | None] = ContextVar(
    "_sdk_handler_var", default=None
)


def _contextvar_llm_sdk_cost_hook(response, model_name: str):
    """Global hook that dispatches to the ContextVar-scoped SDK handler."""
    handler = _sdk_handler_var.get(None)
    if handler is not None:
        handler.on_llm_end(response, model_name)


# Register the hook ONCE at module import — no more per-request register/unregister.
hook_manager.register("on_llm_end", _contextvar_llm_sdk_cost_hook)


@contextmanager
def get_rag_cost_callback(
    name: str = "rag_cost_callback",
) -> Generator[RAGCostCallback, None, None]:
    from langchain_core.tracers.context import register_configure_hook

    rag_cost_callback_var: ContextVar[RAGCostCallback | None] = ContextVar(
        name, default=None
    )
    register_configure_hook(rag_cost_callback_var, inheritable=True)
    cb = RAGCostCallback(model_name=settings.OPENAI_EMBEDDING_MODEL)
    rag_cost_callback_var.set(cb)
    yield cb
    rag_cost_callback_var.set(None)


def _process_cost(result, usage_metadata, label: str):
    """Shared helper for computing and attaching cost metadata to a result.

    Parameters
    ----------
    result : Any
        The return value from the decorated function.
    usage_metadata : dict
        Pre-computed usage metadata (already converted / formatted by the caller).
    label : str
        Human-readable label used in log messages (e.g. "RAG", "Langchain", "LLM SDK").
    """
    try:
        logger.info(f"Calculating {label} Cost")
        own_cost_metadata = calculate_cost(usage_metadata)

        # Log only THIS tracker's own cost (before combining with inner costs)
        own_total_cost = calculate_total_cost(own_cost_metadata)
        own_cost_dump = {k: v.model_dump() for k, v in own_cost_metadata.items()}
        logger.info(
            f"{label} Cost: {{'total_cost': {own_total_cost}, "
            f"'llm_cost_details': {own_cost_dump}}}\n"
        )

        # Combine with any cost metadata from inner decorators for the result
        combined_cost_metadata = combine_with_result_cost_metadata(
            result, own_cost_metadata
        )
        combined_total_cost = calculate_total_cost(combined_cost_metadata)
        combined_cost_dump = {
            k: v.model_dump() for k, v in combined_cost_metadata.items()
        }
        cost_metadata_with_total_cost = {
            "total_cost": combined_total_cost,
            "llm_cost_details": combined_cost_dump,
        }
        return return_result_with_cost_metadata(
            result,
            cost_metadata_with_total_cost,
        )
    except Exception as e:
        logger.error(f"Error calculating {label} Cost: {e}")
        logger.info(f"Traceback: {traceback.format_exc()}")
        return {
            "result": result,
            "metadata": {
                "ai_metadata": {
                    "cost_metadata": {},
                }
            },
        }


def _process_rag_cost(result, rag_cb):
    """Compute RAG cost from callback and attach to result."""
    try:
        usage_metadata = convert_to_custom_usage_metadata(rag_cb.usage_metadata)
    except Exception as e:
        logger.error(f"Error converting RAG usage metadata: {e}")
        logger.info(f"Traceback: {traceback.format_exc()}")
        return {
            "result": result,
            "metadata": {
                "ai_metadata": {
                    "cost_metadata": {},
                }
            },
        }
    return _process_cost(result, usage_metadata, "RAG")


#  define a decorator to track the cost of the embedding
def track_rag_cost(func):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with get_rag_cost_callback() as rag_cb:
                result = await func(*args, **kwargs)
                return _process_rag_cost(result, rag_cb)

        return async_wrapper
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_rag_cost_callback() as rag_cb:
                result = func(*args, **kwargs)
                return _process_rag_cost(result, rag_cb)

        return wrapper


def _process_langchain_cost(result, langchain_cb):
    """Compute Langchain cost from callback and attach to result."""
    try:
        usage_metadata = convert_to_custom_usage_metadata(langchain_cb.usage_metadata)
        usage_metadata = format_model_names(usage_metadata)
    except Exception as e:
        logger.error(f"Error converting Langchain usage metadata: {e}")
        logger.info(f"Traceback: {traceback.format_exc()}")
        return {
            "result": result,
            "metadata": {
                "ai_metadata": {
                    "cost_metadata": {},
                }
            },
        }
    return _process_cost(result, usage_metadata, "Langchain")


def track_langchain_cost(func):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with get_usage_metadata_callback() as langchain_cb:
                result = await func(*args, **kwargs)
                return _process_langchain_cost(result, langchain_cb)

        return async_wrapper
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_usage_metadata_callback() as langchain_cb:
                result = func(*args, **kwargs)
                return _process_langchain_cost(result, langchain_cb)

        return wrapper


@contextmanager
def get_llm_sdk_cost_callback(
    name: str = "llm_sdk_cost_callback",
) -> Generator[SdkUsageMetadataCallbackHandler, None, None]:
    cb = SdkUsageMetadataCallbackHandler()
    # Use the module-level ContextVar so the global hook dispatches to *this*
    # handler for the current execution context only (concurrency-safe).
    token = _sdk_handler_var.set(cb)
    try:
        yield cb
    finally:
        _sdk_handler_var.reset(token)


def _process_llm_sdk_cost(result, sdk_cb):
    """Compute LLM SDK cost from callback and attach to result."""
    usage_metadata = sdk_cb.usage_metadata
    return _process_cost(result, usage_metadata, "LLM SDK")


def track_llm_sdk_cost(func):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with get_llm_sdk_cost_callback() as sdk_cb:
                result = await func(*args, **kwargs)
                return _process_llm_sdk_cost(result, sdk_cb)

        return async_wrapper
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_llm_sdk_cost_callback() as sdk_cb:
                result = func(*args, **kwargs)
                return _process_llm_sdk_cost(result, sdk_cb)

        return wrapper


def track_all_llm_costs(func):
    # Each inner decorator auto-detects async vs sync and applies @wraps,
    # so the composed result already has correct metadata and type.
    return track_llm_sdk_cost(track_langchain_cost(track_rag_cost(func)))
