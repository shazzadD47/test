from functools import wraps

from langchain_community.callbacks.manager import get_openai_callback

from app.core.callbacks.manager import get_anthropic_callback
from app.logging import logger


def _format_openai_usage(result, callback):
    if callback.total_tokens == 0:
        return result

    if isinstance(result, dict):
        if "usage" not in result:
            result["usage"] = {}

        result["usage"]["openai"] = {
            "total_tokens": callback.total_tokens,
            "prompt_tokens": callback.prompt_tokens,
            "completion_tokens": callback.completion_tokens,
            "cost": callback.total_cost,
        }
    else:
        logger.warning(f"OpenAI usage info not added to response type {type(result)}")

    return result


def _format_anthropic_usage(result, callback):
    if callback.total_tokens == 0:
        return result

    if isinstance(result, dict):
        if "usage" not in result:
            result["usage"] = {}

        result["usage"]["anthropic"] = {
            "total_tokens": callback.total_tokens,
            "prompt_tokens": callback.prompt_tokens,
            "completion_tokens": callback.completion_tokens,
            "cost": callback.total_cost,
        }
    else:
        logger.warning(
            f"Anthropic usage info not added to response type {type(result)}"
        )

    return result


def track_openai_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with get_openai_callback() as callback:
            result = func(*args, **kwargs)

            return _format_openai_usage(result, callback)

    return wrapper


def async_track_openai_usage(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        with get_openai_callback() as callback:
            result = await func(*args, **kwargs)

            return _format_openai_usage(result, callback)

    return wrapper


def async_track_anthropic_usage(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        with get_anthropic_callback() as callback:
            result = await func(*args, **kwargs)

            return _format_anthropic_usage(result, callback)

    return wrapper
