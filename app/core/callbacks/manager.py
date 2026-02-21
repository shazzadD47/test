from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.tracers.context import register_configure_hook

from app.core.callbacks.anthropic import AnthropicTokenUsageCallbackHandler

anthropic_callback_var: (ContextVar)[AnthropicTokenUsageCallbackHandler | None] = (
    ContextVar("anthropic_callback", default=None)
)

register_configure_hook(anthropic_callback_var, True)


@contextmanager
def get_anthropic_callback() -> (
    Generator[AnthropicTokenUsageCallbackHandler, None, None]
):
    """Get the anthropic callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        AnthropicTokenUsageCallbackHandler:
            The  anthropic callback handler.

    Example:
        >>> with get_anthropic_callback() as cb:
        ...     # Use the anthropic callback handler
    """
    cb = AnthropicTokenUsageCallbackHandler()
    anthropic_callback_var.set(cb)
    yield cb
    anthropic_callback_var.set(None)
