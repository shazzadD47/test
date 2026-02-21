import tiktoken
from langchain_core.messages import BaseMessage


def count_tokens_in_messages(
    messages: list[BaseMessage], model_name: str = "gpt-4"
) -> int:
    """Count tokens in a list of messages using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for non-OpenAI models
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for message in messages:
        # Handle cases where message.content might be None or not a string
        content = message.content
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        total_tokens += len(encoding.encode(content))

    return total_tokens


def count_tokens_in_text(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Handle cases where text might be None or not a string
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    return len(encoding.encode(text))


def truncate_contexts_to_fit_limit(
    contexts: list[str], max_tokens: int, model_name: str = "gpt-4"
) -> list[str]:
    """Truncate contexts to fit within token limit, prioritizing earlier contexts."""
    truncated_contexts = []
    current_tokens = 0

    for context in contexts:
        context_tokens = count_tokens_in_text(context, model_name)
        if current_tokens + context_tokens <= max_tokens:
            truncated_contexts.append(context)
            current_tokens += context_tokens
        else:
            # Try to fit a truncated version of this context
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 100:  # Only if we have meaningful space left
                truncated_context = truncate_text_to_tokens(
                    context, remaining_tokens, model_name
                )
                truncated_contexts.append(truncated_context + "... [TRUNCATED]")
            break

    return truncated_contexts


def truncate_text_to_tokens(
    text: str, max_tokens: int, model_name: str = "gpt-4"
) -> str:
    """Truncate text to fit within token limit."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Handle cases where text might be None or not a string
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
