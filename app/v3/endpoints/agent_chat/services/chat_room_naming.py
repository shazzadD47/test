from langchain_core.messages import HumanMessage, SystemMessage

from app.core.auto.chat_model import AutoChatModel
from app.utils.llms import ainvoke_llm_with_retry, get_message_text
from app.v3.endpoints.agent_chat.logging import logger
from app.v3.endpoints.agent_chat.prompts import CHAT_TITLE_GENERATION_PROMPT


def generate_chat_title(query: str, max_length: int = 50) -> str:
    """
    Generate a chat title from user query by intelligently truncating.
    Fallback method if LLM-based generation fails.

    Args:
        query: The user's query string
        max_length: Maximum character length (default: 50)

    Returns:
        A concise title for the chat history
    """
    # Remove extra whitespace and newlines
    cleaned = " ".join(query.split())

    # If short enough, return as-is
    if len(cleaned) <= max_length:
        return cleaned

    # Truncate at last complete word within max_length
    truncated = cleaned[:max_length].rsplit(" ", 1)[0]
    return f"{truncated}..."


async def generate_chat_title_with_llm(query: str) -> str:
    """
    Generate a concise chat title using an LLM.
    Falls back to simple truncation if LLM fails.

    Args:
        query: The user's query string

    Returns:
        A 3-5 word title summarizing the conversation topic
    """
    try:
        llm = AutoChatModel.from_model_name(model_name="gpt-4o-mini", temperature=0)

        messages = [
            SystemMessage(content=CHAT_TITLE_GENERATION_PROMPT),
            HumanMessage(content=query),
        ]

        response = await ainvoke_llm_with_retry(
            llm=llm,
            messages=messages,
            max_retries=3,
        )

        title = get_message_text(response).strip()
        # Remove surrounding quotes if present
        title = title.strip('"').strip("'")
        logger.info("Generated title with LLM: %s", title)
        return title

    except Exception as e:
        logger.warning(
            "LLM title generation failed: %s. Falling back to truncation.", str(e)
        )
        # Fallback to simple truncation if LLM fails
        return generate_chat_title(query, max_length=50)
