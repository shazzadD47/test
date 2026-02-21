import threading

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.auto import AutoChatModel
from app.v3.endpoints.agent_chat.configs import settings as agent_chat_settings
from app.v3.endpoints.agent_chat.prompts import IMAGE_REASONING_SYSTEM_PROMPT

# Lazy-initialized to speed up app startup
_image_reasoning_llm = None
_image_reasoning_llm_lock = threading.Lock()


def get_image_reasoning_llm():
    """Get or create the image reasoning LLM instance (thread-safe)."""
    global _image_reasoning_llm
    if _image_reasoning_llm is None:
        with _image_reasoning_llm_lock:
            # Double-check pattern to avoid race conditions
            if _image_reasoning_llm is None:
                _image_reasoning_llm = AutoChatModel.from_model_name(
                    model_name=agent_chat_settings.REASONING_LLM,
                )
    return _image_reasoning_llm


async def image_reasoning_agent(
    description: str,
):
    """
    This agent is used to reason about the image and the description.

    Args:
        description: The description of the image.

    Returns:
        The reasoning about the image and the description.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", IMAGE_REASONING_SYSTEM_PROMPT),
            ("user", "Describe the image where user query is {description}."),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    prompt = prompt.partial(description=description)

    chain = prompt | get_image_reasoning_llm()
    return chain
