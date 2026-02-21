from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from app.core.auto.factory import BaseAutoCallbackHandler
from app.core.callbacks.anthropic import AnthropicTokenUsageCallbackHandler

OPENAI_MODEL_MAPPING = {
    "gpt-4.1": OpenAICallbackHandler,
    "gpt-4.1-mini": OpenAICallbackHandler,
    "gpt-4o": OpenAICallbackHandler,
    "gpt-4o-2024-08-06": OpenAICallbackHandler,
    "chatgpt-4o-latest": OpenAICallbackHandler,
    "gpt-4o-mini": OpenAICallbackHandler,
    "gpt-4-turbo": OpenAICallbackHandler,
    "gpt-4": OpenAICallbackHandler,
    "gpt-3.5-turbo": OpenAICallbackHandler,
    "gpt-3.5-turbo-1106": OpenAICallbackHandler,
    "o3-mini": OpenAICallbackHandler,
    "o4-mini": OpenAICallbackHandler,
    "o3": OpenAICallbackHandler,
}

ANTHROPIC_MODEL_MAPPING = {
    "claude-sonnet-4-20250514": AnthropicTokenUsageCallbackHandler,
    "claude-3-7-sonnet-latest": AnthropicTokenUsageCallbackHandler,
    "claude-3-7-sonnet-20250219": AnthropicTokenUsageCallbackHandler,
    "claude-3-5-sonnet-20240620": AnthropicTokenUsageCallbackHandler,
    "claude-3-opus-20240229": AnthropicTokenUsageCallbackHandler,
    "claude-3-sonnet-20240229": AnthropicTokenUsageCallbackHandler,
    "claude-3-haiku-20240307": AnthropicTokenUsageCallbackHandler,
    "claude-sonnet-4-0": AnthropicTokenUsageCallbackHandler,
    "claude-opus-4-0": AnthropicTokenUsageCallbackHandler,
}


class AutoCallbackHandler(BaseAutoCallbackHandler):
    _model_mapping = {**OPENAI_MODEL_MAPPING, **ANTHROPIC_MODEL_MAPPING}
