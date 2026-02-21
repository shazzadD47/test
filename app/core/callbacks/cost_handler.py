"""Callback Handler that tracks AIMessage.usage_metadata."""

import threading
from collections import defaultdict
from typing import Any

import tiktoken
from anthropic.types.usage import Usage
from google.genai.types import ModalityTokenCount
from google.genai.types import UsageMetadata as GoogleUsageMetadata
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.outputs import LLMResult
from openai.types.responses.response_usage import ResponseUsage
from typing_extensions import override

from app.configs import settings
from app.core.auto.chat_model import (
    ANTHROPIC_MODEL_MAPPING,
    GOOGLE_GENAI_MODEL_MAPPING,
    MODEL_MAPPING,
    OPENAI_MODEL_MAPPING,
)
from app.core.usage.helpers import combine_usage_metadatas
from app.core.usage.models import CustomUsageMetadata


class RAGCostCallback(BaseCallbackHandler):
    def __init__(self, model_name: str = settings.OPENAI_EMBEDDING_MODEL):
        """Initialize the UsageMetadataCallbackHandler."""
        super().__init__()
        self._lock = threading.Lock()
        self.usage_metadata: dict[str, UsageMetadata] = {}
        self.model_name = model_name

    @override
    def __repr__(self) -> str:
        return str(f"RAG Cost for model {self.model_name} is {self.usage_metadata}")

    @override
    def on_retriever_start(
        self, serialized: dict[str, Any], query: str, **kwargs: Any
    ) -> None:
        total_tokens = len(tiktoken.encoding_for_model(self.model_name).encode(query))
        self.total_tokens = total_tokens

    @override
    def on_retriever_end(self, documents, **kwargs: Any):
        usage_metadata = UsageMetadata(
            input_tokens=self.total_tokens,
            output_tokens=0,
            total_tokens=self.total_tokens,
        )
        with self._lock:
            if self.model_name not in self.usage_metadata:
                self.usage_metadata[self.model_name] = usage_metadata
            else:
                self.usage_metadata[self.model_name] = add_usage(
                    self.usage_metadata[self.model_name],
                    usage_metadata,
                )


def _standardize_google_token_usage_details(
    usage_details: list[ModalityTokenCount], total_tokens: int
) -> defaultdict[str, int]:
    """
    Standardize the token usage details for Google. Converts google's
    own modality token count to a dictionary with modalities as keys.

    Args:
        usage_details (list[ModalityTokenCount]): The token usage details.
        total_tokens (int): The total tokens.

    Returns:
        defaultdict[str, int]: The standardized token usage details.
    """
    if not usage_details:
        return defaultdict(
            int,
            {
                "text": total_tokens,
            },
        )
    token_details = defaultdict(int)
    for each in usage_details:
        if str(each.modality) == "MediaModality.TEXT":
            token_details["text"] += each.token_count
        elif str(each.modality) == "MediaModality.IMAGE":
            token_details["image"] += each.token_count
        elif str(each.modality) == "MediaModality.AUDIO":
            token_details["audio"] += each.token_count
        elif str(each.modality) == "MediaModality.VIDEO":
            token_details["video"] += each.token_count
        elif str(each.modality) == "MediaModality.DOCUMENT":
            token_details["document"] += each.token_count
        elif str(each.modality) == "MediaModality.MODALITY_UNSPECIFIED":
            token_details["unspecified"] += each.token_count
    return token_details


def standardize_google_usage(usage: GoogleUsageMetadata) -> CustomUsageMetadata:
    """
    Standardize Google usage metadata to a CustomUsageMetadata format.

    Converts Google's GenerateContentResponseUsageMetadata to CustomUsageMetadata
    format, handling various token types including input, output, cached content,
    tool use, and reasoning tokens. Also processes modality-specific token details
    for different media types (text, image, audio, video, document).

    Args:
        usage (GoogleUsageMetadata): The Google usage metadata to standardize.

    Returns:
        CustomUsageMetadata: The standardized usage metadata with token counts
        and details organized by type and modality.
    """
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    cache_read_tokens = 0
    tool_use_prompt_tokens = 0
    total_tokens = 0
    cache_read_token_details = defaultdict(int)
    input_token_details = defaultdict(int)
    output_token_details = defaultdict(int)
    reasoning_token_details = defaultdict(int)

    if usage.prompt_token_count:
        input_tokens = usage.prompt_token_count

    if usage.cached_content_token_count:
        cache_read_tokens = usage.cached_content_token_count
        input_tokens -= cache_read_tokens

    if usage.tool_use_prompt_token_count:
        tool_use_prompt_tokens = usage.tool_use_prompt_token_count
        input_tokens += tool_use_prompt_tokens

    input_token_details = _standardize_google_token_usage_details(
        usage.prompt_tokens_details, input_tokens
    )
    if cache_read_tokens > 0:
        cache_read_token_details = _standardize_google_token_usage_details(
            usage.cache_tokens_details, cache_read_tokens
        )
        for key, value in cache_read_token_details.items():
            if key in input_token_details:
                input_token_details[key] -= value

    if tool_use_prompt_tokens > 0:
        tool_token_details = _standardize_google_token_usage_details(
            usage.tool_use_prompt_tokens_details, tool_use_prompt_tokens
        )
        for key, value in tool_token_details.items():
            if key in input_token_details:
                input_token_details[key] += value

    if usage.candidates_token_count:
        output_tokens = usage.candidates_token_count

    if usage.candidates_tokens_details:
        output_token_details = _standardize_google_token_usage_details(
            usage.candidates_tokens_details, output_tokens
        )

    if usage.thoughts_token_count:
        reasoning_tokens = usage.thoughts_token_count

    reasoning_token_details = {
        "text": reasoning_tokens,
    }

    if usage.total_token_count:
        total_tokens = usage.total_token_count
    else:
        total_tokens = (
            input_tokens + cache_read_tokens + output_tokens + reasoning_tokens
        )

    return CustomUsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_read_token_details=cache_read_token_details,
        input_token_details=input_token_details,
        output_token_details=output_token_details,
        reasoning_token_details=reasoning_token_details,
        total_tokens=total_tokens,
    )


def standardize_anthropic_usage(usage: Usage) -> CustomUsageMetadata:
    """
    Standardize Anthropic usage metadata to a CustomUsageMetadata format.

    Converts Anthropic's Usageobject to CustomUsageMetadata format, handling
    various token types including input, output, cached content, and
    reasoning tokens. Also processes cache-related token details for different
    cache types (ephemeral, persistent).

    Args:
        usage (Usage): The Anthropic usage metadata to standardize.

    Returns:
        CustomUsageMetadata: The standardized usage metadata with token counts
        and details organized by type and modality.
    """
    input_tokens = 0
    output_tokens = 0
    cache_write_tokens = 0
    cache_read_tokens = 0
    total_tokens = 0
    input_token_details = defaultdict(int)
    output_token_details = defaultdict(int)
    cache_read_token_details = defaultdict(int)
    cache_write_token_details = defaultdict(int)

    if usage.input_tokens:
        input_tokens = usage.input_tokens
    if usage.output_tokens:
        output_tokens = usage.output_tokens
    if usage.cache_read_input_tokens:
        cache_read_tokens = usage.cache_read_input_tokens
    if usage.cache_creation_input_tokens:
        cache_write_tokens = usage.cache_creation_input_tokens

    input_token_details["text"] = input_tokens
    output_token_details["text"] = output_tokens
    cache_read_token_details["text"] = cache_read_tokens

    if hasattr(usage, "cache_creation") and usage.cache_creation:
        cache_write_token_details["cache_5m"] = (
            usage.cache_creation.ephemeral_5m_input_tokens
        )
        cache_write_token_details["cache_1h"] = (
            usage.cache_creation.ephemeral_1h_input_tokens
        )
    else:
        cache_write_token_details["text"] = cache_write_tokens

    total_tokens = input_tokens + cache_write_tokens + cache_read_tokens + output_tokens
    return CustomUsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        input_token_details=input_token_details,
        output_token_details=output_token_details,
        cache_write_token_details=cache_write_token_details,
        cache_read_token_details=cache_read_token_details,
    )


def standardize_openai_usage(usage: ResponseUsage) -> CustomUsageMetadata:
    """
    Standardize OpenAI usage metadata to a CustomUsageMetadata format.

    Converts OpenAI's ResponseUsage to CustomUsageMetadata format, handling
    various token types including input, output, cached content, and reasoning tokens.

    Args:
        usage (ResponseUsage): The OpenAI usage metadata to standardize.

    Returns:
        CustomUsageMetadata: The standardized usage metadata with token counts
        and details organized by type and modality.
    """
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    total_tokens = 0
    input_token_details = defaultdict(int)
    output_token_details = defaultdict(int)
    cache_read_token_details = defaultdict(int)

    if usage.input_tokens:
        input_tokens = usage.input_tokens
    if usage.input_tokens_details:
        cache_read_tokens = usage.input_tokens_details.cached_tokens
        input_tokens -= cache_read_tokens
    if usage.output_tokens:
        output_tokens = usage.output_tokens
    if usage.total_tokens:
        total_tokens = usage.total_tokens
    else:
        total_tokens = input_tokens + output_tokens

    input_token_details["text"] = input_tokens
    cache_read_token_details["text"] = cache_read_tokens
    output_token_details["text"] = output_tokens

    return CustomUsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read_tokens,
        input_token_details=input_token_details,
        output_token_details=output_token_details,
        cache_read_token_details=cache_read_token_details,
    )


class SdkUsageMetadataCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        """Initialize the SdkCostTracker."""
        super().__init__()
        self._lock = threading.Lock()
        self.usage_metadata: dict[str, CustomUsageMetadata] = {}

    @override
    def __repr__(self) -> str:
        return str(self.usage_metadata)

    @override
    def on_llm_end(self, response: LLMResult, model_name: str, **kwargs: Any) -> None:
        """Collect token usage from the response."""
        if model_name not in MODEL_MAPPING:
            raise ValueError(f"Model {model_name} is not supported.")

        if model_name in OPENAI_MODEL_MAPPING:
            usage_metadata = standardize_openai_usage(response.usage)

        elif model_name in ANTHROPIC_MODEL_MAPPING:
            usage_metadata = standardize_anthropic_usage(response.usage)

        elif model_name in GOOGLE_GENAI_MODEL_MAPPING:
            usage_metadata = standardize_google_usage(response.usage_metadata)

        # update shared state behind lock to avoid race conditions
        if usage_metadata and model_name:
            with self._lock:
                if model_name not in self.usage_metadata:
                    self.usage_metadata[model_name] = usage_metadata
                else:
                    self.usage_metadata[model_name] = combine_usage_metadatas(
                        [self.usage_metadata[model_name], usage_metadata]
                    )
