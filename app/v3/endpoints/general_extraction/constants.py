from dataclasses import dataclass

from app.core.auto.chat_model import AutoChatModel
from app.v3.endpoints.general_extraction.configs import settings as ge_settings


@dataclass(frozen=True)
class ErrorCode:
    CLAUDE_RESPONSE_PARSING_FAILED = "Failed to parse response from Claude."
    OPENAI_RESPONSE_PARSING_FAILED = "Failed to parse response from OpenAI."
    GEMINI_RESPONSE_PARSING_FAILED = "Failed to parse response from Gemini."

    QUESTION_REPHRASING_FAILED = "Failed to rephrase question."
    CONTEXT_SUMMARIZATION_FAILED = "Failed to summarize context."
    GENERAL_EXTRACTION_ERROR_MESSAGE = "General extraction error: "


@dataclass(frozen=True)
class MaxAllowedFields:
    OPENAI = 100
    GEMINI = 100
    CLAUDE = -1
    OTHER = -1


def initialize_llm(model_name: str):
    return AutoChatModel.from_model_name(
        model_name=model_name,
    )


class _LazyLLM:
    """Defers LLM creation until first attribute access.
    Avoids allocating heavy model clients at import time in forked workers."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._instance = None

    def _ensure(self):
        if self._instance is None:
            self._instance = initialize_llm(self._model_name)
        return self._instance

    def __getattr__(self, name):
        return getattr(self._ensure(), name)

    def with_structured_output(self, *args, **kwargs):
        return self._ensure().with_structured_output(*args, **kwargs)


analyzer_llm = _LazyLLM(ge_settings.ANALYZER_LLM)
query_generator_llm = _LazyLLM(ge_settings.QUERY_GENERATOR_LLM)

run_name = "general_extraction"
