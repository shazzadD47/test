from app.exceptions.http import DetailedHTTPException, status
from app.v3.endpoints.covariate_extraction.constants import (
    FAILED_TO_PARSE_COVARIATE_TABLE,
    FAILED_TO_PARSE_IMAGE_TYPE,
    ErrorCode,
)


class ClaudeImageProcessingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.CLAUDE_IMAGE_PROCESSING_FAILED


class OpenAiImageProcessingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.OPENAI_IMAGE_PROCESSING_FAILED


class TableTypeParsingException(DetailedHTTPException):
    DETAIL = FAILED_TO_PARSE_IMAGE_TYPE


class TableExtractionException(DetailedHTTPException):
    DETAIL = FAILED_TO_PARSE_COVARIATE_TABLE


class AnthropicServerFailed(DetailedHTTPException):
    STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR
    DETAIL = "Server error at Anthropic's end"


class QuestionRephrasingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.QUESTION_REPHRASING_FAILED


class ContextSummarizationFailed(DetailedHTTPException):
    DETAIL = ErrorCode.CONTEXT_SUMMARIZATION_FAILED
