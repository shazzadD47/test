from app.exceptions.http import DetailedHTTPException
from app.v3.endpoints.plot_digitizer.constants import ErrorCode


class ClaudeImageProcessingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.CLAUDE_IMAGE_PROCESSING_FAILED


class ClaudeResponseParsingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.CLAUDE_RESPONSE_PARSING_FAILED


class OpenAiImageProcessingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.OPENAI_IMAGE_PROCESSING_FAILED


class OpenAIResponseParsingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.OPENAI_RESPONSE_PARSING_FAILED


class QuestionRephrasingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.QUESTION_REPHRASING_FAILED


class ContextSummarizationFailed(DetailedHTTPException):
    DETAIL = ErrorCode.CONTEXT_SUMMARIZATION_FAILED


class LineFormerFailed(DetailedHTTPException):
    DETAIL = ErrorCode.LINE_FORMER_FAILED


class ContextQAFailed(DetailedHTTPException):
    DETAIL = ErrorCode.CONTEXT_QA_FAILED
