from app.exceptions.http import DetailedHTTPException
from app.v3.endpoints.get_paper_labels.constants import ErrorCode


class ClaudeResponseParsingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.CLAUDE_RESPONSE_PARSING_FAILED


class OpenAIResponseParsingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.OPENAI_RESPONSE_PARSING_FAILED


class QuestionRephrasingFailed(DetailedHTTPException):
    DETAIL = ErrorCode.QUESTION_REPHRASING_FAILED


class ContextSummarizationFailed(DetailedHTTPException):
    DETAIL = ErrorCode.CONTEXT_SUMMARIZATION_FAILED


class ContextQAFailed(DetailedHTTPException):
    DETAIL = ErrorCode.CONTEXT_QA_FAILED
