class RAGException(Exception):
    """Base exception for RAG-related errors."""

    DETAIL = "An error occurred during RAG operations."

    def __init__(self, detail: str = None):
        super().__init__(detail or self.DETAIL)


class ContextRetrievalError(RAGException):
    """Raised when there's an error retrieving context."""

    DETAIL = "An error occurred retrieving context."

    def __init__(self, detail: str = None):
        super().__init__(detail or self.DETAIL)


class ResponseGenerationError(RAGException):
    """Raised when there's an error generating a response."""

    DETAIL = "An error occurred generating a response."

    def __init__(self, detail: str = None):
        super().__init__(detail or self.DETAIL)
