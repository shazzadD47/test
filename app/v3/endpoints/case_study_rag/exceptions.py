class CaseStudyRAGException(Exception):
    """Base exception for Case Study RAG operations."""

    DETAIL = "An error occurred during Case Study RAG operations."

    def __init__(self, detail: str = None):
        super().__init__(detail or self.DETAIL)


class DatabaseConnectionError(CaseStudyRAGException):
    """Raised when there's an error connecting to the database."""

    DETAIL = "An error occurred connecting to the database."

    def __init__(self, detail: str = None):
        super().__init__(detail or self.DETAIL)


class DataFetchError(CaseStudyRAGException):
    """Raised when there's an error fetching data."""

    DETAIL = "An error occurred fetching data."

    def __init__(self, detail: str = None):
        super().__init__(detail or self.DETAIL)
