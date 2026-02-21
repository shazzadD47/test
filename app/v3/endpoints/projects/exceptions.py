from fastapi import status

from app.exceptions.http import DetailedHTTPException


class StorageError(DetailedHTTPException):
    DETAIL = "An error occurred while processing the storage operation."


class DatabaseError(DetailedHTTPException):
    DETAIL = "An error occurred while processing the database operation."


class UnexpectedError(DetailedHTTPException):
    DETAIL = "An unexpected error occurred."


class DataFetchFailed(DetailedHTTPException):
    DETAIL = "Failed to retrieve paper_summary values from the database."


class NoSummariesFound(DetailedHTTPException):
    STATUS_CODE = status.HTTP_404_NOT_FOUND
    DETAIL = "No summaries found for this project."
