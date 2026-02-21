from fastapi import status

from app.exceptions.http import DetailedHTTPException
from app.v3.endpoints.column_standardization.constants import (
    COLUMN_STANDARDIZATION_TASK_FAILED,
    TABLE_ID_REQUIRED,
)


class TableIdRequiredException(DetailedHTTPException):
    STATUS_CODE = status.HTTP_400_BAD_REQUEST
    DETAIL = TABLE_ID_REQUIRED


class ColumnStandardizationTaskFailedException(DetailedHTTPException):
    STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR
    DETAIL = COLUMN_STANDARDIZATION_TASK_FAILED


class BackendAPIError(Exception):
    """Backend API related errors."""

    pass


class ProcessingError(Exception):
    """Processing related errors."""

    pass
