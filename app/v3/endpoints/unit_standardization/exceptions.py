from fastapi import status

from app.exceptions.http import DetailedHTTPException
from app.v3.endpoints.unit_standardization.constants import (
    TABLE_ID_REQUIRED,
    UNIT_STANDARDIZATION_TASK_FAILED,
)


class TableIdRequiredException(DetailedHTTPException):
    STATUS_CODE = status.HTTP_400_BAD_REQUEST
    DETAIL = TABLE_ID_REQUIRED


class UnitStandardizationTaskFailedException(DetailedHTTPException):
    STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR
    DETAIL = UNIT_STANDARDIZATION_TASK_FAILED


class BackendAPIError(Exception):
    """Backend API related errors."""

    pass


class ProcessingError(Exception):
    """Processing related errors."""

    pass
