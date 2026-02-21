from fastapi import status

from app.exceptions.http import DetailedHTTPException
from app.v3.endpoints.get_title_summery.constant import (
    DOI_EXCEPTIONS,
    ENCRYPTION_ERROR,
    FLAG_ID_EXCEPTIONS,
    FLAG_ID_NOT_FOUND,
)


class DoiExtractionFailed(Exception):
    DETAIL = DOI_EXCEPTIONS

    def __init__(self, detail: str = None):
        super().__init__(detail or self.DETAIL)


class PdfEncryptionError(Exception):
    DETAIL = ENCRYPTION_ERROR


class FlagIDException(DetailedHTTPException):
    DETAIL = FLAG_ID_EXCEPTIONS


class FlagIDNotFound(DetailedHTTPException):
    STATUS_CODE = status.HTTP_404_NOT_FOUND
    DETAIL = FLAG_ID_NOT_FOUND


class EncryptionError(DetailedHTTPException):
    STATUS_CODE = status.HTTP_400_BAD_REQUEST
    DETAIL = ENCRYPTION_ERROR
