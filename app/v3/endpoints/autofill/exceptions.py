from fastapi import status

from app.exceptions.http import DetailedHTTPException
from app.v3.endpoints.autofill.constants import (
    NO_ROOT_CHOICES_PROVIDED,
    ROOT_CHOICES_NOT_ALLOWED,
)


class RootChoiceNotAllowedException(DetailedHTTPException):
    STATUS_CODE = status.HTTP_400_BAD_REQUEST
    DETAIL = ROOT_CHOICES_NOT_ALLOWED


class NoRootChoiceProvided(DetailedHTTPException):
    STATUS_CODE = status.HTTP_400_BAD_REQUEST
    DETAIL = NO_ROOT_CHOICES_PROVIDED
