from app.exceptions.http import DetailedHTTPException
from app.v3.endpoints.general_extraction.constants import (
    FAILED_TO_PARSE_GENERAL_EXTRACTION,
)


class GeneralExtractionParsingException(DetailedHTTPException):
    DETAIL = FAILED_TO_PARSE_GENERAL_EXTRACTION
