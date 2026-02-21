from app.exceptions.http import DetailedHTTPException
from app.v3.endpoints.dosing_table.constants import FAILED_TO_PARSE_DOSING_TABLE


class DosingTableParsingException(DetailedHTTPException):
    DETAIL = FAILED_TO_PARSE_DOSING_TABLE
