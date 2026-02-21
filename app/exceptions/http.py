from typing import Any

import sentry_sdk
from fastapi import HTTPException, status


class DetailedHTTPException(HTTPException):
    STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR
    DETAIL = "Internal Server Error"
    SENTRY_FINGERPRINT = None
    SENTRY_LEVEL = "error"

    def __init__(
        self, status_code=None, detail=None, headers=None, **kwargs: dict[str, Any]
    ) -> None:
        if status_code:
            self.STATUS_CODE = status_code

        if detail:
            self.DETAIL = detail

        self._capture_sentry_context(kwargs)

        super().__init__(
            status_code=self.STATUS_CODE, detail=self.DETAIL, headers=headers
        )

    def _capture_sentry_context(self, extra_data=None):
        """Add context to Sentry for better error tracking"""
        if not sentry_sdk.get_client():
            return

        sentry_sdk.set_context(
            "error_details",
            {
                "status_code": self.STATUS_CODE,
                "detail": self.DETAIL,
                **(extra_data or {}),
            },
        )

        sentry_sdk.set_level(self.SENTRY_LEVEL)

        if self.SENTRY_FINGERPRINT:
            sentry_sdk.set_tag("fingerprint", self.SENTRY_FINGERPRINT)


class AnthropicServerFailed(DetailedHTTPException):
    STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR
    DETAIL = "Server error at Anthropic's end"
    SENTRY_FINGERPRINT = "anthropic-server-error"
    SENTRY_LEVEL = "error"


class OpenAIServerFailed(DetailedHTTPException):
    STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR
    DETAIL = "Server error at OpenAI's end"
    SENTRY_FINGERPRINT = "openai-server-error"
    SENTRY_LEVEL = "error"
