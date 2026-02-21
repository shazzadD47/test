from datetime import datetime
from typing import Any

import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from app.auth.bearer.jwt import decode_jwt
from app.configs import settings


class AuthenticatedUser(BaseModel):
    """Schema for authenticated user."""

    model_config = ConfigDict(extra="allow")

    token: str
    sub: str | None = None
    raw_claims: dict[str, Any] = Field(default_factory=dict)


class UserAuthClient(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        self.client_id = None
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> AuthenticatedUser:
        """Call the client to authenticate the request."""
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)

        if credentials:
            if credentials.scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme.",
                )

            user_info = self.from_token(credentials.credentials)
            if not user_info:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token or expired token.",
                )

            return user_info
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization code.",
            )

    def from_token(self, token: str) -> AuthenticatedUser:
        """Decodes a JWT token and extracts client details.

        :param token: The JWT token to decode.
        :return: A dictionary containing user information.
        :raises HTTPException: If the token is invalid or expired.
        """
        try:
            decoded_token = decode_jwt(token, settings.JWT_USER_AUTH_PUBLIC_KEY)
            exp = decoded_token.get("exp")

            if exp < int(datetime.now().timestamp()):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired. Please login again.",
                )

            return AuthenticatedUser(
                token=token,
                sub=decoded_token.get("sub"),
                raw_claims=decoded_token,
            )
        except jwt.ExpiredSignatureError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired. Please login again.",
            ) from e
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
            ) from e
