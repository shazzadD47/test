import base64
from datetime import datetime

import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.auth.bearer.jwt import decode_jwt
from app.configs import settings


class S2SSecurityModel(BaseModel):
    client_id: str
    scopes: list[str]


class S2SClient(HTTPBearer):
    """
    A class to represent a client authenticated via token.
    """

    def __init__(self, required_scope: list[str], auto_error: bool = True):
        self.required_scope = required_scope
        self.client_id = None
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> S2SSecurityModel:
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

    def from_token(self, token: str) -> S2SSecurityModel:
        """
        Decodes a JWT token and extracts client details.

        :param token: The JWT token to decode.
        :return: A dictionary containing user information.
        :raises HTTPException: If the token is invalid or expired.
        """
        try:
            public_key = settings.JWT_S2S_PUBLIC_KEY
            public_key = base64.b64decode(public_key.strip()).decode("utf-8")
            decoded_token = decode_jwt(token, public_key)

            client_id = decoded_token.get("clientId")
            scopes = decoded_token.get("scopes", [])
            exp = decoded_token.get("exp")

            if not client_id or not scopes:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing required claims.",
                )

            if exp < int(datetime.now().timestamp()):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired. Please login again.",
                )

            # Check if the required scopes are present in the token
            if not any(scope in scopes for scope in self.required_scope):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"""Insufficient scopes. Required: {self.required_scope},
                    Provided: {scopes}""",
                )

            return S2SSecurityModel(client_id=client_id, scopes=scopes)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired. Please login again.",
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
            )
