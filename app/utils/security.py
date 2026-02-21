import hashlib
import time

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.configs import settings

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def generate_signature(salt: str, secret: str = settings.API_SECRET_KEY):
    return hashlib.sha256(f"{salt}:{secret}".encode()).hexdigest()


def verify_signature(salt: str, signature: str, secret: str = settings.API_SECRET_KEY):
    expected_signature = generate_signature(salt, secret)
    return signature == expected_signature


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing API Key",
        )

    try:
        salt, signature = api_key.split(":")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key format"
        )

    if not verify_signature(salt, signature):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )

    current_time = int(time.time())
    if abs(current_time - int(salt)) > settings.API_TIME_OUT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Request expired"
        )

    return True
