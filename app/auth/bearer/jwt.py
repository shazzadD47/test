import time

import jwt

JWT_ALGORITHM = "ES512"


def decode_jwt(token: str, public_key: str) -> dict:
    """
    Decodes a JWT token using the ES512 algorithm and validates expiration.

    :param token: The JWT token to decode.
    :return: The decoded token as a dictionary if valid.
    :raises jwt.InvalidTokenError: If the token is invalid or expired.
    """
    try:
        decoded_token = jwt.decode(token, public_key, algorithms=[JWT_ALGORITHM])
        if "exp" not in decoded_token:
            raise jwt.InvalidTokenError("Missing expiration claim in token.")

        if decoded_token["exp"] < time.time():
            raise jwt.ExpiredSignatureError("Token has expired.")

        return decoded_token
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        raise e
    except Exception as e:
        raise jwt.InvalidTokenError(f"Token decoding failed: {str(e)}")
