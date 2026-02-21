import hashlib
import time


def generate_signature(salt: str, secret: str):
    return hashlib.sha256(f"{salt}:{secret}".encode()).hexdigest()


def generate_salt():
    return str(int(time.time()))


def generate_api_token(secret):
    salt = generate_salt()
    signature = generate_signature(salt, secret)
    return f"{salt}:{signature}"
