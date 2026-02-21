import hashlib
import os
import time

from dotenv import load_dotenv

load_dotenv()

API_SECRET = os.getenv("API_SECRET_KEY")


def generate_salt():
    return str(int(time.time()))


def generate_signature(salt, secret=API_SECRET):
    return hashlib.sha256(f"{salt}:{secret}".encode()).hexdigest()


def generate_api_key():
    salt = generate_salt()
    signature = generate_signature(salt)
    return f"{salt}:{signature}"


if __name__ == "__main__":
    api_key = generate_api_key()
    print(f"Generated API Key: {api_key}")
