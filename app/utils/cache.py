import json
import os

from google.genai import types
from google.genai.client import Client

from app.logging import logger
from app.redis import redis_client
from app.utils.json_utils import DelineateJSONEncoder


def save_to_cache(key: str, data: dict, expiration: int = 60 * 60 * 24):
    """Save data to Redis cache."""
    redis_client.setex(key, expiration, json.dumps(data, cls=DelineateJSONEncoder))
    return "saved to redis cache"


def load_from_cache(key: str) -> dict | None:
    """Load data from Redis cache."""
    cached_data = redis_client.get(key)
    return json.loads(cached_data) if cached_data else None


def retrieve_gemini_cache(cache_display_name: str) -> bool:
    client = Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    for cache in client.caches.list():
        if cache.display_name == cache_display_name:
            return cache
    return None


def create_gemini_cache(
    messages: list[types.Part],
    model_name: str,
    cache_name: str,
    ttl: str = "600s",
    system_instruction: str = None,
) -> None:
    logger.info(f"Creating gemini cache: {cache_name}")
    client = Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    if system_instruction:
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=cache_name,
                system_instruction=system_instruction,
                contents=messages,
                ttl=ttl,
            ),
        )
    else:
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=cache_name,
                contents=messages,
                ttl=ttl,
            ),
        )
    return cache


def batch_save_to_cache(mappings: dict[str, dict], expiration: int = 60 * 60 * 24):
    """Save multiple key-value pairs in Redis cache in one pipeline."""
    if not mappings:
        return "no data to save"
    try:
        pipe = redis_client.pipeline()
        for key, data in mappings.items():
            pipe.setex(key, expiration, json.dumps(data, cls=DelineateJSONEncoder))
        pipe.execute()
        return f"saved {len(mappings)} entries to redis cache"
    except Exception as e:
        logger.exception(f"Batch save error: {e}")
        return f"error: {e}"
