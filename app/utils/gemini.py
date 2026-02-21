import os
from datetime import datetime, timedelta
from pathlib import Path

from google import genai
from google.genai.types import File

from app.configs import settings
from app.logging import logger
from app.utils.cache import load_from_cache, save_to_cache
from app.utils.files import check_file_llm_usability
from app.utils.texts import convert_data_to_string

logger = logger.getChild("utils.gemini")
__all__ = ["upload_file_to_gemini"]


def extract_data_by_reading_file(file_path: str | Path) -> str | bytes:
    try:
        with open(file_path) as fi:
            return fi.read()
    except Exception as e:
        logger.info(f"Error reading file, error: {e}")
        logger.info("Reading file in bytes and then converting bytes to string")
        with open(file_path, "rb") as fi:
            return convert_data_to_string(fi.read())


def upload_file_to_gemini(file_path: str | Path, flag_id: str) -> File:
    client = genai.Client(
        api_key=settings.GOOGLE_API_KEY, http_options={"timeout": 2 * 60 * 1000}
    )

    cache_key = f"gemini_file_{flag_id}"
    cached_info = load_from_cache(cache_key)

    expiration_time = cached_info.get("expiration_time") if cached_info else None
    if expiration_time:
        expiration_time = datetime.fromisoformat(expiration_time)

    if expiration_time and expiration_time > datetime.now(
        expiration_time.tzinfo
    ) + timedelta(hours=1):
        try:
            return client.files.get(name=cached_info.get("name"))
        except Exception:
            logger.warning(
                "File with flag_id found in cache but not found in gemini. "
                "Re-uploading file..."
            )

    if isinstance(file_path, str):
        file_path = Path(file_path)

    file_name = file_path.name
    try:
        uploaded_file = client.files.upload(
            file=file_path, config={"display_name": file_name}
        )
        valid_file = check_file_llm_usability(
            uploaded_file.mime_type, "gemini-2.5-flash"
        )
        if not valid_file:
            raise Exception("File type not supported")

    except Exception as e:
        logger.info(f"Error uploading file to gemini, error: {e}")
        logger.info("File type not supported by gemini. Converting to text.")
        file_data = extract_data_by_reading_file(file_path)
        extension = os.path.splitext(file_path)[1]
        save_path = str(file_path).replace(extension, ".txt")
        save_path = Path(save_path)
        with open(save_path, "w") as fo:
            fo.write(file_data)
        try:
            uploaded_file = client.files.upload(
                file=save_path, config={"display_name": file_name}
            )
        except Exception as e:
            logger.info(f"Error uploading file to gemini, error: {e}")
            return None

        if os.path.exists(save_path):
            os.remove(save_path)

    save_to_cache(cache_key, uploaded_file.model_dump())

    return uploaded_file
