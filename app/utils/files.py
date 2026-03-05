import base64
from mimetypes import guess_type
from pathlib import Path
from uuid import uuid4

from app.core.auto.chat_model import (
    ANTHROPIC_MODEL_MAPPING,
    GOOGLE_GENAI_MODEL_MAPPING,
    OPENAI_MODEL_MAPPING,
)
from app.logging import logger
from app.utils.texts import convert_data_to_string


def _read_and_encode_file(file_path: str) -> tuple[str, str]:
    """Read a file and return (base64_data, mime_type)."""
    with open(file_path, "rb") as f:
        file_content = f.read()

    mime_type, _ = guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    b64 = base64.b64encode(file_content).decode("utf-8")
    return b64, mime_type


def create_data_uri_schema(
    file_path: str, mime_type: str | None = None, return_mime_type: bool = False
):
    if mime_type is None:
        b64, detected_mime = _read_and_encode_file(str(file_path))
        mime_type = detected_mime
    else:
        b64, _ = _read_and_encode_file(str(file_path))

    data_uri = f"data:{mime_type};base64,{b64}"

    if return_mime_type:
        return data_uri, mime_type
    else:
        return data_uri


def check_file_llm_usability(mime_type: str, model_name: str) -> bool:
    genai_supported_mimetypes = [
        "text/csv",
        "text/javascript",
        "text/markdown",
        "application/pdf",
        "image/png",
        "image/jpeg",
        "text/x-python",
        "text/plain",
    ]
    openai_supported_mimetypes = [
        "application/pdf",
        "image/png",
        "image/jpeg",
    ]
    claude_supported_mimetypes = [
        "application/pdf",
        "image/png",
        "image/jpeg",
    ]
    if model_name in GOOGLE_GENAI_MODEL_MAPPING:
        return mime_type in genai_supported_mimetypes
    elif model_name in OPENAI_MODEL_MAPPING:
        return mime_type in openai_supported_mimetypes
    elif model_name in ANTHROPIC_MODEL_MAPPING:
        return mime_type in claude_supported_mimetypes
    else:
        return False


def create_file_input(file_path: str | Path, model_name) -> dict:
    data_uri, mime_type = create_data_uri_schema(file_path, return_mime_type=True)
    is_image = mime_type.startswith("image")

    if is_image:
        file_data = {
            "type": "image",
            "source_type": "base64",
            "mime_type": mime_type,
            "data": data_uri.removeprefix(f"data:{mime_type};base64,"),
        }
    else:
        file_data = {
            "type": "file",
            "source_type": "base64",
            "data": data_uri.removeprefix(f"data:{mime_type};base64,"),
            "mime_type": mime_type,
            "filename": f"{uuid4()}_{uuid4()}",
        }

    valid_data = check_file_llm_usability(mime_type, model_name)

    if valid_data:
        return file_data
    else:
        try:
            with open(file_path) as fi:
                file_data = {"type": "text", "text": fi.read()}
                return file_data
        except Exception as e:
            logger.info(f"Error reading file, error: {e}")
            logger.info("Reading file again in byte mode")
            with open(file_path, "rb") as fi:
                file_data = {"type": "text", "text": convert_data_to_string(fi.read())}
                return file_data
