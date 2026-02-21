import base64
import gzip
import json
from typing import Any


def decompress_file_contents(compressed_data: str) -> list[dict[str, Any]]:
    """
    Decompress gzip-compressed base64 file contents.

    Args:
        compressed_data: Base64-encoded gzip-compressed JSON string

    Returns:
        List of notebook file dictionaries

    Raises:
        ValueError: If decompression fails
    """
    try:
        # Decode from base64
        compressed_bytes = base64.b64decode(compressed_data)

        # Decompress using gzip
        decompressed_bytes = gzip.decompress(compressed_bytes)

        # Parse JSON
        file_contents = json.loads(decompressed_bytes.decode("utf-8"))

        return file_contents

    except Exception as e:
        raise ValueError(f"Failed to decompress file contents: {str(e)}")


def is_compressed_file_contents(file_contents: str | list) -> bool:
    """Check if file_contents is a compressed string."""
    return isinstance(file_contents, str)


def ensure_file_contents_decompressed(
    file_contents: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """
    Ensure file_contents is decompressed and return as list.

    Args:
        file_contents: Either compressed string or list of file dicts

    Returns:
        List of file content dictionaries
    """
    if file_contents is None:
        return []

    if isinstance(file_contents, str):
        # It's compressed, decompress it
        return decompress_file_contents(file_contents)

    if isinstance(file_contents, list):
        # It's already decompressed
        return file_contents

    # Fallback
    return []
