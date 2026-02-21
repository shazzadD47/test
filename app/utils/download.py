import logging
import os
import pathlib
from typing import Literal
from urllib.parse import urlparse

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from app.configs import settings
from app.logging import logger
from app.utils.utils import sanitize_flag_id

logger = logger.getChild("utils.download")


def get_file_extension_from_url(download_url):
    """
    Extracts the file extension from a download URL, correctly handling
    query parameters often found in signed URLs (like S3 pre-signed URLs).
    """
    # 1. Parse the full URL to separate the path from the query string.
    #    The 'urlparse' function returns a named tuple; we are interested in 'path'.
    parsed_url = urlparse(download_url)

    # 2. The 'path' component is everything before the '?' (query parameters).
    url_path = parsed_url.path

    # 3. Use os.path.splitext to split the path into the root and
    # the extension.
    # This correctly handles cases where the filename itself contains
    # dots (e.g., 'my.file.tar.gz').
    # It returns a tuple: (root, extension)
    _, file_extension = os.path.splitext(url_path)

    return file_extension.strip().lower()


def cache_file_to_path(
    file_extension: str,
    file_bytes: bytes,
    flag_id: str,
    cache_dir_path: pathlib.Path,
) -> pathlib.Path | None:
    """Cache a file to the cache directory."""
    if not file_bytes:
        logger.warning(f"File bytes are empty for flag_id: {flag_id}")
        return None

    # make parent directories if they don't exist
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    cache_file_path = cache_dir_path / f"{flag_id}{file_extension}"
    with open(cache_file_path, "wb") as fo:
        fo.write(file_bytes)

    return cache_file_path


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.75, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def download_file_from_url(url: str) -> tuple[str, bytes]:
    """Download a file from a URL.

    Args:
        url (str): The URL of the file to download.

    Returns:
        tuple[str, bytes]: The file type and the file content in bytes
    """
    try:
        with httpx.Client(timeout=30) as client:
            downloaded_file = client.get(url)
            file_type = downloaded_file.headers["Content-Type"]
            file_content = downloaded_file.content
            return file_type, file_content
    except Exception as e:
        logger.error(f"Error downloading file from URL: {url}")
        logger.error(e)
        return None, None


def get_file_urls_from_fs(
    flag_id: str,
    with_supplementaries: bool = False,
) -> dict[str, str] | str | None:
    """Get the file URLs from the Fullstack API."""
    if with_supplementaries:
        endpoint = settings.BACKEND_FILE_URL_WITH_SUPPLEMENTS_ENDPOINT.format(
            flag_id=flag_id
        )
    else:
        endpoint = settings.BACKEND_FILE_URL_ENDPOINT.format(flag_id=flag_id)

    url = f"{settings.BACKEND_BASE_URL}/{endpoint}"
    try:
        with httpx.Client(timeout=30) as client:
            if with_supplementaries:
                file_urls = client.get(url).json()
                return file_urls
            else:
                file_url = client.get(url).text
                return file_url
    except Exception as e:
        logger.error(f"Error getting file URLs from FS: {url}")
        logger.error(e)
        return None


def get_supplementary_file_url_from_fs(
    supplementary_flag_id: str,
) -> str | None:
    """Get the supplementary file URL from the Fullstack API."""
    flag_id = sanitize_flag_id(supplementary_flag_id)
    file_urls = get_file_urls_from_fs(flag_id, with_supplementaries=True)
    if file_urls and isinstance(file_urls, dict) and "supplementaries" in file_urls:
        for supplement in file_urls["supplementaries"]:
            if supplement["supplementary_flag_id"] == supplementary_flag_id:
                return supplement["url"]
    return None


def download_files_from_flag_id(
    flag_id: str,
    force_redownload: bool = False,
    return_type: Literal["path", "bytes"] = "path",
    return_supplementaries: bool = False,
) -> str | bytes | dict | None:
    """Download a file from the Fullstack API and cache it.

    Args:
        flag_id (str): The flag ID of the file to download.
        force_redownload (bool, optional): Whether to force a redownload of the file. Defaults to False.
        return_type (Literal["path", "bytes"], optional): The type of return value. Defaults to "path".
        return_supplementaries (bool, optional): Whether to return supplementary files as well. Defaults to False.

    Returns:
        str | bytes | dict | None: The path to the file or the file content in bytes if return_supplementaries is False.
        If return_supplementaries is True, returns a dict with main_file and supplementaries keys.
        Returns None if the file is not found or failed to download.
    """  # noqa: E501
    if return_supplementaries:
        return download_files_with_supplements_from_flag_id(
            flag_id,
            force_redownload,
            return_type,
        )
    else:
        flag_id = sanitize_flag_id(flag_id)
        file_url = get_file_urls_from_fs(flag_id, with_supplementaries=False)
        cache_dir_path = settings.PDF_CACHE_DIR / f"{flag_id}"

        if cache_dir_path.exists() and (not force_redownload):
            cached_file_paths = [f for f in cache_dir_path.glob("*") if f.is_file()]
            cached_file_path = (
                cached_file_paths[0] if len(cached_file_paths) == 1 else None
            )

            if len(cached_file_paths) > 1:
                logger.warning(f"Multiple files found in cache for flag_id: {flag_id}")

                for file_path in cached_file_paths:  # clean the cache directory
                    if file_path.is_file():
                        file_path.unlink(missing_ok=True)

            if cached_file_path:
                if return_type == "path":
                    return str(cached_file_path)
                elif return_type == "bytes":
                    with open(cached_file_path, "rb") as f:
                        return f.read()

        if not file_url:
            return None

        _, file_bytes = download_file_from_url(file_url)

        if not file_bytes:
            return None

        cache_dir_path.mkdir(parents=True, exist_ok=True)

        file_extension = get_file_extension_from_url(file_url)
        cache_file_path = cache_file_to_path(
            file_extension, file_bytes, flag_id, cache_dir_path
        )
        if cache_file_path:
            if return_type == "path":
                return str(cache_file_path)
            elif return_type == "bytes":
                return file_bytes
        return None


def download_files_with_supplements_from_flag_id(
    flag_id: str,
    force_redownload: bool = False,
    return_type: Literal["path", "bytes"] = "path",
) -> dict:
    """Download all files (main pdf and supplements) from the Fullstack API and cache them.

    Args:
        flag_id (str): The flag ID of the file to download.
        force_redownload (bool, optional): Whether to force a redownload of all files. Defaults to False.
        return_type (Literal["path", "bytes"], optional): The type of return value.
            Defaults to "path". "bytes" will return the file content in bytes.

    Returns:
        dict: A dictionary containing the path/file content of the main pdf and the supplements.
            Format: {"main_file": str|bytes|None, "supplementaries": list[str|bytes]}
    """  # noqa: E501
    flag_id = sanitize_flag_id(flag_id)
    file_urls = get_file_urls_from_fs(flag_id, with_supplementaries=True)
    return_dict = {"main_file": None, "supplementaries": []}
    cache_dir_path = settings.PDF_CACHE_DIR / f"{flag_id}"
    supplements_dir_path = cache_dir_path / "supplementaries"
    cached_supplementary_flag_ids = []

    if cache_dir_path.exists() and (not force_redownload):
        cached_file_paths = [f for f in cache_dir_path.glob("*") if f.is_file()]
        cached_file_path = cached_file_paths[0] if len(cached_file_paths) == 1 else None

        if len(cached_file_paths) > 1:
            logger.warning(f"Multiple files found in cache for flag_id: {flag_id}")

            for file_path in cached_file_paths:  # clean the cache directory
                if file_path.is_file():
                    file_path.unlink(missing_ok=True)

        if cached_file_path:
            if return_type == "path":
                return_dict["main_file"] = str(cached_file_path)
            elif return_type == "bytes":
                with open(cached_file_path, "rb") as f:
                    return_dict["main_file"] = f.read()

        if supplements_dir_path.exists():
            supplements_file_paths = [
                f for f in supplements_dir_path.glob("*") if f.is_file()
            ]
            if return_type == "path":
                return_dict["supplementaries"] = [
                    str(f) for f in supplements_file_paths
                ]
            elif return_type == "bytes":
                for file_path in supplements_file_paths:
                    with open(file_path, "rb") as f:
                        return_dict["supplementaries"].append(f.read())

            for file_path in supplements_file_paths:
                supplementary_flag_id = file_path.name.removesuffix(".pdf")
                cached_supplementary_flag_ids.append(supplementary_flag_id)

    if not file_urls:
        return return_dict

    if not return_dict["main_file"]:
        main_file_url = file_urls["main_file_url"]

        _, main_file_bytes = download_file_from_url(main_file_url)
        if main_file_bytes:
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            main_file_extension = get_file_extension_from_url(main_file_url)
            cache_file_path = cache_file_to_path(
                main_file_extension, main_file_bytes, flag_id, cache_dir_path
            )
            if cache_file_path:
                if return_type == "path":
                    return_dict["main_file"] = str(cache_file_path)
                elif return_type == "bytes":
                    return_dict["main_file"] = main_file_bytes

    supplementary_file_infos = file_urls["supplementaries"]

    for supplementary_file_info in supplementary_file_infos:
        supplementary_flag_id = supplementary_file_info["supplementary_flag_id"]
        if supplementary_flag_id in cached_supplementary_flag_ids:
            continue

        supplementary_file_url = supplementary_file_info["url"]
        _, supplementary_file_bytes = download_file_from_url(supplementary_file_url)
        if not supplementary_file_bytes:
            continue

        supplementary_file_extension = get_file_extension_from_url(
            supplementary_file_url
        )
        cache_file_path = cache_file_to_path(
            supplementary_file_extension,
            supplementary_file_bytes,
            supplementary_file_info["supplementary_flag_id"],
            supplements_dir_path,
        )
        if cache_file_path:
            if return_type == "path":
                return_dict["supplementaries"].append(str(cache_file_path))
            elif return_type == "bytes":
                return_dict["supplementaries"].append(supplementary_file_bytes)

    return return_dict


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.75, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def async_download_file_from_url(url: str) -> tuple[str, bytes]:
    """Download a file from a URL.

    Args:
        url (str): The URL of the file to download.

    Returns:
        tuple[str, bytes]: The file type and the file content in bytes
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            downloaded_file = await client.get(url)
            file_type = downloaded_file.headers["Content-Type"]
            file_content = downloaded_file.content
            return file_type, file_content
    except Exception as e:
        logger.error(f"Error downloading file from URL: {url}")
        logger.error(e)
        return None, None


async def async_get_file_urls_from_fs(
    flag_id: str,
    with_supplementaries: bool = False,
) -> dict[str, str] | str | None:
    """Get the file URLs from the Fullstack API."""
    if with_supplementaries:
        endpoint = settings.BACKEND_FILE_URL_WITH_SUPPLEMENTS_ENDPOINT.format(
            flag_id=flag_id
        )
    else:
        endpoint = settings.BACKEND_FILE_URL_ENDPOINT.format(flag_id=flag_id)

    url = f"{settings.BACKEND_BASE_URL}/{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if with_supplementaries:
                file_urls = (await client.get(url)).json()
                return file_urls
            else:
                file_url = (await client.get(url)).text
                return file_url
    except Exception as e:
        logger.error(f"Error getting file URLs from FS: {url}")
        logger.error(e)
        return None


async def async_get_supplementary_file_url_from_fs(
    supplementary_flag_id: str,
) -> str | None:
    """Get the supplementary file URL from the Fullstack API."""
    flag_id = sanitize_flag_id(supplementary_flag_id)
    file_urls = await async_get_file_urls_from_fs(flag_id, with_supplementaries=True)
    if file_urls and isinstance(file_urls, dict) and "supplementaries" in file_urls:
        for supplement in file_urls["supplementaries"]:
            if supplement["supplementary_flag_id"] == supplementary_flag_id:
                return supplement["url"]
    return None


async def async_download_files_from_flag_id(
    flag_id: str,
    force_redownload: bool = False,
    return_type: Literal["path", "bytes"] = "path",
    return_supplementaries: bool = False,
) -> str | bytes | dict | None:
    """Download a file from the Fullstack API and cache it.

    Args:
        flag_id (str): The flag ID of the file to download.
        force_redownload (bool, optional): Whether to force a redownload of the file. Defaults to False.
        return_type (Literal["path", "bytes"], optional): The type of return value. Defaults to "path".
        return_supplementaries (bool, optional): Whether to return supplementary files. Defaults to False.

    Returns:
        str | bytes | dict | None: The path to the file or the file content in bytes if return_supplementaries is False.
        If return_supplementaries is True, returns a dict with main_file and supplementaries keys.
        Returns None if the file is not found or failed to download.
    """  # noqa: E501
    if return_supplementaries:
        return await async_download_files_with_supplements_from_flag_id(
            flag_id,
            force_redownload,
            return_type,
        )
    else:
        flag_id = sanitize_flag_id(flag_id)
        file_url = await async_get_file_urls_from_fs(
            flag_id, with_supplementaries=False
        )
        cache_dir_path = settings.PDF_CACHE_DIR / f"{flag_id}"

        if cache_dir_path.exists() and (not force_redownload):
            cached_file_paths = [f for f in cache_dir_path.glob("*") if f.is_file()]
            cached_file_path = (
                cached_file_paths[0] if len(cached_file_paths) == 1 else None
            )

            if len(cached_file_paths) > 1:
                logger.warning(f"Multiple files found in cache for flag_id: {flag_id}")

                for file_path in cached_file_paths:  # clean the cache directory
                    if file_path.is_file():
                        file_path.unlink(missing_ok=True)

            if cached_file_path:
                if return_type == "path":
                    return str(cached_file_path)
                elif return_type == "bytes":
                    with open(cached_file_path, "rb") as f:
                        return f.read()

        if not file_url:
            return None

        _, file_bytes = await async_download_file_from_url(file_url)

        if not file_bytes:
            return None

        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_file_extension = get_file_extension_from_url(file_url)
        cache_file_path = cache_file_to_path(
            cache_file_extension, file_bytes, flag_id, cache_dir_path
        )
        if cache_file_path:
            if return_type == "path":
                return str(cache_file_path)
            elif return_type == "bytes":
                return file_bytes
        return None


async def async_download_files_with_supplements_from_flag_id(
    flag_id: str,
    force_redownload: bool = False,
    return_type: Literal["path", "bytes"] = "path",
) -> dict:
    """Download all files (main pdf and supplements) from the Fullstack API and cache them.

    Args:
        flag_id (str): The flag ID of the file to download.
        force_redownload (bool, optional): Whether to force a redownload of all files. Defaults to False.
        return_type (Literal["path", "bytes"], optional): The type of return value.
            Defaults to "path". "bytes" will return the file content in bytes.

    Returns:
        dict: A dictionary containing the path/file content of the main pdf and the supplements.
            Format: {"main_file": str|bytes|None, "supplementaries": list[str|bytes]}
    """  # noqa: E501
    flag_id = sanitize_flag_id(flag_id)
    file_urls = await async_get_file_urls_from_fs(flag_id, with_supplementaries=True)
    return_dict = {"main_file": None, "supplementaries": []}
    cache_dir_path = settings.PDF_CACHE_DIR / f"{flag_id}"
    supplements_dir_path = cache_dir_path / "supplementaries"
    cached_supplementary_flag_ids = []

    if cache_dir_path.exists() and (not force_redownload):
        cached_file_paths = [f for f in cache_dir_path.glob("*") if f.is_file()]
        cached_file_path = cached_file_paths[0] if len(cached_file_paths) == 1 else None

        if len(cached_file_paths) > 1:
            logger.warning(f"Multiple files found in cache for flag_id: {flag_id}")

            for file_path in cached_file_paths:  # clean the cache directory
                if file_path.is_file():
                    file_path.unlink(missing_ok=True)

        if cached_file_path:
            if return_type == "path":
                return_dict["main_file"] = str(cached_file_path)
            elif return_type == "bytes":
                with open(cached_file_path, "rb") as f:
                    return_dict["main_file"] = f.read()

        if supplements_dir_path.exists():
            supplements_file_paths = [
                f for f in supplements_dir_path.glob("*") if f.is_file()
            ]
            if return_type == "path":
                return_dict["supplementaries"] = [
                    str(f) for f in supplements_file_paths
                ]
            elif return_type == "bytes":
                for file_path in supplements_file_paths:
                    with open(file_path, "rb") as f:
                        return_dict["supplementaries"].append(f.read())

            for file_path in supplements_file_paths:
                supplementary_flag_id = file_path.name.removesuffix(".pdf")
                cached_supplementary_flag_ids.append(supplementary_flag_id)

    if not file_urls:
        return return_dict

    if not return_dict["main_file"]:
        main_file_url = file_urls["main_file_url"]

        _, main_file_bytes = await async_download_file_from_url(main_file_url)
        if main_file_bytes:
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            main_file_extension = get_file_extension_from_url(main_file_url)
            cache_file_path = cache_file_to_path(
                main_file_extension, main_file_bytes, flag_id, cache_dir_path
            )
            if cache_file_path:
                if return_type == "path":
                    return_dict["main_file"] = str(cache_file_path)
                elif return_type == "bytes":
                    return_dict["main_file"] = main_file_bytes

    supplementary_file_infos = file_urls["supplementaries"]

    for supplementary_file_info in supplementary_file_infos:
        supplementary_flag_id = supplementary_file_info["supplementary_flag_id"]
        if supplementary_flag_id in cached_supplementary_flag_ids:
            continue

        supplementary_file_url = supplementary_file_info["url"]
        _, supplementary_file_bytes = await async_download_file_from_url(
            supplementary_file_url
        )
        if not supplementary_file_bytes:
            continue

        supplementary_file_extension = get_file_extension_from_url(
            supplementary_file_url
        )
        cache_file_path = cache_file_to_path(
            supplementary_file_extension,
            supplementary_file_bytes,
            supplementary_file_info["supplementary_flag_id"],
            supplements_dir_path,
        )
        if cache_file_path:
            if return_type == "path":
                return_dict["supplementaries"].append(str(cache_file_path))
            elif return_type == "bytes":
                return_dict["supplementaries"].append(supplementary_file_bytes)

    return return_dict
