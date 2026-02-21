import asyncio
from urllib.parse import unquote, urlparse

import fitz
import httpx

from app.configs import settings as global_settings
from app.v3.endpoints.agent_chat.configs import settings
from app.v3.endpoints.agent_chat.constants import FILE_TYPES
from app.v3.endpoints.agent_chat.logging import logger

from ..chains import data_file_chooser_chain


def file_type(file_path: str) -> str:
    extension = file_path.split(".")[-1]
    return FILE_TYPES.get(extension, "unknown")


def get_data_file_contents(
    files: list[str | dict], file_paths: list[str] = None
) -> str:
    contents = []
    for file in files:
        if file_type(file["path"]) == "data":
            if file_paths and file["path"] not in file_paths:
                continue

            contents.append(f"File path in user's machine: {file['path']}")
            contents.append(f"File content:\n{file['content']}\n\n")

    return "\n".join(contents)


def get_code_file_contents(files: list[str | dict], active_notebook: str) -> str:
    contents = []
    for file in files:
        if file_type(file["path"]) == "code":
            contents.append(f"File path in user's machine: {file['path']}")
            if file["path"] == active_notebook:
                contents.append(
                    f"{file['path']} is the active notebook, "
                    "i.e. that user is currently working on this notebook."
                )
            contents.append(f"File content:\n{file['content']}\n\n")

    return "\n".join(contents)


def get_relevant_data_file_paths(
    query: str, contents: list[dict], is_user_selected: bool = False
) -> list[str]:
    paths = []

    if is_user_selected:
        return [content["path"] for content in contents]

    for content in contents:
        if file_type(content["path"]) == "data":
            path = (
                f"File path in user's machine: {content['path']}\n"
                f"File content preview:\n{content['content']}\n\n"
            )
            paths.append(path)

    relevant_paths = []
    if paths:
        chain = data_file_chooser_chain()
        result = chain.invoke(
            {
                "data_files": paths,
                "query": query,
            }
        )
        relevant_paths = result if isinstance(result, list) else [result]

    return relevant_paths


async def get_file_contents(
    project_id: str,
    file_path: str,
    access_token: str,
) -> dict:
    """
    Download/get the contents of a specific file from the backend API.

    Args:
        project_id: The project ID
        file_path: The file path
            (e.g., "exports/Sample Data-2025-05-26T22_12_22.922Z.csv")
        access_token: Authorization token

    Returns:
        dict: Response containing file contents and path

    Raises:
        Exception: If the API request fails
    """
    try:
        url = settings.FILE_CONTENTS_ENDPOINT.format(project_id=project_id)
        async with httpx.AsyncClient(timeout=global_settings.API_TIME_OUT) as client:
            response = await client.get(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {access_token}",
                },
                params={
                    "file_path": file_path,
                },
            )
            if response.status_code != 200:
                raise Exception(
                    f"{response.status_code}! Something went wrong "
                    "while getting file contents"
                )

            # Handle both JSON and text responses
            try:
                content = response.json()
            except Exception:
                content = response.text

            return {"path": file_path, "content": content}

    except httpx.HTTPError as e:
        logger.error("HTTP error while getting file contents: %s", str(e))
        raise Exception(f"Failed to get file contents: {str(e)}") from e
    except Exception as e:
        logger.error("Error getting file contents: %s", str(e))
        raise Exception(f"Failed to get file contents: {str(e)}") from e


async def fetch_file_contents_parallel(
    file_paths: list[str],
    project_id: str,
    access_token: str,
) -> list[dict]:
    """
    Fetch multiple file contents in parallel.

    Args:
        file_paths: List of file paths to fetch
        project_id: The project ID
        access_token: Authorization token

    Returns:
        list[dict]: List of file contents with path and content

    Raises:
        Exception: If fetching file contents fails
    """
    if not file_paths:
        return []

    # Fetch all file contents in parallel
    tasks = [
        get_file_contents(
            project_id=project_id, file_path=file_path, access_token=access_token
        )
        for file_path in file_paths
    ]

    file_contents = await asyncio.gather(*tasks)
    logger.info("File contents fetched: %s files", len(file_contents))
    return file_contents


def is_pdf_content(content: bytes) -> bool:
    """
    Check if the content is a PDF file by examining magic bytes.

    Args:
        content: The file content as bytes

    Returns:
        bool: True if content appears to be a PDF file
    """
    if not content or len(content) < 4:
        return False
    # PDF files start with %PDF
    return content[:4] == b"%PDF"


def extract_text_from_pdf_bytes(content: bytes, file_path: str = "unknown") -> str:
    """
    Extract text content from PDF bytes, removing images and binary data.

    Args:
        content: The PDF file content as bytes
        file_path: The file path for logging purposes

    Returns:
        str: Extracted text content from the PDF
    """
    try:
        original_size = len(content)
        logger.debug(
            f"Extracting text from PDF: {file_path} (size: {original_size} bytes)"
        )

        # Open PDF from bytes using fitz (PyMuPDF)
        pdf_document = fitz.open(stream=content, filetype="pdf")

        extracted_text = []
        page_count = pdf_document.page_count

        for page_num in range(page_count):
            page = pdf_document[page_num]
            text = page.get_text()
            if text.strip():
                extracted_text.append(f"--- Page {page_num + 1} ---\n{text}")

        pdf_document.close()

        result_text = "\n\n".join(extracted_text)
        result_size = len(result_text.encode("utf-8"))

        # Log compression ratio
        compression_ratio = (
            (1 - result_size / original_size) * 100 if original_size > 0 else 0
        )
        logger.info(
            f"PDF text extracted from {file_path}: "
            f"{page_count} pages, "
            f"original size: {original_size} bytes, "
            f"text size: {result_size} bytes, "
            f"compression: {compression_ratio:.1f}%"
        )

        return result_text

    except Exception as e:
        logger.error(f"Failed to extract text from PDF {file_path}: {str(e)}")
        # Fallback to returning a message about the error
        return f"[PDF text extraction failed for {file_path}: {str(e)}]"


def _derive_stash_file_path(file_url: str) -> str:
    try:
        parsed = urlparse(file_url)
        path = parsed.path.lstrip("/")
        if path:
            return unquote(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to parse stash file url %s: %s", file_url, exc)
    return file_url


async def _download_stash_file(
    client: httpx.AsyncClient, file_url: str
) -> dict[str, str | bytes]:
    response = await client.get(file_url)
    if response.status_code != 200:
        raise Exception(
            f"{response.status_code}! Failed to download stash file from {file_url}"
        )

    path = _derive_stash_file_path(file_url)
    logger.debug(
        "Downloaded stash file %s (content_type=%s, size=%s)",
        path,
        response.headers.get("Content-Type"),
        len(response.content),
    )
    return {"path": path, "content": response.content, "is_stash_file": True}


async def fetch_stash_file_contents(file_urls: list[str]) -> list[dict]:
    """
    Download stash file contents directly from their pre-signed URLs.

    Args:
        file_urls: List of pre-signed URLs pointing to stash files.

    Returns:
        list[dict]: Each item contains {"path": <file path>, "content": <bytes>}.
    """
    if not file_urls:
        return []

    async with httpx.AsyncClient(timeout=global_settings.API_TIME_OUT) as client:
        tasks = [_download_stash_file(client, file_url) for file_url in file_urls]
        file_contents = await asyncio.gather(*tasks)
        logger.info("Stash file contents fetched: %s files", len(file_contents))
        return file_contents
