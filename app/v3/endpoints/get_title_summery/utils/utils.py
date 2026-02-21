import base64
import hashlib
import logging
import mimetypes
import os
import re
import secrets
from io import BytesIO
from uuid import UUID, uuid4

from boto3 import session
from bs4 import BeautifulSoup
from fastapi import HTTPException, status
from fitz import Document, Matrix
from PIL import Image
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.configs import settings
from app.core.database.models import PageDetails
from app.v3.endpoints.get_title_summery.logging import logger
from app.v3.endpoints.get_title_summery.services.crud import (
    insert_page_details,
    update_legend_paths,
)


def clean_jats_tags(text: str) -> str:
    """
    Cleans JATS tags from the given text using BeautifulSoup.

    Args:
        text (str): The text containing JATS tags.

    Returns:
        str: The cleaned text without JATS tags.
    """
    if isinstance(text, list):
        text = " ".join(text)

    soup = BeautifulSoup(text, "lxml")

    for tag in soup.find_all():
        tag.replace_with(tag.text)

    clean_text = soup.get_text()

    return clean_text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.75, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def upload_page_buffer_to_storage(
    image_buffer: BytesIO, upload_path: str, page_number: int
) -> str:
    """
    Uploads the given image buffer to file storage with retry logic.

    Args:
        image_buffer (BytesIO): The image buffer to upload.
        upload_path (str): The path where the image will be uploaded.
        page_number (int): The page number of the image.

    Raises:
        HTTPException: If there is an error during the upload process.

    Returns:
        str: The object key of the uploaded image.
    """
    session_ = session.Session()
    object_key = f"{upload_path}/page_{page_number}.png"

    client = session_.client(
        "s3",
        region_name=settings.S3_SPACES_REGION,
        endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
        aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
    )

    try:
        image_buffer.seek(0)  # Ensure the buffer is at the start
        client.upload_fileobj(
            image_buffer,
            settings.S3_SPACES_BUCKET,
            object_key,
            ExtraArgs={"ContentType": "image/png", "ACL": "public-read"},
        )
        return object_key
    except Exception as e:
        logger.exception(f"Upload failed on page {page_number}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {e}",
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.75, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def upload_file_to_storage(file_path: str, upload_path: str, filename: str) -> str:
    """
    Upload a file from disk to object storage with retry logic.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File path not found: {file_path}")

    object_key = f"{upload_path}/{filename}"
    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "application/octet-stream"

    session_ = session.Session()
    client = session_.client(
        "s3",
        region_name=settings.S3_SPACES_REGION,
        endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
        aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
    )

    try:
        with open(file_path, "rb") as f:
            client.upload_fileobj(
                f,
                settings.S3_SPACES_BUCKET,
                object_key,
                ExtraArgs={"ContentType": content_type, "ACL": "public-read"},
            )
        return object_key
    except Exception as e:
        logger.exception("File upload failed for '%s': %s", filename, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {e}",
        ) from e


def upload_legend_to_storage(
    image_buffer: BytesIO, upload_path: str, figure_id: UUID
) -> str:
    """
    Uploads the given image buffer to file storage.

    Args:
        image_buffer (BytesIO): The image buffer to upload.
        upload_path (str): The path where the image will be uploaded.
        page_number (int): The page number of the image.

    Raises:
        HTTPException: If there is an error during the upload process.

    Returns:
        str: The object key of the uploaded image.
    """
    session_ = session.Session()
    object_key = f"{upload_path}/{figure_id}.png"

    client = session_.client(
        "s3",
        region_name=settings.S3_SPACES_REGION,
        endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
        aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
    )

    try:
        client.upload_fileobj(
            image_buffer,
            settings.S3_SPACES_BUCKET,
            object_key,
            ExtraArgs={"ContentType": "image/png", "ACL": "public-read"},
        )

        return object_key
    except Exception as e:
        logger.exception(f"Error: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {e}"
        )


def convert_uploaded_image_to_pdf_style_resolution(image_bytes: bytes, resolution: int):
    image = Image.open(BytesIO(image_bytes))
    scale_factor = resolution / 72
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    output_buffer = BytesIO()
    resized_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return output_buffer, new_width, new_height


def extract_and_upload_page(
    document: Document,
    page_number: int,
    resolution: int,
    flag_id: str,
    project_id: str,
    supp_id: str = None,
) -> PageDetails:
    """
    Extracts a specific page from a document, converts it to an image,
    and uploads it to file storage.

    Args:
        document (Document): The document from which to extract the page.
        page_number (int): The page number to extract.
        resolution (int): The resolution of the extracted image.
        flag_id (str): The flag ID used for uploading the image.
        supp_id (str, optional): The supplemental ID used for uploading the image.

    Raises:
        HTTPException: If there is an error during the extraction or upload process.

    Returns:
        None
    """
    try:
        page = document.load_page(page_number)
        image = page.get_pixmap(
            matrix=Matrix(float(resolution) / 72, float(resolution) / 72)
        )

        image = Image.frombytes("RGB", (image.width, image.height), image.samples)
        width, height = image.size

        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        if supp_id:
            upload_path = f"documents/pages/{flag_id}/{supp_id}"
        else:
            upload_path = f"documents/pages/{flag_id}"

        object_key = upload_page_buffer_to_storage(
            image_bytes, upload_path, page_number + 1
        )

        page_details = PageDetails(
            flag_id=flag_id,
            project_id=project_id,
            page_number=page_number + 1,
            bucket_path=object_key,
            image_width=width,
            image_height=height,
        )

        image_bytes.close()

        return page_details
    except Exception as e:
        logger.exception(f"[flag_id: {flag_id}] Error: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {e}"
        )


def upload_single_image_to_storage(
    image_bytes: bytes, flag_id: str, project_id: str, supp_id: str = None
):

    try:
        converted_buffer, width, height = (
            convert_uploaded_image_to_pdf_style_resolution(
                image_bytes, resolution=settings.PDF_PAGE_IMAGE_RESOLUTION
            )
        )

        if supp_id:
            upload_path = f"documents/pages/{flag_id}/{supp_id}"
        else:
            upload_path = f"documents/pages/{flag_id}"

        object_key = upload_page_buffer_to_storage(
            image_buffer=converted_buffer,
            upload_path=upload_path,
            page_number=0,
        )
        page_details = PageDetails(
            flag_id=flag_id,
            project_id=project_id,
            page_number=0,
            bucket_path=object_key,
            image_width=width,
            image_height=height,
        )
        converted_buffer.close()

        insert_page_details([page_details])
        return {
            "message": "Single image uploaded successfully",
            "image_url": object_key,
        }
    except Exception as e:
        logger.exception(f"[flag_id: {flag_id}] Error uploading image")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image upload failed: {str(e)}",
        )


def upload_legend(
    image_buffer: BytesIO,
    figure_id: UUID,
    flag_id: str,
) -> str:

    try:

        upload_path = f"documents/legends/{flag_id}"

        object_key = upload_legend_to_storage(image_buffer, upload_path, figure_id)
        update_legend_paths(figure_id=figure_id, legend_paths=[object_key])

        return object_key
    except Exception as e:
        logger.exception(f"[flag_id: {flag_id}] Error: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {e}"
        )


def encode_image(image: str | bytes) -> str:
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")

    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_supp_id():

    supp_id = hashlib.sha256(uuid4().hex.encode()).digest()

    start = secrets.randbelow(len(supp_id) - 10)
    supp_id = supp_id[start : start + 6].hex()

    return supp_id


def secure_filename(filename: str) -> str:
    """
    Sanitize a filename to ensure it doesn't contain any potentially dangerous
    characters.

    Args:
        filename (str): The original filename

    Returns:
        str: A sanitized filename
    """
    filename = os.path.basename(filename)

    filename = re.sub(r"[^\w\-\.]", "_", filename)

    if not filename:
        filename = "unnamed_file"

    return filename


def secure_file_path(base_dir: str, filename: str) -> str:
    """
    Create a secure file path that is guaranteed to be within the intended directory.

    Args:
        base_dir (str): The base directory where the file should be stored
        filename (str): The filename (already sanitized)

    Returns:
        str: A secure absolute file path

    Raises:
        HTTPException: If the resulting path would be outside the base directory
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    full_path = os.path.normpath(os.path.join(base_dir, filename))

    if not full_path.startswith(base_dir):
        logger.error(f"Path traversal attempt detected: {filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename",
        )

    return full_path
