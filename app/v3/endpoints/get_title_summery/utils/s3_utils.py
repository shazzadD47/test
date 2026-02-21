from celery.utils.log import get_task_logger
from fastapi import HTTPException, status
from fitz import Document, Matrix
from PIL import Image

from app.configs import settings

celery_logger = get_task_logger(__name__)


def build_public_s3_url(object_key: str) -> str:
    """
    Builds a public S3 URL for a given object key using settings.

    Args:
        object_key (str): The key/path of the object in the S3 bucket.

    Returns:
        str: The full public URL of the object.
    """
    return f"{settings.S3_SPACES_PUBLIC_BASE_URL}/{object_key}"


def extract_page_dimensions(
    document: Document, page_number: int, resolution: int, flag_id: str
) -> dict[str, int]:
    """
    Extract width and height of a specific page from a PDF document.

    Args:
        document (Document): The PDF document object.
        page_number (int): The page number to extract.
        resolution (int): The resolution to scale the image for dimension accuracy.
        flag_id (str): Used only for logging/debugging.

    Returns:
        dict: {"width": int, "height": int}
    """
    try:
        page = document.load_page(page_number)
        pixmap = page.get_pixmap(
            matrix=Matrix(float(resolution) / 72, float(resolution) / 72)
        )

        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        width, height = image.size
        return {"width": width, "height": height}

    except Exception as e:
        celery_logger.exception(
            f"[flag_id: {flag_id}, page: {page_number}] Failed extract dimensions: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract dimensions: {e}",
        )
