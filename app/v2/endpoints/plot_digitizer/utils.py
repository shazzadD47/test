import httpx
from fastapi import HTTPException, status

from app.logging import logger


def get_image_from_url(url: str, return_media_type: bool = False) -> bytes:
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must start with 'http://' or 'https://'.",
        )

    try:
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.RequestError:
        logger.exception("An error occurred while fetching the image")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to fetch image. Please check the URL and try again.",
        )
    except httpx.HTTPStatusError as exc:
        logger.exception(
            f"""HTTP error occurred: {exc.response.status_code}
            {exc.response.reason_phrase}"""
        )
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"""HTTP error occurred while fetching the image:
            {exc.response.reason_phrase}""",
        )
    except Exception:
        logger.exception("Unexpected error occurred while fetching the image")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while fetching the image.",
        )

    image = response.content
    image_type = response.headers.get("content-type")

    if return_media_type:
        return image, image_type

    return image
