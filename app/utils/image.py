import base64
from io import BytesIO
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from app.logging import logger

logger = logger.getChild("get_image")


def get_image_from_url(
    url: str, return_media_type: bool = False, return_pil_image: bool = False
):
    """
    Synchronously fetch an image from a URL.

    Args:
        url (str): The URL of the image to fetch.
        return_media_type (bool): Whether to return the media type along with the image.

    Returns:
        bytes: The image content as bytes. If return_media_type is True,
          returns a tuple of (image, media_type).

    Raises:
        HTTPException: If the image cannot be fetched due to
        connection issues or other errors.
    """
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.exception(f"HTTP error occurred: {e}")
        raise HTTPException(
            status_code=response.status_code,
            detail="Failed to fetch image due to an HTTP error.",
        )
    except httpx.RequestError:
        logger.exception("Network error occurred.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to fetch image due to a network error.",
        )
    except Exception:
        logger.exception("Failed to fetch image.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to fetch image.",
        )

    try:
        image = response.content
        image = Image.open(BytesIO(image))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The URL does not point to a valid image.",
        )
    except Exception:
        logger.exception("Failed to open downloaded image.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to open image.",
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    media_type = "image/png"
    if return_pil_image and return_media_type:
        return image, media_type

    elif return_pil_image:
        return image

    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")

    image_bytes = image_bytes.getvalue()

    if return_media_type:
        return image_bytes, media_type

    return image_bytes


async def async_get_image_from_url(
    url: str, return_media_type: bool = False, return_pil_image: bool = False
):
    """
    Asynchronously fetch an image from a URL.

    Args:
        url (str): The URL of the image to fetch.
        return_media_type (bool): Whether to return the media type along with the image.

    Returns:
        bytes: The image content as bytes. If return_media_type is True,
          returns a tuple of (image, media_type).

    Raises:
        HTTPException: If the image cannot be fetched due to
        connection issues or other errors.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.exception(f"HTTP error occurred: {e}")
        raise HTTPException(
            status_code=response.status_code,
            detail="Failed to fetch image due to an HTTP error.",
        )
    except httpx.RequestError:
        logger.exception("Network error occurred.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to fetch image due to a network error.",
        )
    except Exception:
        logger.exception("Failed to fetch image.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to fetch image.",
        )

    try:
        image = response.content
        image = Image.open(BytesIO(image))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The URL does not point to a valid image.",
        )
    except Exception:
        logger.exception("Failed to open downloaded image.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to open image.",
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    media_type = "image/png"
    if return_pil_image and return_media_type:
        return image, media_type

    elif return_pil_image:
        return image

    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")

    image_bytes = image_bytes.getvalue()

    if return_media_type:
        return image_bytes, media_type

    return image_bytes


def convert_image_to_base64(image: bytes) -> str:
    """
    Convert an image to a base64 string.

    Args:
        image (bytes): The image content as bytes.

    Returns:
        str: The image content as a base64 string.
    """
    return base64.b64encode(image).decode("utf-8")


def is_color(img):
    if img.ndim <= 2:
        return False

    if img.shape[-1] == 1:
        return False

    return True


def draw_lines(img, lines, color):
    if is_color(img):
        annot_img = img.copy()
    else:
        annot_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    classes = list(range(len(lines)))

    if len(classes):
        count = 0
        for _, line in enumerate(lines):
            count = count + 1
            drawing_lines = []
            for pt_idx in range(len(line) - 1):
                drawing_lines.append([line[pt_idx], line[pt_idx + 1]])
            annot_img = cv2.polylines(
                annot_img,
                np.array(drawing_lines),
                isClosed=False,
                color=color,
                thickness=2,
            )

    return annot_img


def combine_images_vertically_with_padding(
    image1_path: str | Path,
    image2_path: str | Path,
    bounding_box_legend: dict | None = None,
    bounding_box: dict | None = None,
    padding: int = 30,
    return_media_type: bool = False,
) -> bytes:
    # Open the two images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    if bounding_box_legend and bounding_box is not None:
        bbox_coords = [
            bounding_box["top_left_x"],
            bounding_box["top_left_y"],
            bounding_box["bottom_right_x"],
            bounding_box["bottom_right_y"],
        ]
        bbox_legend_coords = [
            bounding_box_legend["top_left_x"],
            bounding_box_legend["top_left_y"],
            bounding_box_legend["bottom_right_x"],
            bounding_box_legend["bottom_right_y"],
        ]
        bbox_plot_height = abs(bbox_coords[3] - bbox_coords[1])
        bbox_legend_height = abs(bbox_legend_coords[3] - bbox_legend_coords[1])

        bbox_plot_width = abs(bbox_coords[2] - bbox_coords[0])
        bbox_legend_width = abs(bbox_legend_coords[2] - bbox_legend_coords[0])

        plot_heigth, plot_width = image1.height, image1.width

        new_legend_height = int(bbox_legend_height * plot_heigth / bbox_plot_height)
        new_legend_width = int(bbox_legend_width * plot_width / bbox_plot_width)

        image2 = image2.resize((new_legend_width, new_legend_height))

    # Calculate the new width and height
    new_width = max(image1.width, image2.width)
    new_height = image1.height + image2.height + padding

    # Create a new image with a white background
    combined_image = Image.new("RGB", (new_width, new_height), "white")

    # Paste the first image at the top
    combined_image.paste(image1, (0, 0))

    # Center the second image horizontally and paste it with padding
    x_offset = (new_width - image2.width) // 2
    combined_image.paste(image2, (x_offset, image1.height + padding))

    # Save the image to an in-memory bytes buffer
    image_bytes = BytesIO()
    combined_image.save(image_bytes, format="PNG")

    image_bytes = image_bytes.getvalue()

    media_type = "image/png"

    if return_media_type:
        return image_bytes, media_type

    return image_bytes
