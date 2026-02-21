import io
from dataclasses import dataclass

import cv2
import numpy as np
import requests
from langchain_anthropic import ChatAnthropic

from app.configs import settings
from app.core.database.models import FigureDetails
from app.v3.endpoints.get_title_summery.configs import settings as project_settings
from app.v3.endpoints.get_title_summery.logging import logger
from app.v3.endpoints.get_title_summery.utils.crop_legends import crop_legend


@dataclass
class ImageData:
    buffer_reader: io.BufferedReader
    image: np.ndarray


# Initialize Anthropic client
client = ChatAnthropic(
    api_key=settings.ANTHROPIC_API_KEY,
    model=project_settings.CLAUDE_MODEL,
    temperature=0.2,
    max_tokens_to_sample=4096,
)


def get_figures_subfigures(
    figure_details: list[FigureDetails],
) -> tuple[dict, dict, dict]:
    """
    Organize figures and their subfigures into mappings.

    Args:
        figure_details: List of FigureDetails objects

    Returns:
        Tuple containing:
        - parent_map: Dict mapping parent figure IDs to their bucket paths
        - parent_subfigure_map: Dict mapping parent IDs to lists of subfigure IDs
        - subfigure_map: Dict mapping subfigure IDs to their bucket paths
    """
    parent_map = {
        img.figure_id: img.bucket_path
        for img in figure_details
        if img.parent_figure_id is None
    }

    parent_subfigure_map = {}
    subfigure_map = {}

    for img in figure_details:
        if img.parent_figure_id in parent_map:
            if img.parent_figure_id not in parent_subfigure_map:
                parent_subfigure_map[img.parent_figure_id] = []
            parent_subfigure_map[img.parent_figure_id].append(img.figure_id)
            subfigure_map[img.figure_id] = img.bucket_path

    return parent_map, parent_subfigure_map, subfigure_map


def load_image_from_url(url: str) -> ImageData:
    """Load and process an image from a URL."""
    response = requests.get(url, timeout=50)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    _, buffer = cv2.imencode(".png", image)
    image_io = io.BytesIO(buffer)
    buffered_reader = io.BufferedReader(image_io)

    return ImageData(buffered_reader, image)


def process_subfigures(
    subfigure_ids: list[str], subfigure_map: dict, url_start: str
) -> tuple[list, list, list]:
    """Process all subfigures for a given parent figure."""
    subfigure_list = []
    subfigure_image_list = []

    for subfigure_id in subfigure_ids:
        subfigure_url = (
            f"{url_start}/{settings.S3_SPACES_BUCKET}/{subfigure_map[subfigure_id]}"
        )
        try:
            image_data = load_image_from_url(subfigure_url)
            subfigure_list.append(image_data.buffer_reader)
            subfigure_image_list.append(image_data.image)
        except Exception as e:
            logger.exception(f"Failed to process subfigure {subfigure_url}: {str(e)}")
            continue

    return subfigure_list, subfigure_image_list, subfigure_ids


def detect_legends(flag_id: str, figure_details: list[FigureDetails]) -> None:
    """
    Detect legends in figures and their subfigures.

    Args:
        flag_id: Identifier for the processing run
        figure_details: List of figure details to process
    """
    url_start = settings.S3_SPACES_PUBLIC_BASE_URL
    logger.debug("detecting legends test ")
    try:
        parent_map, parent_subfigure_map, subfigure_map = get_figures_subfigures(
            figure_details
        )
    except Exception:
        logger.exception(f"[flag_id: {flag_id}] Failed to process figure mappings")
        return

    for parent_id, subfigure_ids in parent_subfigure_map.items():
        parent_url = f"{url_start}/{parent_map[parent_id]}"
        try:
            logger.debug(f"parent_url: {parent_url}")
            parent_data = load_image_from_url(parent_url)
        except Exception:
            logger.exception(
                f"[flag_id: {flag_id}] Failed to load parent figure {parent_id}"
            )
            continue

        subfigure_list, subfigure_image_list, subfigure_ids = process_subfigures(
            subfigure_ids, subfigure_map, url_start
        )

        top_box = None
        bbox_list = None
        bounding_box_list = None
        total_result = None

        for idx, subfigure in enumerate(subfigure_image_list):
            try:
                crop_legend_args = {
                    "extract_img": subfigure,
                    "extracted_img_path": subfigure_list[idx],
                    "total_image": parent_data.image,
                    "total_img_path": parent_data.buffer_reader,
                    "figure_id": str(subfigure_ids[idx]),
                    "client": client,
                    "flag_id": flag_id,
                }

                if top_box is not None:
                    crop_legend_args.update(
                        {
                            "top_box": top_box,
                            "bbox_list": bbox_list,
                            "bounding_box_list": bounding_box_list,
                            "total_result": total_result,
                        }
                    )

                result = crop_legend(**crop_legend_args)
                top_box, bbox_list, bounding_box_list, total_result, _ = result

            except Exception:
                logger.exception(
                    f"[flag_id: {flag_id}] Legend detection failed for"
                    f" subfigure {subfigure_ids[idx]}"
                )
                continue
