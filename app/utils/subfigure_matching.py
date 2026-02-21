from sqlalchemy import select
from sqlalchemy.sql import func

from app.core.database.crud import select_with_retry
from app.core.database.models import FigureDetails
from app.logging import logger


def compute_gtc(ground_truth: tuple, prediction: tuple) -> float:
    """
    Compute the Ground Truth Coverage (GTC) metric.

    Parameters:
    - ground_truth: Tuple (x_min, y_min, x_max, y_max) for ground truth bounding box.
    - prediction: Tuple (x_min, y_min, x_max, y_max) for predicted bounding box.

    Returns:
    - GTC value.
    """
    x_min_inter = max(ground_truth[0], prediction[0])
    y_min_inter = max(ground_truth[1], prediction[1])
    x_max_inter = min(ground_truth[2], prediction[2])
    y_max_inter = min(ground_truth[3], prediction[3])

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection = inter_width * inter_height

    gt_width = ground_truth[2] - ground_truth[0]
    gt_height = ground_truth[3] - ground_truth[1]
    gt_area = gt_width * gt_height

    if gt_area == 0:
        return 0.0

    gtc = intersection

    return gtc


def match_by_bbox(
    bbox: list[int], figure_details: list[FigureDetails]
) -> FigureDetails | None:
    if figure_details is None or len(figure_details) == 0:
        logger.debug(f"No figure details to match against bbox {bbox}")
        return None

    overlap = float("-inf")
    matched_figure = None

    for figure in figure_details:
        gtc = compute_gtc(bbox, figure.converted_bbox_coordinates)

        if gtc > overlap:
            overlap = gtc
            matched_figure = figure

    return matched_figure


def get_matching_figure(
    flag_id: str, page: int, bbox: list[int]
) -> FigureDetails | None:
    children = FigureDetails.__table__.alias("children")

    query = (
        select(FigureDetails)
        .where(
            FigureDetails.flag_id == flag_id,
            FigureDetails.page_number == page,
        )
        .outerjoin(children, FigureDetails.figure_id == children.c.parent_figure_id)
        .group_by(FigureDetails)
        .having(
            (FigureDetails.parent_figure_id.is_not(None))
            | (func.count(children.c.id) == 0)
        )
    )

    try:
        result = select_with_retry(query)

        return match_by_bbox(bbox, result)
    except Exception as e:
        logger.exception(f"Error matching figure: {e}")
        return None
