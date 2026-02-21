import copy

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import HttpUrl

from app.logging import logger
from app.v2.endpoints.plot_digitizer.template_matching.matching_iou import (
    detect_bb_box_error_bar,
)
from app.v2.endpoints.plot_digitizer.tick_origin_detection.axes_min_max_tick_based import (  # noqa E501
    detect_x_y_ticks,
)
from app.v2.endpoints.plot_digitizer.tick_origin_detection.origin_detect_local import (
    detect_axes_origin_min_max_local,
)
from app.v2.endpoints.plot_digitizer.tick_origin_detection.services import (
    crop_lower_intersection,
    morphological_detection,
    origin_based_axes_detection,
)
from app.v2.endpoints.plot_digitizer.utils import get_image_from_url

plot_digitization_router = APIRouter(tags=["plot_digitization"])


@plot_digitization_router.get("/plot-digitizer/detect-axes-origin-min-max")
async def detect_axes_origin_min_max(figure_url: HttpUrl):
    try:
        url = str(figure_url)
        image_bytes = get_image_from_url(url, False)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        main_image = image_cv.copy()

        # Updated algorithm
        intersection = crop_lower_intersection(image_cv)
        (
            morph_origin,
            horizontal_right_x,
            vertical_top_y,
            horizontal_left_x,
            vertical_bottom_y,
        ) = morphological_detection(image_cv)
        distance_origin = np.sqrt(
            (intersection[0] - morph_origin[0]) ** 2
            + (intersection[1] - morph_origin[1]) ** 2
        )
        if distance_origin > 10:
            intersection = morph_origin

        if intersection:
            box_size = 20
            top_left_x = max(0, intersection[0] - box_size)
            top_left_y = max(0, intersection[1] - box_size)
            bottom_right_x = min(image_cv.shape[1], intersection[0] + box_size)
            bottom_right_y = min(image_cv.shape[0], intersection[1] + box_size)
            bbox = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            x = bbox[0]
            y = bbox[1]
            width = bbox[2] - x
            height = bbox[3] - y

            result = origin_based_axes_detection(url, x, y, width, height)

            y_line = {"y_max": 0, "y_min": 0}  # (y_max, y_min)
            x_line = {"x_max": 0, "x_min": 0}  # (x_max, x_min)

            for point in result["points"]:
                if point["label"] == "ymax":
                    y = int(point["y"])
                    if y > vertical_top_y:
                        point["y"] = int(vertical_top_y)
                    y_line["y_max"] = point["y"]

                if point["label"] == "ymin":
                    y = int(point["y"])
                    if y < vertical_bottom_y:
                        point["y"] = int(vertical_bottom_y)
                    y_line["y_min"] = point["y"]

                if point["label"] == "xmax":
                    x = int(point["x"])
                    if x < horizontal_right_x:
                        point["x"] = int(horizontal_right_x)
                    x_line["x_max"] = point["x"]

                if point["label"] == "xmin":
                    x = int(point["x"])
                    if x > horizontal_left_x:
                        point["x"] = int(horizontal_left_x)
                    x_line["x_min"] = point["x"]

            diff_y = {"y_max": 0, "y_min": 0}
            diff_x = {"x_max": 0, "x_min": 0}

            for point in y_line:
                diff = abs(y_line[point] - intersection[1])
                diff_y[point] = diff

            for point in x_line:
                diff = abs(x_line[point] - intersection[0])
                diff_x[point] = diff

            # Adjust the x_max and y_max for the axes based on origin
            for point in result["points"]:
                if point["label"] == "ymax" and diff_y["y_max"] < diff_y["y_min"]:
                    point["y"] = int(intersection[1])

                if point["label"] == "ymin" and diff_y["y_min"] < diff_y["y_max"]:
                    point["y"] = int(intersection[1])

                if point["label"] == "xmax" and diff_x["x_max"] < diff_x["x_min"]:
                    point["x"] = int(intersection[0])

                if point["label"] == "xmin" and diff_x["x_min"] < diff_x["x_max"]:
                    point["x"] = int(intersection[0])

    except Exception:
        logger.exception("intersection detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="intersection detection failed",
        )

    # tick based min max correction
    try:
        result, _ = detect_x_y_ticks(
            main_image, copy.copy(result), copy.copy(intersection)
        )
        return result

    except Exception:
        # if failed return previous min max, no error
        return result


@plot_digitization_router.get("/plot-digitizer/match-template/iou")
async def match_template_iou(
    figure_url: str,
    rectangle_top_x: int | float,
    rectangle_top_y: int | float,
    rectangle_width: int | float,
    rectangle_height: int | float,
):
    """
    Matches a template image within a reference image using a the IoU method with the
    user selected marker's mask

    Args:
        figure_url (str): The URL of the reference image.
        rectangle_top_x (int | float): The x-coordinate of the top-left corner of the
            rectangle.
        rectangle_top_y (int | float): The y-coordinate of the top-left corner of the
            rectangle.
        rectangle_width (int | float): The width of the rectangle.
        rectangle_height (int | float): The height of the rectangle.
    Returns:
        dict: A dictionary containing the message and the extracted data.

    Raises:
        Any exceptions that may occur during image processing.
    """

    # Origin bounding box
    template_bounding_box = (
        int(rectangle_top_x),
        int(rectangle_top_y),
        int(rectangle_width),
        int(rectangle_height),
    )

    # Extract bounding box data using IoU
    axes_information, intersection = detect_axes_origin_min_max_local(
        figure_url, local_request=True
    )
    data = detect_bb_box_error_bar(
        figure_url, template_bounding_box, axes_information, intersection
    )

    return {"message": "successfully extracted data", "data": data}
