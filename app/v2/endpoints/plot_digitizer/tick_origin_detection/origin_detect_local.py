import cv2
import httpx
import numpy as np
from fastapi import HTTPException, status

from app.v2.endpoints.plot_digitizer.tick_origin_detection.services import (
    crop_lower_intersection,
    morphological_detection,
    origin_based_axes_detection,
)


def get_image_from_url(url: str, return_media_type: bool = False) -> bytes:
    try:
        response = httpx.get(url)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can not connect the URL. Please check the URL and try again.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch image",
        )

    image = response.content
    image_type = response.headers.get("content-type")

    if return_media_type:
        return image, image_type

    return image


def detect_axes_origin_min_max_local(figure_url, local_request=False):
    try:
        url = str(figure_url)
        image_bytes = get_image_from_url(url, False)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        # intersection = find_axes_intersection_advanced(image_cv)

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
                if (point["label"] == "ymax") and (diff_y["y_max"] < diff_y["y_min"]):
                    point["y"] = intersection[1]

                if (point["label"] == "ymin") and (diff_y["y_min"] < diff_y["y_max"]):
                    point["y"] = intersection[1]

                if (point["label"] == "xmax") and (diff_x["x_max"] < diff_x["x_min"]):
                    point["x"] = intersection[0]

                if (point["label"] == "xmin") and (diff_x["x_min"] < diff_x["x_max"]):
                    point["x"] = intersection[0]

            if local_request:
                return result, intersection
            else:
                return result

        else:
            return None

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="intersection detection failed",
        )
