import base64
from dataclasses import dataclass
from typing import Any

import cv2
import httpx
import numpy as np
from pydantic import BaseModel

from app.utils.image import get_image_from_url
from app.v3.endpoints.plot_digitizer.configs import error_bar_config, settings
from app.v3.endpoints.plot_digitizer.constants import ERROR_BAR_SUPPORTED_PLOTS
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.schemas import (
    DataPointsAutofil,
    Florence2OutputErrorBar,
)


@dataclass
class LineProcessingUtils:
    @staticmethod
    def cal_contour_height(contour: np.ndarray) -> int:
        """
        Return height of a contour
        """
        all_vertical_end = contour[:, 0, 1]
        start_vertical = min(all_vertical_end)
        end_vertical = max(all_vertical_end)
        length_vertical = abs(end_vertical - start_vertical)
        return length_vertical

    @staticmethod
    def cal_contour_length(contour: np.ndarray) -> int:
        """
        Return horizontal length of a contour
        """
        all_horizontal_end = contour[:, 0, 0]
        start_horizontal = min(all_horizontal_end)
        end_horizontal = max(all_horizontal_end)
        length_horizontal = abs(end_horizontal - start_horizontal)
        return length_horizontal

    @staticmethod
    def cal_contour_center(contour: np.ndarray) -> tuple[int, int]:
        """
        Calculate center of a given contour.
        """
        all_horizontal_end = contour[:, 0, 0]
        start_horizontal = min(all_horizontal_end)
        end_horizontal = max(all_horizontal_end)
        cX = (end_horizontal + start_horizontal) / 2

        all_vertical_end = contour[:, 0, 1]
        start_vertical = min(all_vertical_end)
        end_vertical = max(all_vertical_end)
        cY = (end_vertical + start_vertical) / 2

        return (round(cX), round(cY))


class ErrorBarResult(BaseModel):
    """Result for a single image inference."""

    x_main: float | None = None  # X coordinate of the main point (pixels)
    y_main: float | None = None  # Y coordinate of the main point (pixels)
    y1: float | None = None  # Top error bar position (pixels)
    y2: float | None = None  # Bottom error bar position (pixels)
    y1_normalized: int | None = None  # Top error bar (loc token value 0-999)
    y2_normalized: int | None = None  # Bottom error bar (loc token value 0-999)
    success: bool = True
    error: str | None = None


class ErrorBarDetection:
    def __init__(
        self, image_main: np.ndarray, ai_detected_data_points: list[tuple[int, int]]
    ):
        self.image_main = image_main
        self.ai_detected_data_points = ai_detected_data_points
        self.chart_dete_output = None

    def _is_equal(self, distance_1: int, distance_2: int, threshold: int) -> bool:
        """
        Check if two distances are equal
        """
        if abs(distance_1 - distance_2) < threshold:
            return True
        return False

    def _apply_morphology_operation(self, thresh_image: np.ndarray) -> np.ndarray:
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (error_bar_config.MORPHOLOGICAL_LENGTH, 1)
        )
        detect_horizontal = cv2.morphologyEx(
            thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
        )
        return detect_horizontal

    def _filter_error_bars_contour(
        self, contours: list[np.ndarray]
    ) -> list[np.ndarray]:
        # point are in (width, height) position fashion
        contours = contours[0] if len(contours) == 2 else contours[1]
        output_contour = []

        for contour in contours:
            length_contour = LineProcessingUtils.cal_contour_length(contour)
            height_contour = LineProcessingUtils.cal_contour_height(contour)

            if (
                length_contour < error_bar_config.FILTER_LENGTH
            ) and height_contour < error_bar_config.FILTER_HEIGHT:
                output_contour.append(contour)
        return output_contour

    def _extract_error_bars(self, thresh_image: np.ndarray) -> list[np.ndarray]:
        horizontal_lines = self._apply_morphology_operation(thresh_image)

        contours = cv2.findContours(
            horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            logger.info("No error bars detected")
            filtered_contours = []
        else:
            filtered_contours = self._filter_error_bars_contour(contours)
        return filtered_contours

    def _detect_error_bars_contours(self) -> list[np.ndarray]:
        """
        Detects all the error bars in a given image
        """
        gray = cv2.cvtColor(self.image_main, cv2.COLOR_BGR2GRAY)
        thresh_direct = np.where(gray < error_bar_config.GRAY_THRESHOLD, 1, 0).astype(
            "uint8"
        )

        blurred = cv2.GaussianBlur(gray, error_bar_config.GAUSSIAN_BLUR_KERNEL, 0)
        thresh_blur = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        ouput_error_bar_contour_1 = self._extract_error_bars(thresh_direct)
        ouput_error_bar_contour_2 = self._extract_error_bars(thresh_blur)

        error_bars_output = ouput_error_bar_contour_1 + ouput_error_bar_contour_2

        return error_bars_output

    def _get_error_bar_contour_centers(
        self, error_bars_contours: list[np.ndarray]
    ) -> list[tuple[int, int]]:
        return [
            LineProcessingUtils.cal_contour_center(contour)
            for contour in error_bars_contours
        ]

    def _is_error_in_range(
        self,
        error_bar_point: tuple[int, int],
        data_point: tuple[int, int],
        y_range: int,
    ) -> bool:
        is_in_x_range = (
            abs(error_bar_point[0] - data_point[0]) < error_bar_config.RANGE_X_VALUE
        )
        is_in_y_range = y_range < abs(error_bar_point[1] - data_point[1])
        in_range = is_in_x_range and is_in_y_range
        return in_range

    def _get_upper_lower_error_bars(
        self, error_bar_centers: list[tuple[int, int]], data_point: tuple[int, int]
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        upper_error_bars = []
        lower_error_bars = []

        for error_bar_point in error_bar_centers:
            for y_range in range(
                error_bar_config.RANGE_Y_HIGH_VALUE,
                error_bar_config.RANGE_Y_LOW_VALUE - 1,
                -1,
            ):
                if self._is_error_in_range(
                    error_bar_point, data_point, y_range=y_range
                ):
                    difference = error_bar_point[1] - data_point[1]
                    if difference < 0:
                        upper_error_bars.append(error_bar_point)
                    else:
                        lower_error_bars.append(error_bar_point)
                    break

        return upper_error_bars, lower_error_bars

    def _select_equal_distance_error_bars(
        self,
        upper_error_bars: list[tuple[int, int]],
        lower_error_bars: list[tuple[int, int]],
        data_point: tuple[int, int],
    ) -> list[dict]:
        selected_error_bars = []
        if len(upper_error_bars) != 0 and len(lower_error_bars) != 0:
            for point_top in upper_error_bars:
                topBarPixelDistance = abs(point_top[1] - data_point[1])
                for point_bottom in lower_error_bars:
                    bottomBarPixelDistance = abs(point_bottom[1] - data_point[1])
                    for threshold in range(
                        error_bar_config.THRESHOLD_LOW_VALUE,
                        error_bar_config.THRESHOLD_HIGH_VALUE,
                    ):
                        if self._is_equal(
                            topBarPixelDistance, bottomBarPixelDistance, threshold
                        ):
                            bar_pair = {
                                "topBarPixelDistance": topBarPixelDistance,
                                "bottomBarPixelDistance": bottomBarPixelDistance,
                                "distance": max(
                                    bottomBarPixelDistance, topBarPixelDistance
                                ),
                            }
                            selected_error_bars.append(bar_pair)
                            break
        return selected_error_bars

    def extract_error_bar_info(
        self, equal_distance_error_bars: list[dict], data_point: tuple[int, int]
    ) -> tuple[int, int, int, int, int]:
        selected_error_bars = sorted(
            equal_distance_error_bars, key=lambda x: x["distance"]
        )

        equal_distance_error_bar_pair = selected_error_bars[0]
        topBarPixelDistance = equal_distance_error_bar_pair["topBarPixelDistance"]
        bottomBarPixelDistance = equal_distance_error_bar_pair["bottomBarPixelDistance"]
        deviationPixelDistance = max(topBarPixelDistance, bottomBarPixelDistance)
        error_bar_data_point = data_point + (
            deviationPixelDistance,
            topBarPixelDistance,
            bottomBarPixelDistance,
        )
        return error_bar_data_point

    def _extract_closest_error_bar(
        self,
        upper_error_bars: list[tuple[int, int]],
        lower_error_bars: list[tuple[int, int]],
        data_point: tuple[int, int],
    ) -> tuple[int, int, int]:

        if len(upper_error_bars) != 0:
            sorted_upper_error_bars = sorted(upper_error_bars, key=lambda x: x[1])
            upper_error_point = sorted_upper_error_bars[0]
            topBarPixelDistance = abs(upper_error_point[1] - data_point[1])
            topBarPixelDistance = (
                0
                if topBarPixelDistance > error_bar_config.TOP_BAR_RANGE
                else topBarPixelDistance
            )
        else:
            topBarPixelDistance = 0

        if len(lower_error_bars) != 0:
            sorted_lower_error_bars = sorted(lower_error_bars, key=lambda x: x[1])
            lower_error_point = sorted_lower_error_bars[0]
            bottomBarPixelDistance = abs(lower_error_point[1] - data_point[1])
            bottomBarPixelDistance = (
                0
                if bottomBarPixelDistance > error_bar_config.BOTTOM_BAR_RANGE
                else bottomBarPixelDistance
            )
        else:
            bottomBarPixelDistance = 0

        deviationPixelDistance = max(topBarPixelDistance, bottomBarPixelDistance)
        error_bar_data_point = data_point + (
            deviationPixelDistance,
            topBarPixelDistance,
            bottomBarPixelDistance,
        )
        return error_bar_data_point

    def call_error_bar_api(
        self, images: list[dict], endpoint_url: str = settings.ERROR_BAR_ENDPOINT
    ):
        payload = {"images": images}

        response = httpx.post(
            endpoint_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(
                ErrorBarResult(
                    x_main=item.get("x_main"),
                    y_main=item.get("y_main"),
                    y1=item.get("y1"),
                    y2=item.get("y2"),
                    y1_normalized=item.get("y1_normalized"),
                    y2_normalized=item.get("y2_normalized"),
                    success=item.get("success", False),
                    error=item.get("error"),
                )
            )

        return results

    def detect_error_bars_data_points(self) -> list[tuple[int, int, int, int, int]]:
        error_bar_data_points_list = []

        # for each data point, detect the error bars
        image_slices = []
        for data_point in self.ai_detected_data_points:
            x = data_point[0]
            y = data_point[1]

            # Create slice from x-32 to x+32
            x_start = max(0, x - 32)
            x_end = min(self.image_main.shape[1], x + 32)

            # Extract the image slice
            image_slice = self.image_main[:, x_start:x_end].copy()

            # Convert to base64
            _, buffer = cv2.imencode(".png", image_slice)
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            # Create the dict
            slice_dict = {
                "image_base64": image_base64,
                "x": float(x),
                "y": float(y),
            }

            image_slices.append(slice_dict)

        detected_error_bars = self.call_error_bar_api(image_slices)
        logger.info(f"Detected error bars: {detected_error_bars}")

        for error_bar_result in detected_error_bars:
            x = error_bar_result.x_main
            y = error_bar_result.y_main
            topBarPixelDistance = abs(y - error_bar_result.y1)
            bottomBarPixelDistance = abs(y - error_bar_result.y2)
            deviationPixelDistance = max(topBarPixelDistance, bottomBarPixelDistance)

            error_bar_tuple = (
                x,
                y,
                deviationPixelDistance,
                topBarPixelDistance,
                bottomBarPixelDistance,
            )
            error_bar_data_points_list.append(error_bar_tuple)

        return error_bar_data_points_list


def add_error_bar(ai_detection_func) -> list[Florence2OutputErrorBar]:
    def error_bar_wrapper(*args, **kwargs):
        chart_type = kwargs.get("chart_type")
        image_main = args[0] if len(args) > 0 else kwargs.get("image_main")
        data_point_structured = ai_detection_func(*args, **kwargs)
        if chart_type in ERROR_BAR_SUPPORTED_PLOTS:
            logger.info("Running error bar detection.....")
            for data_point_info in data_point_structured:
                data_points = data_point_info["data_points"]
                error_bar_detection = ErrorBarDetection(
                    image_main=image_main, ai_detected_data_points=data_points
                )
                error_bar_data_points = (
                    error_bar_detection.detect_error_bars_data_points()
                )
                data_point_info["data_points"] = error_bar_data_points
        else:
            logger.info("Skipping error bar detection for Spider Plot")
        return data_point_structured

    return error_bar_wrapper


def convert_to_data_points_autofil(
    data_points: list[tuple[int, int, int, int, int]],
) -> list[DataPointsAutofil]:
    temp_points = []
    for point in data_points:
        temp_points.append(
            {
                "x": round(point[0]),
                "y": round(point[1]),
                "topBarPixelDistance": round(point[3]),
                "bottomBarPixelDistance": round(point[4]),
                "deviationPixelDistance": round(point[2]),
            }
        )
    return temp_points


def add_error_bar_autofill_response(
    autofill_response: Any,
    image_url: str,
) -> Any:
    logger.info("Running error bar detection for rescaled images.")
    image_byte = get_image_from_url(image_url)
    image_main = np.frombuffer(image_byte, np.uint8)
    image_main = cv2.imdecode(image_main, cv2.IMREAD_COLOR)
    image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)

    for line_data in autofill_response["data"]["lines"]:
        ai_detected_data_points = []
        for point in line_data["points"]:
            x = point["x"]
            y = point["y"]
            ai_detected_data_points.append((x, y))
        error_bar_detection = ErrorBarDetection(
            image_main=image_main.copy(),
            ai_detected_data_points=ai_detected_data_points,
        )
        error_bar_data_points = error_bar_detection.detect_error_bars_data_points()
        autofill_data_points = convert_to_data_points_autofil(error_bar_data_points)
        line_data["points"] = autofill_data_points

    manual_mapped_lines = []
    for line_data in autofill_response["data"]["detected_line_points"]:
        ai_detected_data_points = []
        for point in line_data:
            x = point["x"]
            y = point["y"]
            ai_detected_data_points.append((x, y))
        error_bar_detection = ErrorBarDetection(
            image_main=image_main.copy(),
            ai_detected_data_points=ai_detected_data_points,
        )
        error_bar_data_points = error_bar_detection.detect_error_bars_data_points()
        autofill_data_points = convert_to_data_points_autofil(error_bar_data_points)
        manual_mapped_lines.append(autofill_data_points)
    autofill_response["data"]["detected_line_points"] = manual_mapped_lines

    return autofill_response
