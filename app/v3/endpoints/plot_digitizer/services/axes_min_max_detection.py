from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from scipy import stats

from app.utils.image import async_get_image_from_url
from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.constants import (
    MAX_RETRIES,
    AxisMinMax,
    ChartDete,
)
from app.v3.endpoints.plot_digitizer.logging import logger
from app.v3.endpoints.plot_digitizer.schemas import AxesMinMaxOutput
from app.v3.endpoints.plot_digitizer.utils import (
    numpy_to_string_binary,
    post_request_image_aync,
)


@dataclass
class ProcessLine:
    @staticmethod
    def filter_largest_vertical_line(
        points: list[tuple[int, int]], t: int = 30
    ) -> list[tuple[int, int]]:
        if not points:
            return []

        points = sorted(points, key=lambda p: p[0])

        clusters = []
        for x, y in points:
            added = False
            for cluster_x, cluster_points in clusters:
                if abs(x - cluster_x) <= t:
                    cluster_points.append((x, y))
                    added = True
                    break
            if not added:
                clusters.append((x, [(x, y)]))

        largest_cluster = max(clusters, key=lambda c: len(c[1]))

        return largest_cluster[1]

    @staticmethod
    def find_largest_horizontal_line(
        points: list[list[tuple[int, int]]],
        axis_constant: int,
        y1: int,
        t: int = 30,
    ) -> list[tuple[int, int]]:
        points = sorted(points, key=lambda p: p[1])

        clusters = []
        for x, y in points:
            if abs(y + y1 - axis_constant) > 50:
                continue
            added = False
            for cluster_y, cluster_points in clusters:
                if abs(y - cluster_y) <= t:
                    cluster_points.append((x, y))
                    added = True
                    break
            if not added:
                clusters.append((y, [(x, y)]))
        if len(clusters) > 0:
            largest_cluster = max(clusters, key=lambda c: len(c[1]))
            return largest_cluster[1]
        else:
            return []

    @staticmethod
    def cal_contour_center(contour: np.ndarray) -> tuple[int, int]:
        all_horizontal_end = contour[:, 0, 0]
        start_horizontal = min(all_horizontal_end)
        end_horizontal = max(all_horizontal_end)
        cX = (end_horizontal + start_horizontal) / 2

        all_horizontal_end = contour[:, 0, 1]
        start_horizontal = min(all_horizontal_end)
        end_horizontal = max(all_horizontal_end)
        cY = (end_horizontal + start_horizontal) / 2

        return (round(cX), round(cY))


class Plot:
    def __init__(
        self,
        img_url: str,
        image_main: np.ndarray,
        chart_dete_out: dict,
        scale_percent: float,
    ):
        self.img_url = img_url
        self.image_main = image_main
        self.height, self.width, _ = self.image_main.shape
        self.scale_percent = scale_percent
        self.chart_dete_out = chart_dete_out
        self.extracted_boxes = self._extract_bboxes(chart_dete_out)
        self._eliminate_extra_area()

    @classmethod
    async def create(cls, img_url: str) -> "Plot":
        image_main = await cls._get_image_main(img_url)
        scale_percent, image_main = cls._calculate_scale_percent(image_main=image_main)
        chart_dete_out = await cls._execute_chart_dete(image_main)
        return cls(img_url, image_main, chart_dete_out, scale_percent)

    @staticmethod
    async def _execute_chart_dete(image_main: np.ndarray) -> Any:
        headers, files = numpy_to_string_binary(image_main.copy())
        chart_dete_out = None
        try_count = 0
        while chart_dete_out is None:
            chart_dete_out = await post_request_image_aync(
                post_url=settings.CHART_DETE_API, headers=headers, files=files
            )
            try_count += 1
            if try_count == MAX_RETRIES:
                logger.error("ChartDete Failed to extract legend patch and label")
        return chart_dete_out

    @staticmethod
    async def _get_image_main(img_url: str) -> np.ndarray:
        image_byte = await async_get_image_from_url(img_url)
        image_main = np.frombuffer(image_byte, np.uint8)
        image_main = cv2.imdecode(image_main, cv2.IMREAD_COLOR)
        image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)
        return image_main

    @staticmethod
    def _calculate_scale_percent(
        image_main: np.ndarray,
    ) -> tuple[int | None, np.ndarray]:
        height, width, _ = image_main.shape
        scale_percent = 1
        if height > 800 or width > 800:
            scale_percent = 800 / max(height, width)
            new_width = round(width * scale_percent)
            new_height = round(height * scale_percent)
            image_main = cv2.resize(
                image_main, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            height, width, _ = image_main.shape
        return scale_percent, image_main

    def _extract_bboxes(self, chart_dete_out: Any) -> dict:
        extraction_list = ["xlabel", "ylabel", "plot_area"]
        criteria_map = {"xlabel": [], "ylabel": [], "plot_area": []}

        for criteria in extraction_list:
            bb_boxes = chart_dete_out.get(criteria)
            if bb_boxes is None:
                continue
            for bb_box in bb_boxes:
                x1 = round(bb_box["x1"])
                y1 = round(bb_box["y1"])
                x2 = round(bb_box["x2"])
                y2 = round(bb_box["y2"])
                score = bb_box["score"]
                if score > ChartDete.AXES_DETECTION_THERESHOLD:
                    if criteria == "xlabel":
                        center = ((x1 + x2) / 2, y1)
                        append_item = (round(center[0]), round(center[1]))
                    elif criteria == "ylabel":
                        center = (x2, (y1 + y2) / 2)
                        append_item = (round(center[0]), round(center[1]))
                    else:
                        append_item = (x1, y1, x2, y2)
                    criteria_map[criteria].append(append_item)

        criteria_map["ylabel"] = ProcessLine.filter_largest_vertical_line(
            criteria_map["ylabel"]
        )
        return criteria_map

    def _eliminate_extra_area(self):
        plot_area = self.extracted_boxes["plot_area"]
        x_label = self.extracted_boxes["xlabel"]

        if len(plot_area) > 0 and len(x_label) > 0:
            plot_area_lower_y = plot_area[0][3]
            x_label_center_y = [x_label_point[1] for x_label_point in x_label]
            lower_y_value = round(max(plot_area_lower_y, max(x_label_center_y)))
            self.image_main = self.image_main[:lower_y_value, :].copy()

    @staticmethod
    def closest(
        plot_area_bbox: tuple[int, int, int, int] | None,
        value: int,
        axis_constant: int | None,
        axis: Literal["x", "y"],
    ) -> int:
        if axis_constant is not None:
            return axis_constant
        if plot_area_bbox is None:
            return value
        if axis == "y":
            a, _, b, _ = plot_area_bbox
        else:
            _, a, _, b = plot_area_bbox
        closest_distance = min(abs(a - value), abs(b - value))
        if axis == "x" and closest_distance > AxisMinMax.plot_area_max_close_distance:
            return max(0, (value - 5))
        return a if abs(a - value) < abs(b - value) else b


class Axis:
    def __init__(
        self,
        plot: Plot,
        white_out_sections: dict,
        axis_type: Literal["x", "y"],
        axis_tick_kernel: tuple[int, int],
    ):
        self.plot = plot
        self.image_main = plot.image_main
        self.height, self.width, _ = self.image_main.shape
        self.chart_dete_out = plot.chart_dete_out
        self.axis_type = axis_type
        if self.axis_type == "y":
            kernel_shape_percentage = AxisMinMax.Y_AXIS_KERNEL_SHAPE_PERCENT
            self.axis_line_kernel = (1, round(self.height * kernel_shape_percentage))
        else:
            kernel_shape_percentage = AxisMinMax.X_AXIS_KERNEL_SHAPE_PERCENT
            self.axis_line_kernel = (round(self.width * kernel_shape_percentage), 1)
        axes_area_name = f"{self.axis_type}_axis_area"
        self.axis_area = (
            self.chart_dete_out.get(axes_area_name, [])
            if self.chart_dete_out is not None
            else []
        )
        self.image_white_out = self._do_white_out_sections(
            white_out_sections=white_out_sections
        )
        self.axis_area_bboxes = self._get_valid_bboxes()
        self.x1, self.y1, self.x2, self.y2 = self._get_axis_boundaries()
        self.process_image = self._extract_process_image()
        self.gray = self._convert_to_grayscale()
        self.thresh_image = self._create_threshold_image()
        self.axis_tick_kernel_start = axis_tick_kernel
        self.axis_line_contour = None
        self.axis_constant_corrdinates = self._get_axis_constant_corrdinate()
        self.thresh_image_tick = self._fill_contour_white()

    def _get_valid_bboxes(self) -> list[tuple]:
        return (
            [
                (round(box["x1"]), round(box["y1"]), round(box["x2"]), round(box["y2"]))
                for box in self.axis_area
                if box.get("score", 0) > 0.3
            ]
            if self.axis_area
            else []
        )

    def _get_axis_boundaries(self) -> tuple:
        if len(self.axis_area_bboxes) > 0:
            return self.axis_area_bboxes[0]
        return (0, 0, self.image_main.shape[1], self.image_main.shape[0])

    def _extract_process_image(self) -> np.ndarray:
        return self.image_white_out.copy()[self.y1 : self.y2, self.x1 : self.x2]

    def _convert_to_grayscale(self) -> np.ndarray:
        return cv2.cvtColor(self.process_image, cv2.COLOR_RGB2GRAY)

    def _create_threshold_image(self) -> np.ndarray:
        return np.where(self.gray < AxisMinMax.GRAY_FILTER_LEVEL, 1, 0).astype("uint8")

    def _fill_contour_white(self) -> np.ndarray | None:
        if self.axis_line_contour is not None:
            return cv2.drawContours(
                self.thresh_image.copy(),
                [self.axis_line_contour],
                contourIdx=-1,
                color=(0, 0, 0),
                thickness=-1,
            )

        else:
            return None

    def _is_inverse_axis(self) -> dict:
        terminal_points = self.axis_constant_corrdinates["terminal_points"]
        plot_area = self.plot.extracted_boxes["plot_area"]
        if terminal_points and plot_area:
            start_point = terminal_points[0]
            end_point = terminal_points[1]
            y1, y2 = start_point[1], end_point[1]
            avg_axis_mid_elevation = round((y1 + y2) / 2)
            _, ytop, _, ybottom = plot_area[0]
            y_plot_area_center = (ytop + ybottom) / 2
            if y_plot_area_center < avg_axis_mid_elevation:
                return {
                    "is_inverse": False,
                    "avg_axis_mid_elevation": avg_axis_mid_elevation,
                }

            else:
                return {
                    "is_inverse": True,
                    "avg_axis_mid_elevation": avg_axis_mid_elevation,
                }

    def _detect_contour(
        self, kernel_shape: tuple[int, int], tick_detection: bool = False
    ) -> tuple[np.ndarray]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
        threshold_image = (
            self.thresh_image_tick if tick_detection else self.thresh_image
        )
        detected_lines = cv2.morphologyEx(
            threshold_image.copy(), cv2.MORPH_OPEN, kernel, iterations=1
        )
        contours = cv2.findContours(
            detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return contours[0] if len(contours) == 2 else contours[1]

    @staticmethod
    def _filter_contour_length(contour):
        if contour is not None and len(contour) > 1:
            contour = [max(contour, key=len)]
        return contour

    def _get_axis_constant_corrdinate(self) -> dict:
        contour = self._detect_contour(self.axis_line_kernel)
        if len(contour) == 0:
            return {
                "axis_constant_cordinate": None,
                "terminal_points": None,
            }
        contour = self._filter_contour_length(contour)
        self.axis_line_contour = contour[0]
        axis_index = 1 if self.axis_type == "x" else 0
        axis_constant_cordinate = round(stats.mode(contour[0][:, 0, axis_index])[0])
        contour_squeeze = np.squeeze(contour)

        def extract_terminal_point(contour_squeeze: np.ndarray, index: int) -> list:
            sorted_contour = contour_squeeze[contour_squeeze[:, index].argsort()]
            return (
                sorted_contour[[0, -1]].tolist() if len(sorted_contour) >= 2 else None
            )

        axis_constant_cordinate += self.y1 if self.axis_type == "x" else self.x1
        terminal_points = extract_terminal_point(
            contour_squeeze, index=0 if self.axis_type == "x" else 1
        )
        terminal_points = (
            [(point[0] + self.x1, point[1] + self.y1) for point in terminal_points]
            if terminal_points
            else None
        )

        if contour_squeeze.size == 0:
            axis_mid_point = None
        else:
            axis_mid_point = np.mean(contour_squeeze, axis=0).tolist()
            axis_mid_point = (axis_mid_point[0] + self.x1, axis_mid_point[1] + self.y1)

        return {
            "axis_constant_cordinate": axis_constant_cordinate,
            "terminal_points": terminal_points,
            "axis_mid_point": axis_mid_point,
        }

    def shift_point_to_new_origin(self, point: tuple[int, int] | np.ndarray) -> list:
        new_origin = np.array((self.x1, self.y1))
        shifted_point = np.array(point) + new_origin
        return shifted_point.tolist()

    def detect_axis_ticks(self, axis_constant) -> list | None:
        kernel_length_list = list(range(self.axis_tick_kernel_start, 2, -1))
        contours = []
        for kernel_length in kernel_length_list:
            kernel_shape = (
                (1, kernel_length) if self.axis_type == "x" else (kernel_length, 1)
            )
            contour = self._detect_contour(kernel_shape, tick_detection=True)
            contours.append(contour)
        if len(contours) == 0:
            return None

        list_ticks_centroids = []
        for contour in contours:
            ticks_centroids = [ProcessLine.cal_contour_center(c) for c in contour]
            list_ticks_centroids.append(ticks_centroids)

        list_largest_line_points = []
        for ticks_centroids in list_ticks_centroids:
            largest_line_points: list = (
                ProcessLine.find_largest_horizontal_line(
                    ticks_centroids, axis_constant, self.y1
                )
                if self.axis_type == "x"
                else ProcessLine.filter_largest_vertical_line(ticks_centroids)
            )
            largest_line_points = (
                None if len(largest_line_points) < 1 else largest_line_points
            )
            if largest_line_points is not None:
                index = 0 if self.axis_type == "x" else 1
                largest_line_points = sorted(
                    largest_line_points, key=lambda point: point[index]
                )
                list_largest_line_points.append(largest_line_points)
        lengths = [len(line) for line in list_largest_line_points]
        print(lengths)
        return max(list_largest_line_points, key=len, default=[])

    def _do_white_out_sections(self, white_out_sections: list) -> np.ndarray:
        image_annotate = self.image_main.copy()
        for plot_section in white_out_sections:
            if plot_section in self.chart_dete_out:
                for box in self.chart_dete_out[plot_section]:
                    if box["score"] > ChartDete.AXES_DETECTION_THERESHOLD:
                        x1, y1, x2, y2 = map(
                            int, (box["x1"], box["y1"], box["x2"], box["y2"])
                        )
                        cv2.rectangle(
                            image_annotate, (x1, y1), (x2, y2), (255, 255, 255), -1
                        )
        return image_annotate


class YAxis(Axis):
    def __init__(self, plot: Plot):
        super().__init__(
            plot=plot,
            white_out_sections=AxisMinMax.Y_AXIS_WHITEOUT_SECTION,
            axis_type="y",
            axis_tick_kernel=AxisMinMax.Y_AXIS_TICK_KERNEL_START,
        )

    def _detect_y_axis_ticks_min_max(self) -> dict:
        y_axis_constant = self.axis_constant_corrdinates["axis_constant_cordinate"]
        axis_ticks = self.detect_axis_ticks(y_axis_constant)
        if axis_ticks is not None and len(axis_ticks) >= 2:
            ymax = (y_axis_constant, axis_ticks[0][1])
            ymin = (y_axis_constant, axis_ticks[-1][1])
            ymax = self.shift_point_to_new_origin(ymax)
            ymin = self.shift_point_to_new_origin(ymin)
            return ymax, ymin
        return None

    def get_y_axis_min_max(self) -> tuple[int, int]:
        ylabel = self.plot.extracted_boxes["ylabel"]
        plot_area = self.plot.extracted_boxes["plot_area"]
        plot_area_bb_box = plot_area[0] if len(plot_area) > 0 else None

        y_axis_constant_value = self.axis_constant_corrdinates
        y_axis_constant = y_axis_constant_value["axis_constant_cordinate"]
        y_axis_terminal_points = y_axis_constant_value["terminal_points"]

        if len(ylabel) >= 2:
            ymin_temp = max(ylabel, key=lambda point: point[1])
            ymax_temp = min(ylabel, key=lambda point: point[1])
        else:
            y_ticks = self._detect_y_axis_ticks_min_max()
            if y_ticks is not None:
                ymax_temp, ymin_temp = y_ticks
            elif y_axis_terminal_points is not None:
                ymax_temp = y_axis_terminal_points[0]
                ymin_temp = y_axis_terminal_points[1]
            else:
                ymax_temp = (round(self.width * 0.25), round(self.height * 0.25))
                ymin_temp = (round(self.width * 0.25), round(self.height * 0.75))

        ymin = (
            self.plot.closest(
                plot_area_bb_box, ymin_temp[0], y_axis_constant, axis="y"
            ),
            ymin_temp[1],
        )
        ymax = (
            self.plot.closest(
                plot_area_bb_box, ymax_temp[0], y_axis_constant, axis="y"
            ),
            ymax_temp[1],
        )

        return (ymin, ymax)


class XAxis(Axis):
    def __init__(self, plot: Plot):

        super().__init__(
            plot=plot,
            white_out_sections=AxisMinMax.X_AXIS_WHITEOUT_SECTION,
            axis_type="x",
            axis_tick_kernel=AxisMinMax.X_AXIS_TICK_KERNEL_START,
        )
        if len(self.axis_area_bboxes) == 0:
            self._update_x_axis_area()
        self._clear_axis_to_plot_area()

    def _clear_axis_to_plot_area(self):
        check_inverse_axis = self._is_inverse_axis()
        is_inverse = check_inverse_axis["is_inverse"]
        avg_axis_mid_elevation = check_inverse_axis["avg_axis_mid_elevation"]
        y_clear_value = abs(round(avg_axis_mid_elevation - self.y1))
        if is_inverse:
            self.thresh_image_tick[y_clear_value:, :] = 0
        else:
            self.thresh_image_tick[:y_clear_value, :] = 0

    def _update_x_axis_area(self):
        terminal_points = self.axis_constant_corrdinates["terminal_points"]
        if terminal_points:
            start_point = terminal_points[0]
            end_point = terminal_points[1]
            x1 = start_point[0]
            y1 = max(0, start_point[1] - AxisMinMax.X_AXIS_AREA_WIDTH)
            x2 = end_point[0]
            y2 = min(
                self.image_main.shape[0], start_point[1] + AxisMinMax.X_AXIS_AREA_WIDTH
            )
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.process_image = self._extract_process_image()
        self.gray = self._convert_to_grayscale()
        self.thresh_image = self._create_threshold_image()
        self.axis_constant_corrdinates = self._get_axis_constant_corrdinate()
        self.thresh_image_tick = self._fill_contour_white()

    def _detect_x_axis_ticks_min_max(self) -> dict:
        x_axis_constant = self.axis_constant_corrdinates["axis_constant_cordinate"]
        axis_ticks = self.detect_axis_ticks(x_axis_constant)

        check_inverse_axis = self._is_inverse_axis()
        is_inverse = check_inverse_axis["is_inverse"]
        axis_mid_point = self.axis_constant_corrdinates["axis_mid_point"]
        threshold_value = abs(round(axis_mid_point[1] - self.y1))

        if axis_ticks is not None and len(axis_ticks) >= 2:
            xmax = (axis_ticks[-1][0], x_axis_constant)
            min_index = 0
            if axis_mid_point:
                while True:
                    value = axis_ticks[min_index][1]
                    condition = (
                        value >= threshold_value
                        if is_inverse
                        else value <= threshold_value
                    )
                    if condition:
                        min_index += 1
                    else:
                        break
                    if min_index == len(axis_ticks) - 1:
                        min_index = 0
                        break
            xmin = (axis_ticks[min_index][0], x_axis_constant)
            xmax = self.shift_point_to_new_origin(xmax)
            xmin = self.shift_point_to_new_origin(xmin)
            return xmax, xmin
        return None


async def chart_dete_min_max(img_url: str) -> AxesMinMaxOutput:
    plot = await Plot.create(img_url=img_url)
    height, width = plot.height, plot.width
    scale_percent = plot.scale_percent
    extracted_boxes = plot.extracted_boxes
    xlabel, plot_area = (
        extracted_boxes["xlabel"],
        extracted_boxes["plot_area"],
    )

    x_axis = XAxis(plot=plot)
    y_axis = YAxis(plot=plot)

    x_axis_constant_value = x_axis.axis_constant_corrdinates
    x_axis_constant = x_axis_constant_value["axis_constant_cordinate"]
    x_axis_terminal_points = x_axis_constant_value["terminal_points"]

    y_axis_constant_value = y_axis.axis_constant_corrdinates
    y_axis_constant = y_axis_constant_value["axis_constant_cordinate"]

    plot_area_bb_box = plot_area[0] if len(plot_area) > 0 else None

    ymin, ymax = y_axis.get_y_axis_min_max()

    x_ticks = x_axis._detect_x_axis_ticks_min_max()
    if x_ticks:
        xmax_temp, xmin_temp = x_ticks
    elif len(xlabel) >= 2:
        xmax_temp = max(xlabel, key=lambda point: point[0])
        xmin_temp = min(xlabel, key=lambda point: point[0])
    elif x_axis_terminal_points is not None:
        xmax_temp = x_axis_terminal_points[1]
        if y_axis_constant is not None:
            xmin_temp = (y_axis_constant, x_axis_terminal_points[0][1])
        else:
            xmin_temp = x_axis_terminal_points[0]
    else:
        xmax_temp = (round(width * 0.75), round(height * 0.75))
        xmin_temp = (round(width * 0.25), round(height * 0.75))

    xmin = (
        xmin_temp[0],
        Plot.closest(plot_area_bb_box, xmin_temp[1], x_axis_constant, axis="x"),
    )
    xmax = (
        xmax_temp[0],
        Plot.closest(plot_area_bb_box, xmax_temp[1], x_axis_constant, axis="x"),
    )

    axes_detection_out = AxesMinMaxOutput.from_tuples(
        ymin, ymax, xmin, xmax, scale_percent
    )

    return axes_detection_out.model_dump()
