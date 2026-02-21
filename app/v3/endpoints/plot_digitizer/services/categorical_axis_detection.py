from dataclasses import dataclass, field

import numpy as np

from app.v3.endpoints.plot_digitizer.constants import CategoricalAxis
from app.v3.endpoints.plot_digitizer.helpers import PlotDigitizerHelper
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger


@dataclass
class XaxisCategoricalLabel:
    x1: int | None = field(default=0)
    y1: int | None = field(default=0)
    x2: int | None = field(default=0)
    y2: int | None = field(default=0)
    label_text: str | None = field(default="")

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            x1=d.get("x1", 0),
            y1=d.get("y1", 0),
            x2=d.get("x2", 0),
            y2=d.get("y2", 0),
            label_text=d.get("label_text", ""),
        )


class CategoricalAxisDetection:
    def __init__(self, response: dict, image_url: str):
        self.response = response
        self.image_url = image_url
        self.helper = PlotDigitizerHelper(image_url=self.image_url)
        self.chart_dete_output = self.helper.chart_dete_out
        self.image_main: np.ndarray | None = self.helper.image_main
        self.x_axis_categorical_list = self._detect_categorical_x_labels()

    @staticmethod
    def _get_single_point(point: dict, point_name: str) -> int | None:
        if point.get(point_name):
            if point_name == "x1" or point_name == "y1":
                multiplier = -1
            else:
                multiplier = 1
            return (
                round(point[point_name])
                + CategoricalAxis.X_AXIS_LABEL_PADDING * multiplier
            )
        else:
            logger.exception(
                f"Point {point_name} not found in point: {point} for Categorical Axis Detection."  # noqa: E501
            )
            return None

    def _detect_categorical_x_labels(self) -> list[XaxisCategoricalLabel]:
        categorical_label_list = []

        if not self.chart_dete_output.get("xlabel"):
            logger.info("Chart Dete failed in categorical axis detection.")
            return []

        x_label_list = (
            self.response.get("data", {})
            .get("plot_axis_data", {})
            .get("x_tick_values", [])
        )

        if x_label_list and len(x_label_list) <= len(
            self.chart_dete_output.get("xlabel")
        ):
            logger.info(
                f"Labels found in response for Categorical Axis Detection: {x_label_list}."  # noqa: E501
            )
            total_labels = len(x_label_list)

            sorted_selected_xlabel = sorted(
                self.chart_dete_output.get("xlabel"),
                key=lambda x: x.get("score", 0),
                reverse=True,
            )[:total_labels]
        else:
            logger.info("Labels not found in response for Categorical Axis Detection.")
            sorted_selected_xlabel = []
            for x_label in self.chart_dete_output.get("xlabel"):
                if not x_label.get("score"):
                    logger.info(
                        "Score not found in xlabel for Categorical Axis Detection."
                    )
                    continue

                if x_label.get("score") > CategoricalAxis.SCORE_THRESHOLD:
                    sorted_selected_xlabel.append(x_label)

        for output in sorted_selected_xlabel:
            x1 = self._get_single_point(output, "x1")
            y1 = self._get_single_point(output, "y1")
            x2 = self._get_single_point(output, "x2")
            y2 = self._get_single_point(output, "y2")

            cropped_part = self.image_main[y1:y2, x1:x2].copy()
            label_text = PlotDigitizerHelper.get_ocr_output(cropped_part)

            categorical_label = XaxisCategoricalLabel.from_dict(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label_text": label_text}
            )
            categorical_label_list.append(categorical_label)

        return categorical_label_list

    @staticmethod
    def format_categorical_point(autofill_response):
        all_lines = autofill_response["data"]["detected_line_points"]

        for line in all_lines:
            for point in line:
                point["x_cat"] = None
                point["y_cat"] = None

        all_lines = autofill_response["data"]["lines"]

        for line in all_lines:
            for point in line["points"]:
                point["x_cat"] = None
                point["y_cat"] = None

        return autofill_response

    def _map_all_lines(self):
        all_lines = self.response["data"]["detected_line_points"]

        for categorical_label in self.x_axis_categorical_list:
            if (
                not categorical_label.label_text
                or not categorical_label.x1
                or not categorical_label.x2
            ):
                logger.info(
                    "Skipped one of the categorical labels. No label text or x1 or x2 found."  # noqa: E501
                )
                continue

            x1 = categorical_label.x1
            x2 = categorical_label.x2

            for line in all_lines:
                for point in line:
                    x = point["x"]
                    if x1 <= x <= x2:
                        point["x_cat"] = categorical_label.label_text

    def _map_mapped_lines(self):
        all_lines = self.response["data"]["lines"]

        for categorical_label in self.x_axis_categorical_list:
            if (
                not categorical_label.label_text
                or not categorical_label.x1
                or not categorical_label.x2
            ):
                logger.info(
                    "Skipped one of the categorical labels. No label text or x1 or x2 found."  # noqa: E501
                )
                continue

            x1 = categorical_label.x1
            x2 = categorical_label.x2

            for line in all_lines:
                for point in line["points"]:
                    x = point["x"]
                    if x1 <= x <= x2:
                        point["x_cat"] = categorical_label.label_text

    def map_categorical_labels_to_points(self):
        logger.info("Categorical axis detection started.")
        if not self.x_axis_categorical_list:
            logger.exception(
                "Categorical axis detection failed. No categorical labels detected."
            )
            return self.response

        self._map_all_lines()
        self._map_mapped_lines()
        logger.info(
            f"Categorical axis detection completed. {len(self.x_axis_categorical_list)} labels detected."  # noqa: E501
        )

        return self.response
