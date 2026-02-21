from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.utils.image import get_image_from_url
from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.prompts import VISION_AGENT_BAR_PLOT_PROMPT
from app.v3.endpoints.plot_digitizer.utils import post_request


def calculate_attention_info(image, attn_scores, n_width, n_height):
    w, h = image.size
    scores = np.array(attn_scores[0]).reshape(n_height, n_width)

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

    high_attention_threshold = 0.5
    high_attention_mask = scores_norm >= high_attention_threshold

    high_attention_indices = np.argwhere(high_attention_mask)

    if len(high_attention_indices) > 0:
        min_y, min_x = high_attention_indices.min(axis=0)
        max_y, max_x = high_attention_indices.max(axis=0)

        x1 = int(min_x * w / n_width)
        y1 = int(min_y * h / n_height)
        x2 = int(max_x * w / n_width)
        y2 = int(max_y * h / n_height)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        bounding_box = (x1, y1, x2, y2)

        bbox_min_x = int(min_x)
        bbox_max_x = int(max_x)
        bbox_min_y = int(min_y)
        bbox_max_y = int(max_y)

        bbox_region = scores_norm[
            bbox_min_y : bbox_max_y + 1, bbox_min_x : bbox_max_x + 1
        ]
        cumulative_attention = np.sum(bbox_region)

        # attention centroid
        y_indices, x_indices = np.meshgrid(
            np.arange(n_height), np.arange(n_width), indexing="ij"
        )
        total_attention = np.sum(scores_norm)

        if total_attention > 0:
            centroid_y = np.sum(y_indices * scores_norm) / total_attention
            centroid_x = np.sum(x_indices * scores_norm) / total_attention

            centroid_x_img = int(centroid_x * w / n_width)
            centroid_y_img = int(centroid_y * h / n_height)

            attention_centroid = (centroid_x_img, centroid_y_img)
        else:
            attention_centroid = None

    else:
        bounding_box = None
        cumulative_attention = 0.0
        attention_centroid = None

    return bounding_box, cumulative_attention, attention_centroid


def get_attn_map_fn(image, pred):
    n_width, n_height = pred["n_width"], pred["n_height"]
    attn_scores = pred["attn_scores"]
    bounding_box, cumulative_attention, attention_centroid = calculate_attention_info(
        image, attn_scores, n_width, n_height
    )
    return bounding_box, cumulative_attention, attention_centroid


class BarPlotDigitizer:
    def __init__(self, autofill_response: Any, image_url: str):
        self.image_url = image_url
        self.image_pil, self.image_main = self.load_image(image_url)
        self.autofill_response = autofill_response
        self.is_autofill_line_empty = False
        self.autofill_legends = self._get_autofill_legends()

    @staticmethod
    def load_image(image_url: str) -> Image.Image:
        image_io = get_image_from_url(image_url)
        image_bytes = np.frombuffer(image_io, np.uint8)
        image_array = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        image = Image.fromarray(image_array)
        return image, image_array

    def format_autofill_response(self, return_output=False):
        self.autofill_response["data"]["detected_line_points"] = []
        list_line = self.autofill_response["data"]["lines"]
        if list_line:
            for data in list_line:
                data["points"] = []
        if return_output:
            return self.autofill_response

    def _get_autofill_legends(self) -> list[str]:
        self.format_autofill_response()
        autofill_legends = []
        list_line = self.autofill_response["data"]["lines"]
        if list_line:
            for line in list_line:
                autofill_legends.append(line["labels"]["line_name"])
        elif self.autofill_response["data"]["legends"]:
            self.is_autofill_line_empty = True
            autofill_legends = self.autofill_response["data"]["legends"]
        return autofill_legends

    def get_vision_agent_out(self):
        x_tick_values = None
        if (
            self.autofill_response
            and self.autofill_response.get("data")
            and self.autofill_response.get("data").get("plot_axis_data")
        ):
            x_tick_values = (
                self.autofill_response.get("data")
                .get("plot_axis_data")
                .get("x_tick_values")
            )
        if self.is_autofill_line_empty or not x_tick_values:
            return self.autofill_response

        x_tick_only = False
        unique_legends = set(self.autofill_legends)
        unique_x_tick_values = set(x_tick_values)
        if len(unique_legends) == len(unique_x_tick_values):
            x_tick_only = True

        input_instructions = []
        legend_map = []
        for idx_legend, legend in enumerate(self.autofill_legends):
            for idx_x_tick_values, x_tick_value in enumerate(x_tick_values):
                if x_tick_only:
                    bar_info = f"{legend}"
                    x_tick_value = legend
                else:
                    bar_info = f"{legend}-{x_tick_value}"
                instruction_id = f"{idx_legend}_{idx_x_tick_values}"
                input_instructions.append(
                    {
                        "input_instruction": VISION_AGENT_BAR_PLOT_PROMPT.format(
                            BAR_INFO=bar_info
                        )
                        .replace("\n", "")
                        .strip(),
                        "instruction_id": instruction_id,
                    }
                )
                legend_map.append(
                    {
                        "legend": legend,
                        "x_tick_value": x_tick_value,
                        "instruction_id": instruction_id,
                    }
                )
                if x_tick_only:
                    break
        self.legend_map = legend_map

        vision_agent_payload = {
            "image_url": self.image_url,
            "input_instructions": input_instructions,
        }

        vision_agent_out = post_request(
            post_url=settings.VISION_AGENT_ENDPOINT, payload=vision_agent_payload
        )

        logger.info("Completed vision agent inference")

        return vision_agent_out

    def _get_attention_info(self):
        attention_info = []
        vision_agent_out = self.get_vision_agent_out()
        if not vision_agent_out:
            return attention_info
        for pred in vision_agent_out["output"]:
            instruction_id = pred["instruction_id"]
            bbox, cumulative_attention, attention_centroid = get_attn_map_fn(
                self.image_pil, pred["pred_info"]
            )

            attention_info.append(
                {
                    "instruction_id": instruction_id,
                    "bbox": bbox,
                    "cumulative_attention": cumulative_attention,
                    "attention_centroid": attention_centroid,
                }
            )

        return attention_info

    def _get_detected_points(self, attention_info):
        detected_points = []

        for info in attention_info:
            if info["bbox"] is None:
                continue

            x1, y1, x2, y2 = info["bbox"]
            cropped_image = self.image_main[y1:y2, x1:x2]

            try:
                gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.info(
                    f"Failed to convert cropped image to grayscale for instruction {info['instruction_id']}. Error: {e}"  # noqa E501
                )
                continue

            white_y_coords = []
            height, width = gray_cropped.shape

            for x in range(width):
                column = gray_cropped[:, x]
                for y in range(1, height):
                    if column[y - 1] < 200 and column[y] >= 200:
                        white_y_coords.append((x, y))

            closest_midpoint = None
            if white_y_coords:
                y_groups = {}
                for x, y in white_y_coords:
                    if y not in y_groups:
                        y_groups[y] = []
                    y_groups[y].append(x)

                midpoints = []
                for y, x_coords in y_groups.items():
                    mid_x = sum(x_coords) / len(x_coords)
                    midpoints.append((mid_x, y))

                if info["attention_centroid"] is not None and midpoints:
                    attn_x, attn_y = info["attention_centroid"]
                    attn_x_cropped = attn_x - x1
                    attn_y_cropped = attn_y - y1

                    min_distance = float("inf")
                    closest_midpoint = None

                    for mid_x, mid_y in midpoints:
                        distance = (
                            (mid_x - attn_x_cropped) ** 2
                            + (mid_y - attn_y_cropped) ** 2
                        ) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            closest_midpoint = (mid_x, mid_y)

            if closest_midpoint is not None:
                mid_x_cropped, mid_y_cropped = closest_midpoint
                closest_midpoint = (mid_x_cropped + x1, mid_y_cropped + y1)

            detected_points.append(
                {
                    "instruction_id": info["instruction_id"],
                    "closest_midpoint": closest_midpoint,
                    "cumulative_attention": info["cumulative_attention"],
                }
            )

        return detected_points

    def digitize_bar_plot(self):
        attention_info = self._get_attention_info()
        if not attention_info:
            return self.autofill_response
        detected_points = self._get_detected_points(attention_info)

        list_line = self.autofill_response["data"]["lines"]

        for points in detected_points:
            instruction_id = points["instruction_id"]
            closest_midpoint = points["closest_midpoint"]
            if not closest_midpoint:
                continue
            for legend_map in self.legend_map:
                if legend_map["instruction_id"] == instruction_id:
                    line_name = legend_map["legend"]
                    for data in list_line:
                        autofill_line_name = data["labels"]["line_name"]
                        if autofill_line_name == line_name:
                            data["points"].append(
                                {
                                    "x": round(closest_midpoint[0]),
                                    "y": round(closest_midpoint[1]),
                                    "topBarPixelDistance": 0,
                                    "bottomBarPixelDistance": 0,
                                    "deviationPixelDistance": 0,
                                    "x_cat": legend_map["x_tick_value"],
                                    "y_cat": None,
                                }
                            )
                            break
                    break
        return self.autofill_response
