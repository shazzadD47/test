import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Union

import cv2
import json_repair
import numpy as np
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langfuse import observe
from PIL import Image
from pydantic import BaseModel

from app.core.auto import AutoChatModel
from app.utils.image import get_image_from_url
from app.utils.llms import invoke_llm_with_retry
from app.utils.tracing import setup_langfuse_handler
from app.utils.utils import (
    fix_response,
)
from app.v3.endpoints.plot_digitizer.configs import settings as pd_settings
from app.v3.endpoints.plot_digitizer.configs import settings as plot_settings
from app.v3.endpoints.plot_digitizer.constants import (
    ERROR_BAR_SUPPORTED_PLOTS,
    MAX_RETRIES,
    ChartDete,
    ResolutionCheck,
    SubstitutionDigitizer,
)
from app.v3.endpoints.plot_digitizer.langchain_schemas import (
    NumberOfLines,
    PlotLegends,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.prompts import (
    FIGURE_INFO_EXTRACTION_PROMPT,
    FIND_INFO_FROM_FIGURE_PROMPT_SYSTEM_MESSAGE,
    FIND_LEGEND_NAMES_PROMPT,
    FIND_NUMBER_OF_LINES_PROMPT,
)
from app.v3.endpoints.plot_digitizer.schemas import (
    DataPointsAutofil,
    PatchLabelGridImageOut,
    ResolutionCheckOutput,
)
from app.v3.endpoints.plot_digitizer.utils import (
    numpy_to_string_binary,
    post_request_image,
)
from utils.file_ops import (
    download_files_from_storage,
    upload_fileobject_to_storage,
    upload_files_to_storage,
)


def get_descriptions_from_schema(
    model_class, visited_classes: set | None = None
) -> dict[str, Any]:
    """
    Recursively extract all attributes and their descriptions
    from a Pydantic model.

    Args:
        model_class: A Pydantic model class
        visited_classes: Set of already visited classes
        to prevent infinite recursion

    Returns:
        A dictionary mapping attribute names to their descriptions,
        with nested models expanded
    """
    if visited_classes is None:
        visited_classes = set()

    # Prevent infinite recursion
    if model_class in visited_classes:
        return {}

    visited_classes.add(model_class)
    result = {}

    # Check if it's a Pydantic model
    if not hasattr(model_class, "__fields__"):
        if not hasattr(model_class, "annotation"):
            return {}
        else:
            if not hasattr(model_class.annotation, "__fields__"):
                return {}

    # Get model name for prefixing nested attributes
    attributes_field = (
        model_class.__fields__
        if hasattr(model_class, "__fields__")
        else model_class.annotation.__fields__
    )

    for field_name, field in attributes_field.items():
        # Check if the field type is another Pydantic model
        if hasattr(field, "__fields__") or (
            hasattr(field, "annotation") and hasattr(field.annotation, "__fields__")
        ):
            # Recursively get descriptions from nested model
            nested_descriptions = get_descriptions_from_schema(
                field, visited_classes.copy()
            )
            # Add nested descriptions to result
            result.update(nested_descriptions)
        else:
            result[field_name] = {
                "description": field.description,
            }

    return result


def get_questions_from_schema(model_class) -> dict[str, Any]:
    descriptions = get_descriptions_from_schema(model_class)
    queries, keys = [], []
    for key, value in descriptions.items():
        queries.append(value["description"])
        keys.append(key)
    return {
        "questions": queries,
        "keys": keys,
    }


def create_ref_instructions(
    table_structure: list[dict],
    ref_name: str,
) -> str:
    categories = [v["name"] for v in table_structure]
    categories_desc = "\n".join(
        [f"{v['name']}: {v['description']}" for v in table_structure]
    )
    return f"""
    Instruction:
    Map the label of the {ref_name} axis to one of the categories below.

    Categories:
    {categories}

    Category Descriptions:
    {categories_desc}

    Guidelines:

    Match the axis label in the figure to the most appropriate category.

    Check both the label and the tick values on the {ref_name} axis for context.

    Example:

    Label = "Patient A" → Category = "Id Number"

    Label = "Drug names" → Category = "Drug"/"Dosage"

    If no suitable category exists, return "x" if x axis
    and "y" if y axis.
    """


def check_if_dummy_legend_incorrect(
    legends: list[str],
) -> bool:
    if legends is None or len(legends) == 0:
        return True
    if any(re.match(r"(bars?|points?|box|boxes)_\d+", legend) for legend in legends):
        return True
    return False


def convert_string_values_to_na(data: dict) -> dict:
    integer_columns = [
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "x_interval",
        "y_interval",
    ]
    for column in data:
        if column in integer_columns and isinstance(data[column], str):
            try:
                value = int(data[column])
                data[column] = value
            except Exception as e:
                logger.info(f"Error converting value to number: {e}")
                logger.info(f"Value: {data[column]}")
                data[column] = None
    return data


@observe
def get_plot_legends(
    image: str,
    media_type: str,
    chart_type: str = "N/A",
    langfuse_session_id: str = None,
) -> dict:
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    prompt = FIND_LEGEND_NAMES_PROMPT.format(
        plot_image_type=chart_type,
    )
    llm = AutoChatModel.from_model_name(pd_settings.LEGEND_EXTRACTION_MODEL_NAME)
    llm = llm.with_structured_output(PlotLegends)
    message = [
        {
            "role": "system",
            "content": FIND_INFO_FROM_FIGURE_PROMPT_SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]
    result = invoke_llm_with_retry(
        llm, message, config={"callbacks": [langfuse_handler]}
    ).model_dump()
    result["legends"] = list(set(result["legends"]))

    if len(result["legends"]) == 0:
        message = [
            {
                "role": "system",
                "content": FIND_INFO_FROM_FIGURE_PROMPT_SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": FIND_NUMBER_OF_LINES_PROMPT,
                    },
                ],
            },
        ]
        llm = AutoChatModel.from_model_name(plot_settings.LLM_NAME)
        llm = llm.with_structured_output(NumberOfLines)
        line_result = invoke_llm_with_retry(
            llm, message, config={"callbacks": [langfuse_handler]}
        ).model_dump()
        result["legends"] = [
            f"line_{i}" for i in range(1, line_result["number_of_lines"] + 1)
        ]

    if check_if_dummy_legend_incorrect(result["legends"]):
        try:
            invalid_types = ["bar", "point", "box"]
            for invalid_type in invalid_types:
                if invalid_type in result["legends"][0]:
                    result["legends"] = [
                        re.sub(r"{invalid_type}s?_?", "line_", legend)
                        for legend in result["legends"]
                    ]
        except Exception:
            return result
    return result


@observe()
def get_plot_structure_details(
    image: str,
    media_type: str,
    questions: list[str] = None,
    figure_data: str = "",
    schema: BaseModel = None,
    langfuse_session_id: str = None,
) -> dict:
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    prompt = FIGURE_INFO_EXTRACTION_PROMPT.format(
        questions=questions,
        figure_data=figure_data,
    )

    message = [
        {
            "role": "system",
            "content": FIND_INFO_FROM_FIGURE_PROMPT_SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    llm = AutoChatModel.from_model_name(plot_settings.LLM_NAME)
    llm = llm.with_structured_output(schema)

    result = invoke_llm_with_retry(
        llm, message, config={"callbacks": [langfuse_handler]}
    )
    result = fix_figure_axis_data(result.model_dump())
    return result


def fix_figure_axis_data(
    axis_data: dict,
) -> dict:
    axis_data["plot_axis_data"] = convert_string_values_to_na(
        axis_data["plot_axis_data"]
    )
    logger.info(f"axis_data: {axis_data}")
    if axis_data["plot_axis_data"]["x_is_categorical"]:
        axis_data["plot_axis_data"]["x_min"] = None
        axis_data["plot_axis_data"]["x_max"] = None
        axis_data["plot_axis_data"]["x_interval"] = None
        axis_data["plot_axis_data"]["x_unit"] = "N/A"
        axis_data["plot_axis_data"]["x_is_log"] = False
    if axis_data["plot_axis_data"]["y_is_categorical"]:
        axis_data["plot_axis_data"]["y_min"] = None
        axis_data["plot_axis_data"]["y_max"] = None
        axis_data["plot_axis_data"]["y_interval"] = None
        axis_data["plot_axis_data"]["y_unit"] = "N/A"
        axis_data["plot_axis_data"]["y_is_log"] = False
    return axis_data


def fix_line_labels(
    result: str,
    parser: PydanticOutputParser,
    datatype_schema: dict = None,
) -> dict:
    try:
        result = parser.parse(result)
        result = result.dict()
    except Exception:
        logger.exception("Failed to parse the result as JSON. Repairing...")
        json_result = JsonOutputParser().parse(result)
        json_result = json_repair.loads(json.dumps(json_result))
        if "lines" in json_result and datatype_schema is not None:
            updated_lines = []
            for line in json_result["lines"]:
                label_value_map = {k: v["value"] for k, v in line["labels"].items()}
                fixed_labels = fix_response(label_value_map, datatype_schema)
                for k, v in fixed_labels.items():
                    line["labels"][k]["value"] = v
                fixed_output = {
                    "labels": line["labels"],
                }
                for k, v in line.items():
                    if k != "labels":
                        fixed_output[k] = v
                updated_lines.append(fixed_output)
            json_result["lines"] = updated_lines
        result = parser.parse(json.dumps(json_result))
        result = result.dict()
    return result


def check_if_dummy_legend(
    legends: list[str],
) -> bool:
    if all(re.match(r"(lines?|bars?)_\d+", legend) for legend in legends):
        return True
    return False


def download_files_with_retries(
    bucket_path: str,
    file_path: str,
):
    download_retries = 0
    download_success = False
    while download_retries < MAX_RETRIES:
        try:
            download_files_from_storage(bucket_path, file_path)
            if os.path.exists(file_path):
                download_success = True
                break
            else:
                download_retries += 1
                logger.exception("Error downloading file, retrying...")
                continue
        except Exception:
            download_retries += 1
            logger.exception("Error downloading file, retrying...")
            continue

    return file_path, download_success


def upload_files_with_retries(
    filepath: str,
    upload_path: str,
):
    upload_retries = 0
    while upload_retries < MAX_RETRIES:
        try:
            upload_files_to_storage(filepath, upload_path)
            break
        except Exception:
            upload_retries += 1
            logger.exception("Error uploading file, retrying...")
            continue


def create_patch_label_grid_image(
    markers: list[np.ndarray],
    line_names: list[str],
    output_width: int,
    marker_width: int,
    marker_height: int,
    font_scale: float,
    font_thickness: int,
) -> PatchLabelGridImageOut:
    """Create grid of legend patch and substitued legend name"""
    max_columns = SubstitutionDigitizer.GRID_MAX_COLUMN
    padding = SubstitutionDigitizer.GRID_PADDING
    text_spacing = SubstitutionDigitizer.GRID_TEXT_SPACING

    font = cv2.FONT_HERSHEY_COMPLEX

    text_widths = []
    for line_name in line_names:
        text_size = cv2.getTextSize(line_name, font, font_scale, font_thickness)[0]
        text_widths.append(text_size[0])

    column_widths = []
    for col_idx in range(max_columns):
        col_text_widths = [
            text_widths[idx] for idx in range(col_idx, len(markers), max_columns)
        ]
        max_col_text_width = max(col_text_widths) if col_text_widths else 0
        column_widths.append(marker_width + text_spacing + max_col_text_width + padding)

    num_items = len(markers)
    num_rows = math.ceil(num_items / max_columns)

    row_height = marker_height + padding
    output_height = num_rows * row_height + padding

    do_pad = False
    pad_width = None
    total_grid_width = sum(column_widths[: min(max_columns, len(markers))])
    if total_grid_width > output_width:
        do_pad = True
        pad_width = abs(total_grid_width - output_width)
    output_width = max(total_grid_width, output_width)
    output_image = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255
    x_start = (output_width - total_grid_width) // 2

    for idx, (marker_path, line_name) in enumerate(zip(markers, line_names)):
        row = idx // max_columns
        col = idx % max_columns

        x_offset = sum(column_widths[:col]) + x_start
        x = x_offset
        y = row * row_height + padding

        marker_img = marker_path
        marker_img = cv2.resize(marker_img, (marker_width, marker_height))

        output_image[y : y + marker_height, x : x + marker_width] = marker_img

        text_x = x + marker_width + text_spacing
        text_y = y + (marker_height // 2) + round(font_scale * 10)

        cv2.putText(
            output_image,
            line_name,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return {"output_image": output_image, "do_pad": do_pad, "pad_width": pad_width}


def image_resolution_check(image_url: str) -> ResolutionCheckOutput:
    image_byte = get_image_from_url(image_url)
    image_main = np.frombuffer(image_byte, np.uint8)
    image_main = cv2.imdecode(image_main, cv2.IMREAD_COLOR)
    image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)
    height, width, _ = image_main.shape

    is_resized_image = False
    new_height = None
    new_width = None
    if height > ResolutionCheck.DIGITIZER_MAX_RESOLUTION:
        is_resized_image = True
        new_height = ResolutionCheck.DIGITIZER_MAX_RESOLUTION
        new_width = round(width * (new_height / height))
        image_main = cv2.resize(image_main, (new_width, new_height))
    if new_width is None:
        new_width = 0
    if (
        width > ResolutionCheck.DIGITIZER_MAX_RESOLUTION
        or new_width > ResolutionCheck.DIGITIZER_MAX_RESOLUTION
    ):
        is_resized_image = True
        new_width = ResolutionCheck.DIGITIZER_MAX_RESOLUTION
        new_height = round(height * (new_width / width))
        image_main = cv2.resize(image_main, (new_width, new_height))

    object_key = None
    if is_resized_image:
        logger.info(
            f"Image resolution: {height}x{width}, resized into: {new_height}x{new_width}"  # noqa E501
        )

        file_object = io.BytesIO()
        image = Image.fromarray(image_main, mode="RGB")
        del image_main
        image.save(file_object, format="PNG")
        file_object.seek(0)
        upload_path = ResolutionCheck.UPLOAD_PATH
        object_key, image_url = upload_fileobject_to_storage(
            file_object, upload_path, "png"
        )

    return {
        "is_resized_image": is_resized_image,
        "new_height": new_height,
        "new_width": new_width,
        "height": height,
        "width": width,
        "object_key": object_key,
        "image_url": image_url,
    }


def rescale_plot_digitizer(
    resolution_check_output: ResolutionCheckOutput, autofill_response: Any
) -> Any:
    new_height = resolution_check_output["new_height"]
    new_width = resolution_check_output["new_width"]
    height = resolution_check_output["height"]
    width = resolution_check_output["width"]

    for line_data in autofill_response["data"]["lines"]:
        points = line_data["points"]
        for point in points:
            x = point["x"]
            y = point["y"]
            x = (x * width) / new_width
            y = (y * height) / new_height
            point["x"] = round(x)
            point["y"] = round(y)

    for points in autofill_response["data"]["detected_line_points"]:
        for point in points:
            x = point["x"]
            y = point["y"]
            x = (x * width) / new_width
            y = (y * height) / new_height
            point["x"] = round(x)
            point["y"] = round(y)

    return autofill_response


def remove_saved_files(paths: dict):
    for _, path in paths.items():
        if isinstance(path, list):
            for single_path in path:
                if os.path.exists(single_path):
                    os.remove(single_path)
        elif isinstance(path, Union[str, Path]) and os.path.exists(path):
            os.remove(path)


class PlotDigitizerHelper:
    def __init__(self, image_url: Any):
        self.image_url = image_url
        self.image_main = self._get_main_image()
        self.chart_dete_out = self._call_chart_dete()

    def _get_main_image(self) -> np.ndarray:
        image_byte = get_image_from_url(self.image_url)
        image_main = np.frombuffer(image_byte, np.uint8)
        image_main = cv2.imdecode(image_main, cv2.IMREAD_COLOR)
        image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)
        return image_main

    def _call_chart_dete(self) -> Any:
        headers, files = numpy_to_string_binary(self.image_main.copy())

        chart_dete_output = None
        try_count = 0
        while not chart_dete_output:
            chart_dete_output = post_request_image(
                post_url=pd_settings.CHART_DETE_API, headers=headers, files=files
            )
            try_count += 1
            if try_count == MAX_RETRIES:
                logger.info("ChartDete Failed to extract legend patch and label")
                break

        return chart_dete_output

    def _get_plot_area_image_filtered(self) -> np.ndarray:
        x_title = self.chart_dete_out.get("x_title", [])
        plot_area_list = self.chart_dete_out.get("plot_area", [])

        all_outputs = []
        for group in [x_title, plot_area_list]:
            if isinstance(group, list):
                all_outputs.extend(group)

        y2_values = [
            item["y2"]
            for item in all_outputs
            if item.get("score", 0) > 0.9 and "y2" in item
        ]

        max_y2 = max(y2_values) if y2_values else None

        upper_image = None
        if max_y2:
            y_cut = int(max_y2) + 5
            if not y_cut > self.image_main.shape[0]:
                upper_image = self.image_main[:y_cut, :, :].copy()

        return upper_image

    def _upload_file_to_storage(
        self, image_main: np.ndarray, upload_path: str
    ) -> tuple[str, str]:
        file_object = io.BytesIO()
        image = Image.fromarray(image_main, mode="RGB")
        del image_main
        image.save(file_object, format="PNG")
        file_object.seek(0)
        object_key, image_url = upload_fileobject_to_storage(
            file_object, upload_path, "png"
        )
        return object_key, image_url

    @staticmethod
    def _replace_newline(text: str) -> str:
        return text.replace("\n", " ").replace("\x0c", "").strip()

    @staticmethod
    def get_ocr_output(label_gray_image: np.ndarray) -> str:
        headers, files = numpy_to_string_binary(label_gray_image)
        label_text = post_request_image(
            post_url=pd_settings.OCR_API, headers=headers, files=files
        )

        return PlotDigitizerHelper._replace_newline(label_text["ocr_label"])

    def get_filtered_plot_area(self) -> dict:
        if not self.chart_dete_out:
            return {
                "is_plot_area_filtered": False,
                "image_url": self.image_url,
                "object_key": None,
            }

        plot_area_image = self._get_plot_area_image_filtered()

        if plot_area_image is None or (
            isinstance(plot_area_image, np.ndarray) and plot_area_image.size == 0
        ):
            return {
                "is_plot_area_filtered": False,
                "image_url": self.image_url,
                "object_key": None,
            }

        object_key, image_url = self._upload_file_to_storage(
            image_main=plot_area_image, upload_path=ChartDete.UPLOAD_PATH
        )

        return {
            "is_plot_area_filtered": True,
            "image_url": image_url,
            "object_key": object_key,
        }

    @staticmethod
    def get_point_dict(
        point: tuple[int, int, int, int, int] | tuple[int, int], chart_type: str
    ) -> DataPointsAutofil:
        point_dict = {
            "x": round(point[0]),
            "y": round(point[1]),
        }

        if chart_type in ERROR_BAR_SUPPORTED_PLOTS:
            point_dict.update(
                {
                    "topBarPixelDistance": round(point[3]),
                    "bottomBarPixelDistance": round(point[4]),
                    "deviationPixelDistance": round(point[2]),
                }
            )
        else:
            point_dict.update(
                {
                    "topBarPixelDistance": 0,
                    "bottomBarPixelDistance": 0,
                    "deviationPixelDistance": 0,
                }
            )

        return point_dict
