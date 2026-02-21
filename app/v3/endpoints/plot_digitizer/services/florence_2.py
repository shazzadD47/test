from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.constants import (
    MAX_RETRIES,
    OCR,
    ChartDete,
    ChartType,
    Florence2Request,
)
from app.v3.endpoints.plot_digitizer.helpers import PlotDigitizerHelper
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.schemas import (
    ChartDeteExtraction,
    Florence2DataInfo,
    Florence2Output,
    MappedLegendPatch,
)
from app.v3.endpoints.plot_digitizer.services.color_filtering import (
    filter_by_color,
    find_dominant_color,
)
from app.v3.endpoints.plot_digitizer.services.error_bar_detection import (
    add_error_bar,
)
from app.v3.endpoints.plot_digitizer.utils import (
    calculate_distance,
    extract_data_florence_2,
    get_request_florence_2,
    map_to_original_coordinates,
    numpy_to_string_binary,
    open_image,
    overlay_circular_crops,
    post_request_image,
    resize_image,
    resize_image_with_point,
)


def chart_dete_extraction(chart_dete_out, line_count: int) -> ChartDeteExtraction:
    legend_label_bbox = []
    legend_patch_bbox = []
    for class_name in chart_dete_out:
        if (
            len(legend_label_bbox) == line_count
            and len(legend_patch_bbox) == line_count
        ):
            break

        if class_name == "legend_patch" or class_name == "legend_label":
            class_bboxes = chart_dete_out[class_name]
            for bbox in class_bboxes:
                x1 = bbox["x1"]
                y1 = bbox["y1"]
                x2 = bbox["x2"]
                y2 = bbox["y2"]
                score = bbox["score"]
                if score > ChartDete.EXTRACTION_THRESHOLD:
                    if class_name == "legend_label":
                        legend_label_bbox.append(
                            (round(x1), round(y1), round(x2), round(y2))
                        )
                    elif class_name == "legend_patch":
                        legend_patch_bbox.append(
                            (round(x1), round(y1), round(x2), round(y2))
                        )

    line_count_mismatch = False
    if len(legend_label_bbox) < line_count or len(legend_patch_bbox) < line_count:
        logger.info("Mismatched between autofill and ChartDete.")
        line_count_mismatch = True
    no_legend_found = False
    if len(legend_label_bbox) == 0 or len(legend_patch_bbox) == 0:
        logger.info("No legend found from ChartDete.")
        no_legend_found = True

    return {
        "legend_label_bbox": legend_label_bbox,
        "legend_patch_bbox": legend_patch_bbox,
        "line_count_mismatch": line_count_mismatch,
        "no_legend_found": no_legend_found,
    }


def extract_patch_label_ocr(
    image_main: np.ndarray,
    legend_label_bbox: list[list[int, int, int, int]],
    legend_patch_bbox: list[list[int, int, int, int]],
) -> MappedLegendPatch:
    def replace_newline(text):
        return text.replace("\n", "").replace("\x0c", "")

    range_len = min(len(legend_patch_bbox), len(legend_label_bbox))

    mapped_legend_patch = []

    for idx in range(range_len):
        distance_array = []
        label_x1, label_y1, label_x2, label_y2 = legend_label_bbox[idx]

        for patch_bbox in legend_patch_bbox:
            x_distance, y_distance = calculate_distance(
                patch_bbox, legend_label_bbox[idx]
            )
            distance_array.append((x_distance, y_distance))

        y_threshold = OCR.LABLE_EXTRACT_Y_THRESHOLD
        filtered_data = [
            (i, t) for i, t in enumerate(distance_array) if t[1] < y_threshold
        ]
        if filtered_data:
            min_index, _ = min(filtered_data, key=lambda x: x[1][0])

        patch_x1, patch_y1, patch_x2, patch_y2 = legend_patch_bbox[min_index]

        patched_image = image_main[patch_y1:patch_y2, patch_x1:patch_x2].copy()
        pad = OCR.PAD
        label_x1, label_y1, label_x2, label_y2 = (
            label_x1 - pad,
            label_y1 - pad,
            label_x2 + pad,
            label_y2 + pad,
        )
        label_image = image_main[label_y1:label_y2, label_x1:label_x2].copy()

        label_gray_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
        headers, files = numpy_to_string_binary(label_gray_image)
        label_text = post_request_image(
            post_url=settings.OCR_API, headers=headers, files=files
        )

        mapped_legend_patch.append(
            {
                "label_bbox": legend_label_bbox[idx],
                "patch_bbox": legend_patch_bbox[min_index],
                "patch_image": patched_image,
                "label_image": label_image,
                "lable_text": replace_newline(label_text["ocr_label"]),
                "data_points": None,
            }
        )

    return mapped_legend_patch


def extract_florence_2_data(
    image_main: np.ndarray, florence_2_out: str, chart_type: str = None
) -> Florence2DataInfo:
    if chart_type == ChartType.SPIDER_PLOT:
        precess_image = Image.fromarray(image_main.copy())
    else:
        precess_image = open_image(image_main.copy())

    image_height = precess_image.height
    image_width = precess_image.width

    suffix_out = extract_data_florence_2(
        florence_2_out["generated_text"], image_width, image_height
    )

    extracted_line_name = []
    data_points_info = {"data_points": []}
    for data in suffix_out:
        line_name = data["line_name"].replace("</s>", "").replace("<s>", "")
        if line_name not in extracted_line_name:
            extracted_line_name.append(line_name)
            tmp_line_data = {line_name: [(data["x"], data["y"])]}
            data_points_info["data_points"].append(tmp_line_data)
        else:
            for line in data_points_info["data_points"]:
                check_line_name = list(line.keys())[0]
                if check_line_name == line_name:
                    line[line_name].append((data["x"], data["y"]))
                    break

    return data_points_info


def generated_point_cropped_image(
    image_main: np.ndarray, data_points_info: Florence2DataInfo
) -> list[np.ndarray]:
    cropped_images = []
    radius = Florence2Request.point_radius
    for line in data_points_info["data_points"]:
        tmp_image = image_main.copy()
        tmp_image = resize_image(tmp_image)
        tmp_line_list = []
        data_points = list(line.values())[0]
        for point in data_points:
            x, y = point
            _, point = resize_image_with_point(tmp_image, (x, y))
            x, y = point
            tmp_line_list.append((round(x), round(y), radius))
        cropped_image = overlay_circular_crops(tmp_image, tmp_line_list)
        cropped_images.append(cropped_image)
    return cropped_images


def extract_point(
    image_main: np.ndarray,
    mapped_legend_patch: MappedLegendPatch,
    cropped_images: list[np.ndarray],
    data_points_info: Florence2DataInfo,
) -> MappedLegendPatch:

    visited = [False] * len(cropped_images)
    for mapped_data in mapped_legend_patch:
        patch_image = mapped_data["patch_image"]
        # TODO Check hollow image
        dominant_color, _ = find_dominant_color(patch_image)
        black_count_list = []

        for idx, line_cluster_image in enumerate(cropped_images):
            if visited[idx]:
                black_count_list.append(0)
                continue
            color_filtered = filter_by_color(line_cluster_image, dominant_color)
            black_count = np.count_nonzero(color_filtered == 0)
            black_count_list.append(black_count)

        max_black_count = max(black_count_list)
        max_black_index = black_count_list.index(max_black_count)
        visited[max_black_index] = True
        data_points = list(data_points_info["data_points"][max_black_index].values())[0]

        mapped_data["data_points"] = data_points

    return mapped_legend_patch


def rescale_points(
    mapped_legend_patch: MappedLegendPatch, image_main: np.ndarray
) -> MappedLegendPatch:
    plot_image = image_main.copy()
    height = plot_image.shape[0]
    width = plot_image.shape[1]

    for lines in mapped_legend_patch:
        data_points = lines["data_points"]
        temp_datapoints = []
        for point in data_points:
            x, y = point
            x, y = map_to_original_coordinates((x, y), height, width)
            temp_datapoints.append((x, y))
        lines["data_points"] = temp_datapoints
    return mapped_legend_patch


@add_error_bar
def extract_data_points_rescale(
    image_main: np.ndarray, data_points_info: Florence2DataInfo, chart_type: str = None
) -> list[Florence2Output]:
    data_point_structured = []
    for line in data_points_info["data_points"]:
        point_list = list(line.values())[0]
        line_name = list(line.keys())[0]
        data_point_structured.append(
            {"line_name": line_name, "data_points": point_list}
        )
    if chart_type != ChartType.SPIDER_PLOT:
        data_point_structured = rescale_points(data_point_structured, image_main)

    return data_point_structured


@add_error_bar
def extract_data_points_rescale_substitution(
    image_main: np.ndarray,
    data_points_info: Florence2DataInfo,
    encompassing_box: list[int, int, int],
) -> list[Florence2Output]:
    data_point_structured = []
    current_origin = np.array([encompassing_box[0], encompassing_box[1]])
    for line in data_points_info["data_points"]:
        point_list = list(line.values())[0]
        numpy_point_list = np.array(point_list)
        numpy_point_list_offset = numpy_point_list + current_origin
        point_list = numpy_point_list_offset.tolist()
        line_name = list(line.keys())[0]
        data_point_structured.append(
            {
                "line_name": line_name,
                "data_points": point_list,
            }
        )
    data_point_structured = rescale_points(data_point_structured, image_main.copy())

    return data_point_structured


def get_florence_2_only(
    image_main: np.ndarray, img_url: str, chart_type: str = None
) -> list[Florence2Output]:
    if chart_type == ChartType.SPIDER_PLOT:
        api_endpoint = settings.SPIDER_PLOT_API
    else:
        api_endpoint = settings.FLORENCE_2_API

    florence_2_out = get_request_florence_2(
        get_url=api_endpoint, params={"img_url": img_url}
    )
    data_points_info = extract_florence_2_data(image_main, florence_2_out, chart_type)
    data_point_structured = extract_data_points_rescale(
        image_main=image_main, data_points_info=data_points_info, chart_type=chart_type
    )

    return data_point_structured


def chart_dete_call(image_main: np.ndarray, line_count: int) -> tuple[bool, Any, Any]:
    headers, files = numpy_to_string_binary(image_main.copy())

    chart_dete_out = None
    no_legend_found = False
    try_count = 0
    while type(chart_dete_out).__name__ == "NoneType":
        chart_dete_out = post_request_image(
            post_url=settings.CHART_DETE_API, headers=headers, files=files
        )
        try_count += 1
        if try_count == MAX_RETRIES:
            logger.info("ChartDete Failed to extract legend patch and label")
            no_legend_found = True

    char_dete_response = chart_dete_extraction(chart_dete_out, line_count)
    line_count_mismatch = char_dete_response["line_count_mismatch"]
    no_legend_found = char_dete_response["no_legend_found"]

    chart_dete_fail = no_legend_found or line_count_mismatch

    return chart_dete_fail, char_dete_response, chart_dete_out


def get_florence_2_points(
    image_main: np.ndarray,
    img_url: str,
    char_dete_response: ChartDeteExtraction,
    line_count: int,
    chart_type: str,
) -> MappedLegendPatch:
    not_enough_line = False
    florence_2_out = get_request_florence_2(
        get_url=settings.FLORENCE_2_API, params={"img_url": img_url}
    )
    legend_label_bbox = char_dete_response["legend_label_bbox"]
    legend_patch_bbox = char_dete_response["legend_patch_bbox"]
    mapped_legend_patch = extract_patch_label_ocr(
        image_main, legend_label_bbox, legend_patch_bbox
    )
    data_points_info = extract_florence_2_data(image_main, florence_2_out, chart_type)
    if (
        len(data_points_info["data_points"]) < line_count / 2
        and chart_type != ChartType.SCATTER_PLOT
    ):
        not_enough_line = True
        mapped_legend_patch = None
        return mapped_legend_patch, not_enough_line

    cropped_images = generated_point_cropped_image(image_main, data_points_info)
    mapped_legend_patch = extract_point(
        image_main, mapped_legend_patch, cropped_images, data_points_info
    )
    mapped_legend_patch = rescale_points(mapped_legend_patch, image_main)

    return mapped_legend_patch, not_enough_line


def run_florence2_randomly(
    image_main: np.ndarray, img_url: str, autofil_response: Any
) -> Any:
    list_line = autofil_response["data"]["lines"]
    data_point_structured = get_florence_2_only(
        image_main,
        img_url,
    )
    point_list = [line_data["data_points"] for line_data in data_point_structured]
    points_count = len(point_list) - 1
    for idx, data in enumerate(list_line):
        points = point_list[idx if idx <= points_count else idx % points_count]
        temp_point = []
        for point in points:
            temp_point.append(
                {
                    "x": round(point[0]),
                    "y": round(point[1]),
                    "topBarPixelDistance": round(point[3]),
                    "bottomBarPixelDistance": round(point[4]),
                    "deviationPixelDistance": round(point[2]),
                }
            )
        data["points"] = temp_point
    return autofil_response


def florence2_call_manual_mapping(
    image_main: np.ndarray, img_url: str, autofil_response: Any, chart_type: str = None
) -> Any:
    data_point_structured = get_florence_2_only(image_main, img_url, chart_type)

    for point_data in data_point_structured:
        point_list = point_data["data_points"]
        temp_point = []
        for point in point_list:
            point_dict = PlotDigitizerHelper.get_point_dict(point, chart_type)
            temp_point.append(point_dict)

        autofil_response["data"]["detected_line_points"].append(temp_point)

    return autofil_response
