import difflib
import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.utils.image import get_image_from_url
from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.constants import (
    MAX_RETRIES,
    ChartDete,
    SubstitutionDigitizer,
)
from app.v3.endpoints.plot_digitizer.helpers import (
    create_patch_label_grid_image,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.schemas import LegendSubstitutionMap
from app.v3.endpoints.plot_digitizer.services.florence_2 import (
    chart_dete_call,
    extract_data_points_rescale_substitution,
    extract_florence_2_data,
    extract_patch_label_ocr,
    florence2_call_manual_mapping,
)
from app.v3.endpoints.plot_digitizer.services.sentence_xfomer_mapping import (  # noqa: E501
    get_sentence_xfomer_mapping,
)
from app.v3.endpoints.plot_digitizer.utils import (
    encompassing_bounding_box,
    get_request_florence_2,
)
from utils.file_ops import delete_bucket_object, upload_fileobject_to_storage


def is_box_overlap(box1: Any, box2: Any) -> bool:
    def calculate_area(box):
        x_top, y_top, x_bottom, y_bottom = box
        return max(0, x_bottom - x_top) * max(0, y_bottom - y_top)

    def calculate_overlap(box1, box2):
        x_top = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_bottom = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        return calculate_area((x_top, y_top, x_bottom, y_bottom))

    overlap_area = calculate_overlap(box1, box2)
    box1_area = calculate_area(box1)
    box2_area = calculate_area(box2)

    min_area = min(box1_area, box2_area)
    return overlap_area >= 0.8 * min_area


def find_best_match(
    input_word: list[str], reference_list: list[str]
) -> tuple[str, int]:
    best_match = ""
    highest_match_length = 0

    for ref_word in reference_list:
        seq_match = difflib.SequenceMatcher(None, input_word, ref_word)
        match = seq_match.find_longest_match(0, len(input_word), 0, len(ref_word))

        if match.size > highest_match_length:
            highest_match_length = match.size
            best_match = ref_word

    return best_match, highest_match_length


def match_input_to_reference(
    input_list: list[str], reference_list: list[str]
) -> list[LegendSubstitutionMap]:
    matched_results = []

    for input_word in input_list:
        matched_word, _ = find_best_match(input_word, reference_list)
        matched_results.append(
            {"input_name": matched_word, "matched_name": matched_word}
        )

    return matched_results


def legend_substitution_digitizer(
    image_main: np.ndarray, img_url: str, autofil_response: Any
) -> Any:
    list_line = autofil_response["data"]["lines"]
    line_count = len(list_line)

    _, _, chart_dete_out = chart_dete_call(image_main.copy(), line_count)

    try:
        legend_area = chart_dete_out["legend_area"][0]
        legend_area = [
            round(legend_area["x1"]),
            round(legend_area["y1"]),
            round(legend_area["x2"]),
            round(legend_area["y2"]),
        ]
    except Exception as e:
        logger.info(f"ChartDete failed on legend_area. error: {e}")
        autofil_response["data"]["has_legend"] = False
        autofil_response = florence2_call_manual_mapping(
            image_main, img_url, autofil_response
        )
        return autofil_response

    legend_label = [
        legend_data
        for legend_data in chart_dete_out["legend_label"]
        if legend_data["score"] > ChartDete.LEGEND_PICK_THEREHOLD
        and is_box_overlap(
            (
                legend_data["x1"],
                legend_data["y1"],
                legend_data["x2"],
                legend_data["y2"],
            ),
            legend_area,
        )
    ]
    legend_patch = [
        legend_data
        for legend_data in chart_dete_out["legend_patch"]
        if legend_data["score"] > ChartDete.LEGEND_PICK_THEREHOLD
        and is_box_overlap(
            (
                legend_data["x1"],
                legend_data["y1"],
                legend_data["x2"],
                legend_data["y2"],
            ),
            legend_area,
        )
    ]
    sustitution_names_list = SubstitutionDigitizer.SUBSTITUTION_LEGENDS

    if (
        len(legend_label) > len(sustitution_names_list)
        or len(legend_label) < len(list_line)
        or len(legend_patch) < len(list_line)
    ):
        logger.info("Legend mismatch ChartDete and Autofil. Calling manual mapping")
        autofil_response["data"]["has_legend"] = False
        autofil_response = florence2_call_manual_mapping(
            image_main, img_url, autofil_response
        )
        return autofil_response
    elif len(legend_label) > len(list_line):
        legend_label = legend_label[: len(list_line)]

    logger.info("Running automatic mapping.")
    legend_bbox = [
        (round(label["x1"]), round(label["y1"]), round(label["x2"]), round(label["y2"]))
        for label in legend_label
    ]

    patch_bbox = [
        (round(patch["x1"]), round(patch["y1"]), round(patch["x2"]), round(patch["y2"]))
        for patch in legend_patch
    ]

    mapped_legend_patch = extract_patch_label_ocr(
        image_main.copy(), legend_bbox, patch_bbox
    )
    try:
        plot_area = chart_dete_out["plot_area"][0]
        plot_area = [
            round(plot_area["x1"]),
            round(plot_area["y1"]),
            round(plot_area["x2"]),
            round(plot_area["y2"]),
        ]
    except Exception as e:
        logger.info(f"ChartDete failed on plot area, error: {e}")
        autofil_response["data"]["has_legend"] = False
        autofil_response = florence2_call_manual_mapping(
            image_main, img_url, autofil_response
        )
        return autofil_response
    try:
        plot_title = chart_dete_out["chart_title"][0]
        plot_title = [
            round(plot_title["x1"]),
            round(plot_title["y1"]),
            round(plot_title["x2"]),
            round(plot_title["y2"]),
        ]
    except Exception as e:
        logger.info(f"No chart title, error: {e}")
        plot_title = []

    try:
        x_title = chart_dete_out["x_title"][0]
        x_title = [
            round(x_title["x1"]),
            round(x_title["y1"]),
            round(x_title["x2"]),
            round(x_title["y2"]),
        ]
    except Exception as e:
        logger.info(f"No x axis title, error: {e}")
        x_title = []
    try:
        y_title = chart_dete_out["y_title"][0]
        y_title = [
            round(y_title["x1"]),
            round(y_title["y1"]),
            round(y_title["x2"]),
            round(y_title["y2"]),
        ]
    except Exception as e:
        logger.info(f"No y axis title, error: {e}")
        y_title = []
    all_boxes = [plot_area, plot_title, x_title, y_title]
    encompassing_box = encompassing_bounding_box(all_boxes, image_main)

    image_annotation = image_main.copy()
    pad = SubstitutionDigitizer.SUBSTITUTION_PAD
    cv2.rectangle(
        image_annotation,
        (legend_area[0] - pad, legend_area[1] - pad),
        (legend_area[2] + pad, legend_area[3] + pad),
        color=(255, 255, 255),
        thickness=-1,
    )
    cv2.rectangle(
        image_annotation,
        (x_title[0], x_title[1]),
        (x_title[2], x_title[3]),
        color=(255, 255, 255),
        thickness=-1,
    )
    cropped_image = image_annotation[
        encompassing_box[1] : encompassing_box[3],
        encompassing_box[0] : encompassing_box[2],
    ].copy()

    markers = [data["patch_image"].copy() for data in mapped_legend_patch]
    for idx, data in enumerate(mapped_legend_patch):
        data["sustituted_legend_name"] = sustitution_names_list[idx]
    line_names = sustitution_names_list[: len(markers)]
    output_width = cropped_image.shape[1]
    marker_width_list = [image.shape[1] for image in markers]
    marker_width = sum(marker_width_list) // len(marker_width_list)
    marker_height_list = [image.shape[0] for image in markers]
    marker_height = sum(marker_height_list) // len(marker_height_list)
    font_scale = SubstitutionDigitizer.FONT_SCALE
    font_thickness = SubstitutionDigitizer.FONT_THICKNESS

    patch_label_grid_output = create_patch_label_grid_image(
        markers,
        line_names,
        output_width,
        marker_width,
        marker_height,
        font_scale,
        font_thickness,
    )
    do_pad = patch_label_grid_output["do_pad"]
    pad_width = patch_label_grid_output["pad_width"]
    output_image = patch_label_grid_output["output_image"]
    if do_pad:
        cropped_image_height = cropped_image.shape[0]
        white_pad = np.full((cropped_image_height, pad_width, 3), 255, dtype=np.uint8)
        cropped_image = np.hstack((cropped_image, white_pad))

    image_substitute = np.vstack((cropped_image, output_image))

    file_object = io.BytesIO()
    image = Image.fromarray(image_substitute, mode="RGB")
    image.save(file_object, format="PNG")
    file_object.seek(0)
    upload_path = "plot-digitizer"
    object_key, image_url = upload_fileobject_to_storage(
        file_object, upload_path, "png"
    )
    logger.info(f"Editted image url: {image_url}")
    if file_object is not None:
        file_object.close()
    del file_object

    call_florence_2 = 0
    while True:
        try:
            florence_2_out = get_request_florence_2(
                get_url=settings.FLORENCE_2_API, params={"img_url": image_url}
            )
            break
        except Exception as e:
            call_florence_2 += 1
            if call_florence_2 > MAX_RETRIES:
                logger.info(f"Florence-2 call error: {e}")
                delete_bucket_object(object_key)
                return autofil_response

    image_byte = get_image_from_url(image_url)
    image_main_sustituted = np.frombuffer(image_byte, np.uint8)
    image_main_sustituted = cv2.imdecode(image_main_sustituted, cv2.IMREAD_COLOR)
    image_main_sustituted = cv2.cvtColor(image_main_sustituted, cv2.COLOR_BGR2RGB)
    delete_bucket_object(object_key)
    data_points_info = extract_florence_2_data(
        image_main_sustituted.copy(), florence_2_out
    )

    data_point_structured = extract_data_points_rescale_substitution(
        image_main_sustituted.copy(), data_points_info, encompassing_box
    )

    for point_data in data_point_structured:
        point_list = point_data["data_points"]
        temp_point = []
        for point in point_list:
            temp_point.append(
                {
                    "x": round(point[0]),
                    "y": round(point[1]),
                    "topBarPixelDistance": round(point[3]),
                    "bottomBarPixelDistance": round(point[4]),
                    "deviationPixelDistance": round(point[2]),
                }
            )
        autofil_response["data"]["detected_line_points"].append(temp_point)

    florence_2_legends = [data["line_name"] for data in data_point_structured]
    ground_substituted_legend = [
        data["sustituted_legend_name"] for data in mapped_legend_patch
    ]

    matched_results = match_input_to_reference(
        input_list=ground_substituted_legend, reference_list=florence_2_legends
    )

    for data in mapped_legend_patch:
        substituted_legend_name = data["sustituted_legend_name"]
        matched_name = None
        for matched_result in matched_results:
            input_name = matched_result["input_name"]
            if input_name == substituted_legend_name:
                matched_name = matched_result["matched_name"]
                break

        if matched_name is not None:
            for line_data in data_point_structured:
                line_name = line_data["line_name"]
                if line_name == matched_name:
                    data["data_points"] = line_data["data_points"]
                    break

    autofil_response = get_sentence_xfomer_mapping(
        mapped_legend_patch, autofil_response
    )

    return autofil_response
