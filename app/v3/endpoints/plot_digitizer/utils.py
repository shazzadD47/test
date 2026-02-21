import io
import math
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

from app.constants import VALID_D_TYPES
from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.constants import (
    VISION_AGENT_TIME_OUT,
    ChartType,
    Florence2Request,
    LineFormerRequest,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.schemas import (
    DataPoints,
    DataPointsAutofil,
)


def cal_contour_height(c) -> int:
    """
    Return height of a contour
    Args:
    c (list[(int, int)]): list of points (x,y) of a contour

    Returns:
    length_horizontal (int): height of the contour
    """

    all_horizontal_end = c[:, 0, 1]
    start_horizontal = min(all_horizontal_end)
    end_horizontal = max(all_horizontal_end)
    length_horizontal = abs(end_horizontal - start_horizontal)
    return length_horizontal


def cal_contour_length(c) -> int:
    """

    Return horizontal length of a contour
    Args:
    c (list[(int, int)]): list of points (x,y) of a contour

    Returns:
    length_horizontal (int): horizontal length of a contour
    """
    all_horizontal_end = c[:, 0, 0]
    start_horizontal = min(all_horizontal_end)
    end_horizontal = max(all_horizontal_end)
    length_horizontal = abs(end_horizontal - start_horizontal)
    return length_horizontal


def line_length(x1, y1, x2, y2) -> float:
    length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return length


def shorten_line(
    line: list[list[int, int, int, int]], short_amount: int
) -> list[list[int, int, int, int]]:
    # Original line coordinates
    x1, y1, x2, y2 = line[0]
    x_min = min(x1, x2)
    x_max = max(x1, x2)

    # Amount by which to shorten the line
    dL = short_amount

    # Calculate the original length of the line
    dx = x2 - x1
    dy = y2 - y1
    L_original = math.sqrt(dx**2 + dy**2)

    # Calculate the new length
    L_new = L_original - dL

    # Calculate the ratio of the new length to the original length
    ratio = L_new / L_original

    # Calculate new endpoints by moving both inward
    x1_new = x1 + (1 - ratio) / 2 * dx
    y1_new = y1 + (1 - ratio) / 2 * dy
    x2_new = x1 + ratio * dx
    y2_new = y1 + ratio * dy

    new_x_max = max(x1_new, x2_new)
    new_x_min = min(x1_new, x2_new)

    if new_x_max <= x_min or new_x_min >= x_max:
        return [[x1, y1, x2, y2]]
    else:
        return [[x1_new, y1_new, x2_new, y2_new]]


def extend_line(lines: list[list[DataPoints]]) -> list[list[DataPoints]]:
    list_line_points = []
    interpolate_length = LineFormerRequest.INTERPOLATE_LENGTH
    new_point_num = LineFormerRequest.NEW_POINT_NUM
    new_point_start = LineFormerRequest.NEW_POINT_START

    func = LineFormerRequest.interpolate_func

    global_x = []

    for line in lines:
        for point in line:
            x = point["x"]
            global_x.append(x)
    max_x = max(global_x)

    for line in lines:
        line_points = []
        x_points = []
        y_points = []
        for point in line:
            x = point["x"]
            global_x.append(x)
            y = point["y"]
            x_points.append(x)
            y_points.append(y)
            line_points.append({"x": x, "y": y})

        max_x = max(global_x)
        # for end part
        popt, _ = curve_fit(
            func, x_points[-interpolate_length:], y_points[-interpolate_length:]
        )

        for x_in in range(x_points[-1], max_x + new_point_num):
            y = int(func(x_in, *popt))
            line_points.append({"x": x_in, "y": y})

        # for beginnging part
        popt, _ = curve_fit(
            func, x_points[:interpolate_length], y_points[:interpolate_length]
        )

        for x_in in range(x_points[0] - new_point_start, x_points[0]):
            y = int(func(x_in, *popt))
            line_points.append({"x": x_in, "y": y})

        line_points = sorted(line_points, key=lambda point: point["x"])
        list_line_points.append(line_points)

    return list_line_points


def points_to_array(pred_ds):
    res = []
    for line in pred_ds:
        line_arr = []
        for pt in line:
            line_arr.append([pt["x"], pt["y"]])
        res.append(line_arr)
    return res


def convert_line_dict_tuple(lines):
    lines_list = []
    for line in lines:
        tmp_line = []
        for point in line:
            x = point["x"]
            y = point["y"]
            tmp_line.append((x, y))
        lines_list.append(tmp_line)
    return lines_list


def convert_tuple_line_dict(line_tuple):
    lines = []
    for line in line_tuple:
        tmp_line = []
        for point in line:
            point_dict = {"x": int(point[0]), "y": int(point[1])}
            tmp_line.append(point_dict)
        lines.append(tmp_line)
    return lines


def normalize_text(text):
    normalized_text = unicodedata.normalize("NFKC", text)
    normalized_text = re.sub(r"[-–—]", "-", normalized_text)
    normalized_text = normalized_text.replace(" ", "").rstrip().lower()
    normalized_text = re.sub(r"(.)\1+", r"\1", normalized_text)
    return normalized_text


# for florence 2 post processing


def numpy_to_string_binary(image_array, format=".png"):
    """Converts a NumPy array to a Base64-encoded string for binary image data."""
    _, encoded_image = cv2.imencode(format, image_array)

    image_file = io.BytesIO(encoded_image.tobytes())
    image_file.name = f"image.{format}"

    files = {"image": (image_file.name, image_file, f"image/{format}")}
    headers = {"accept": "application/json"}

    return headers, files


def post_request_sentence_xformer(post_url, payload):
    try:
        with httpx.Client() as client:
            response = client.post(
                post_url, params=payload, headers={"accept": "application/json"}
            )
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.debug(f"POST request failed: {e}")
        return None


def post_request_image(post_url, headers, files):
    try:
        with httpx.Client() as client:
            response = client.post(post_url, headers=headers, files=files)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.debug(f"POST request failed: {e}")
        return None


async def post_request_image_aync(post_url, headers, files):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(post_url, headers=headers, files=files)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        print(f"POST request failed: {e}")
        return None


def get_request_florence_2(get_url, params):
    try:
        with httpx.Client(timeout=Florence2Request.TIMEOUT) as client:
            response = client.get(get_url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.debug(f"GET request failed: {e}")
        return None


def calculate_distance(box1, box2):
    centroid1_x = (box1[0] + box1[2]) / 2
    centroid1_y = (box1[1] + box1[3]) / 2
    centroid2_x = (box2[0] + box2[2]) / 2
    centroid2_y = (box2[1] + box2[3]) / 2

    if centroid1_x < centroid2_x:
        x_distance = abs(centroid1_x - centroid2_x)
        y_distance = abs(centroid1_y - centroid2_y)
    else:
        x_distance = 9999
        y_distance = 9999
    return x_distance, y_distance


def resize_image(image: np.ndarray) -> np.ndarray:
    orig_height, orig_width = image.shape[:2]

    target_size = min(max(orig_height, orig_width), 800)

    if orig_height > orig_width:
        pad_left = (orig_height - orig_width) // 2
        pad_right = orig_height - orig_width - pad_left
        pad_top, pad_bottom = 0, 0
    else:
        pad_top = (orig_width - orig_height) // 2
        pad_bottom = orig_width - orig_height - pad_top
        pad_left, pad_right = 0, 0

    image_padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )

    processed_image = cv2.resize(image_padded, (target_size, target_size))
    return processed_image


def open_image(image: np.ndarray) -> np.ndarray:
    image = resize_image(image)
    image = Image.fromarray(image)
    return image


def map_to_original_coordinates(
    point: tuple, orig_height: int, orig_width: int
) -> tuple:
    target_size = min(max(orig_height, orig_width), 800)

    if orig_height > orig_width:
        pad_left = (orig_height - orig_width) // 2
        pad_top = 0

    else:
        pad_top = (orig_width - orig_height) // 2
        pad_left = 0

    x_resized, y_resized = point

    scale_factor = target_size / max(orig_height, orig_width)

    x_padded = x_resized / scale_factor
    y_padded = y_resized / scale_factor

    x_original = int(x_padded - pad_left)
    y_original = int(y_padded - pad_top)

    return x_original, y_original


def extract_data_florence_2(input_string, image_width, image_height):
    pattern = r"(.*?)((?:<loc_\w+>)+)"
    matches = re.findall(pattern, input_string)
    output = []
    for match in matches:
        line_name = match[0]
        points = match[1]
        pattern_point = r"<loc_(.*?)><loc_(.*?)>"
        points_match = re.findall(pattern_point, points)
        for x, y in points_match:
            out_temp = {
                "line_name": line_name,
                "x": int(int(x) * image_width / 1000),
                "y": int(int(y) * image_height / 1000),
            }
            output.append(out_temp)
    return output


def resize_image_with_point(image: np.ndarray, point: tuple) -> tuple:
    """
    Resize an image and transform the point coordinates accordingly.

    Args:
    image (np.ndarray): Input image
    point (tuple): Original point coordinates (x, y)

    Returns:
    tuple: Transformed point coordinates in the resized image
    """
    # Get original dimensions
    orig_height, orig_width = image.shape[:2]

    # Calculate target size
    target_size = min(max(orig_height, orig_width), 800)

    # Calculate padding and scaling
    if orig_height > orig_width:
        pad_left = (orig_height - orig_width) // 2
        pad_right = orig_height - orig_width - pad_left
        pad_top, pad_bottom = 0, 0
        scale_x = target_size / orig_height
        scale_y = target_size / orig_height

        # Transform point coordinates
        new_x = (point[0] + pad_left) * scale_x
        new_y = point[1] * scale_y
    else:
        pad_top = (orig_width - orig_height) // 2
        pad_bottom = orig_width - orig_height - pad_top
        pad_left, pad_right = 0, 0
        scale_x = target_size / orig_width
        scale_y = target_size / orig_width

        # Transform point coordinates
        new_x = point[0] * scale_x
        new_y = (point[1] + pad_top) * scale_y

    # Pad image
    image_padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )

    # Resize image
    processed_image = cv2.resize(image_padded, (target_size, target_size))

    return processed_image, (new_x, new_y)


def overlay_circular_crops(image, circles):
    rgb_image = image.copy()

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    result = None
    for center_x, center_y, radius in circles:
        result = cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)

    output_image = np.full_like(rgb_image, (255, 255, 255))

    output_image[result == 255] = rgb_image[result == 255]
    return output_image


def combine_images_vertically_with_padding(
    image1_path: str | Path,
    image2_path: str | Path,
    bounding_box_legend: dict | None = None,
    bounding_box: dict | None = None,
    padding: int = 30,
    return_media_type: bool = False,
) -> bytes:
    # Open the two images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    if bounding_box_legend and bounding_box is not None:
        bbox_coords = [
            bounding_box["top_left_x"],
            bounding_box["top_left_y"],
            bounding_box["bottom_right_x"],
            bounding_box["bottom_right_y"],
        ]
        bbox_legend_coords = [
            bounding_box_legend["top_left_x"],
            bounding_box_legend["top_left_y"],
            bounding_box_legend["bottom_right_x"],
            bounding_box_legend["bottom_right_y"],
        ]
        bbox_plot_height = abs(bbox_coords[3] - bbox_coords[1])
        bbox_legend_height = abs(bbox_legend_coords[3] - bbox_legend_coords[1])

        bbox_plot_width = abs(bbox_coords[2] - bbox_coords[0])
        bbox_legend_width = abs(bbox_legend_coords[2] - bbox_legend_coords[0])

        plot_heigth, plot_width = image1.height, image1.width

        new_legend_height = int(bbox_legend_height * plot_heigth / bbox_plot_height)
        new_legend_width = int(bbox_legend_width * plot_width / bbox_plot_width)

        image2 = image2.resize((new_legend_width, new_legend_height))

    # Calculate the new width and height
    new_width = max(image1.width, image2.width)
    new_height = image1.height + image2.height + padding

    # Create a new image with a white background
    combined_image = Image.new("RGB", (new_width, new_height), "white")

    # Paste the first image at the top
    combined_image.paste(image1, (0, 0))

    # Center the second image horizontally and paste it with padding
    x_offset = (new_width - image2.width) // 2
    combined_image.paste(image2, (x_offset, image1.height + padding))

    # Save the image to an in-memory bytes buffer
    image_bytes = BytesIO()
    combined_image.save(image_bytes, format="PNG")

    image_bytes = image_bytes.getvalue()

    media_type = "image/png"

    if return_media_type:
        return image_bytes, media_type

    return image_bytes


def convert_image_to_bytes(image: Image.Image) -> bytes:
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue()


def group_and_average_points(
    points: list[DataPointsAutofil],
    threshold: int = settings.POINT_GROUP_PIXEL_THRESHOLD,
) -> list[DataPointsAutofil]:
    """Group points that are closer than the threshold and compute their average."""

    def calculate_distance(point1, point2):
        return (
            (point1["x"] - point2["x"]) ** 2 + (point1["y"] - point2["y"]) ** 2
        ) ** 0.5

    visited = [False] * len(points)
    grouped_points = []

    for i in range(len(points)):
        if visited[i]:
            continue

        group = [points[i]]
        visited[i] = True

        for j in range(i + 1, len(points)):
            if abs(points[i]["x"] - points[j]["x"]) < 3:
                threshold = 2 * settings.POINT_GROUP_PIXEL_THRESHOLD
            else:
                threshold = settings.POINT_GROUP_PIXEL_THRESHOLD

            if not visited[j] and calculate_distance(points[i], points[j]) < threshold:
                group.append(points[j])
                visited[j] = True

        avg_x = sum(p["x"] for p in group) / len(group)
        avg_y = sum(p["y"] for p in group) / len(group)
        grouped_points.append(
            {
                "x": avg_x,
                "y": avg_y,
                "topBarPixelDistance": 0,
                "bottomBarPixelDistance": 0,
                "deviationPixelDistance": 0,
            }
        )

    return grouped_points


def filter_points(response):
    list_line = response["data"]["lines"]
    for line in list_line:
        points = line["points"].copy()
        line["points"] = group_and_average_points(points)
    return response


def extract_plot_area(chart_dete_out: Any, input_image: np.ndarray) -> np.ndarray:
    plot_areas = chart_dete_out["plot_area"]
    plot_area = []
    pad_size = 15
    for plot_area_bbox in plot_areas:
        x1 = int(plot_area_bbox["x1"]) + pad_size
        y1 = int(plot_area_bbox["y1"]) + pad_size
        x2 = int(plot_area_bbox["x2"]) - pad_size
        y2 = int(plot_area_bbox["y2"]) - pad_size
        score = plot_area_bbox["score"]
        if score > 0.85:
            plot_area = [x1, y1, x2, y2]
            out_image = input_image[y1:y2, x1:x2].copy()
            return out_image

    if len(plot_area) == 0:
        return input_image.copy()


def encompassing_bounding_box(
    bounding_boxes: list[int, int, int, int], image_main: np.ndarray
) -> list[int, int, int, int]:
    """Generate a bounding box that encompasses all given bounding boxes."""

    _, width = image_main.shape[0], image_main.shape[1]
    xtops = [box[0] for box in bounding_boxes if len(box) > 0]
    ytops = [box[1] for box in bounding_boxes if len(box) > 0]
    xbottoms = [box[2] for box in bounding_boxes if len(box) > 0]
    ybottoms = [box[3] for box in bounding_boxes if len(box) > 0]

    encompassing_box = [
        min(xtops),
        min(ytops),
        max(xbottoms),
        max(ybottoms),
    ]
    encompassing_box[1] = max(encompassing_box[1] - 50, 0)
    if abs(encompassing_box[2] - width) < 200:
        encompassing_box[2] = width
    encompassing_box[2] = min(encompassing_box[2] + 20, width)

    if encompassing_box[0] < 200:
        encompassing_box[0] = 0

    return encompassing_box


def create_input_for_ge(
    paper_id: str,
    project_id: str,
    table_structure: list[dict],
    image_url: str,
    extracted_legends: list[str] = None,
    generate_labels: list[str] = None,
    chart_type: str = None,
) -> dict:
    line_label, line_label_found = identify_line_label(table_structure)
    if line_label_found:
        if extracted_legends:
            for label in table_structure:
                if label["name"] == line_label:
                    label["answers"] = extracted_legends
                    label["name"] = "line_name"
            # Update line_label to match the renamed label name so that
            # downstream comparisons (filter, generate_labels, extracted_data)
            # use the correct key.
            line_label = "line_name"
    else:
        if chart_type == ChartType.SPIDER_PLOT:
            c_type = "general"
        else:
            c_type = "root"
        line_details = {
            "name": "line_name",
            "description": (
                "Individual name of each line/bar/box/other entity in the figure."
            ),
            "d_type": "string",
            "c_type": c_type,
        }
        if extracted_legends:
            line_details["answers"] = extracted_legends
        table_structure.append(line_details)
        line_label = "line_name"

    updated_table_structure = []
    for label in table_structure:
        if (
            generate_labels is None
            or (
                isinstance(generate_labels, list)
                and len(generate_labels) > 0
                and label["name"] in generate_labels
            )
            or label["name"] == line_label
        ):
            # In the plot digitizer context, only line_name should be
            # root (for non-spider plots). All other labels must be
            # "general" to avoid the GE creating cross-product rows
            # from multiple root labels.
            if label["name"] == line_label:
                if chart_type != ChartType.SPIDER_PLOT:
                    label["c_type"] = "root"
                else:
                    label["c_type"] = "general"
            else:
                label["c_type"] = "general"

            if label["d_type"].lower().strip() not in VALID_D_TYPES:
                label["d_type"] = "string"

            label["description"] = label["description"] + (
                ". Give very specific answers."
            )

            updated_table_structure.append(label)

    if not generate_labels:
        generate_labels = [
            label["name"]
            for label in updated_table_structure
            if label["name"] != line_label
        ]

    return {
        "project_id": project_id,
        "flag_id": paper_id,
        "table_structure": updated_table_structure,
        "custom_instruction": (
            "Provide specific answers for all labels. If the answer can be "
            "given in a few words or as a single word, then do not give a "
            "long answer. For example, if the question is 'What is the drug "
            "name for the control group?', then the answer must be "
            "'Paclitaxel' and not 'The drug name for the control group is "
            "Paclitaxel.' You must provide the answer in the most "
            "non-verbose way possible."
        ),
        "inputs": [
            {
                "type": "chart",
                "data": [
                    {
                        "figure_url": image_url,
                    }
                ],
            }
        ],
        "metadata": {
            "generate_labels": generate_labels,
            "extracted_data": {
                "data": [
                    {line_label: extracted_legend}
                    for extracted_legend in extracted_legends
                ]
            },
        },
    }


def identify_line_label(table_structure: list[dict]) -> tuple[str, bool]:
    for label in table_structure:
        if label["name"].strip().lower() in [
            "line_name",
            "line name",
            "linename",
            "name_of_line",
        ]:
            return label["name"], True
    return "N/A", False


def post_request(post_url, payload):
    try:
        with httpx.Client(timeout=VISION_AGENT_TIME_OUT) as client:
            response = client.post(
                post_url,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.debug(f"POST request for endpoing:{post_url} failed: {e}")
        return None
