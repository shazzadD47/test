import asyncio
import json

import cv2
import httpx
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

from app.utils.image import async_get_image_from_url, draw_lines
from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.constants import LineFormerRequest
from app.v3.endpoints.plot_digitizer.exceptions import LineFormerFailed
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.schemas import (
    DataPoints,
    LineFormerOutput,
)
from app.v3.endpoints.plot_digitizer.utils import (
    convert_line_dict_tuple,
    convert_tuple_line_dict,
    extend_line,
    line_length,
    points_to_array,
    shorten_line,
)


async def get_lines_lineformer(img_url: str) -> list[LineFormerOutput]:

    lineformer_api = settings.LINEFORMER_API
    params = {"img_url": img_url}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                lineformer_api, params=params, timeout=LineFormerRequest.TIMEOUT
            )

    except Exception:
        logger.exception("An error occured while processing the lineformer model")
        raise LineFormerFailed()
    return json.loads(response.text)


def gen_claude_legend_map_img(image, extended_lines: list[list[DataPoints]]):
    color_plate = LineFormerRequest.COLOR_PLATE
    claude_input = []
    claude_input.append(image)
    for count, line in enumerate(extended_lines):
        in_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
        color = color_plate[count % len(color_plate)]
        tmp_img = draw_lines(in_image, points_to_array([line]), color=color)
        claude_input.append(tmp_img)
        count += 1

    return claude_input


def filter_lines(lines: list[list[DataPoints]]) -> list[list[DataPoints]]:
    filetered_line_list = []
    avg_dis_thresh = LineFormerRequest.AVERAGE_DISTANCE_THRESHOLD
    not_append_list = []

    for idx, line in enumerate(lines):
        ref_line = np.array(line)

        if idx in not_append_list:
            continue

        for idx_com, line_com in enumerate(lines):
            if idx_com != idx and len(line) == len(line_com):
                com_line = np.array(line_com)
                distances = np.sqrt(np.sum((ref_line - com_line) ** 2, axis=1))
                # print(np.sum(distances)/len(distances))
                if np.sum(distances) / len(distances) < avg_dis_thresh:
                    not_append_list.append(idx_com)

        filetered_line_list.append(line)

    return filetered_line_list


def gen_cropped_line_img(line_tuple, image):
    binary_image_list = []
    binary_image_list_ori = []
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    for line_points in line_tuple:
        line_points = np.array(line_points)

        mask = np.zeros_like(image)
        line_points = line_points.astype(int)

        pixel_length = LineFormerRequest.PIXEL_LENGTH
        for point in line_points:
            x, y = point
            cv2.circle(mask, (x, y), pixel_length, (255, 255, 255), -1)

        masked_image = ~cv2.bitwise_and(~image, mask)
        threshold_value = LineFormerRequest.WHITE_THRESHOLD
        max_value = 255
        _, binary_image = cv2.threshold(
            masked_image, threshold_value, max_value, cv2.THRESH_BINARY
        )

        blurred_image = cv2.blur(binary_image, (5, 5), 0)

        edged = cv2.Canny(blurred_image.copy(), 50, 150, apertureSize=3)

        lsd = cv2.createLineSegmentDetector(
            refine=cv2.LSD_REFINE_STD,
        )
        lines = lsd.detect(edged)[0]

        mask = np.ones_like(image) * 255

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = list(map(int, line[0]))
                length_line = line_length(x1, y1, x2, y2)
                short_amount = LineFormerRequest.SHORT_AMOUNT
                if length_line <= 1.5 * short_amount:
                    continue
                line = shorten_line(line, short_amount=short_amount)
                x1, y1, x2, y2 = list(map(int, line[0]))
                random_color = (0, 0, 0)
                cv2.line(mask, (x1, y1), (x2, y2), random_color, thickness=5)

        line_removed = ~np.bitwise_and(~binary_image, mask)
        line_removed = cv2.medianBlur(line_removed, 3)

        kernel = np.ones((3, 3), np.uint8)
        line_removed = cv2.dilate(line_removed, kernel, iterations=1)
        line_removed = cv2.morphologyEx(line_removed, cv2.MORPH_CLOSE, kernel)

        binary_image_list.append(line_removed)
        binary_image_list_ori.append(binary_image)

    return binary_image_list, binary_image_list_ori


def gen_black_pixel_peaks(image, binary_image_list):
    _, width, _ = image.shape

    black_count_list = []
    for binary_image in binary_image_list:
        black_count = []
        for slice in range(0, width):
            slice_image = binary_image[:, slice]
            black = np.sum(slice_image == 0)
            black_count.append(black)
        black_count_list.append(np.array(black_count))

    # filter lower amplitude signal
    low_pass_filter = []
    for signal_data in black_count_list:
        filtered_data = np.array(signal_data)
        filtered_data[filtered_data < LineFormerRequest.SIGNAL_AMPLITUDE] = 0
        low_pass_filter.append(filtered_data)

    # find black count peaks
    all_peaks_list = []
    for black_array in low_pass_filter:
        all_peaks, _ = find_peaks(black_array, distance=1)
        all_peaks_list.append(all_peaks)

    return all_peaks_list


def detect_points(binary_image_list, all_peaks_list):
    slice_window_width = LineFormerRequest.SLIDING_WINDOW
    data_points_list = []

    for image, peaks in zip(binary_image_list, all_peaks_list):
        data_points = []
        for x in peaks:
            slice_skip = False
            for point in data_points:
                distance_center = abs(point[0] - (x + 10 / 2))
                if distance_center <= 10 * (2 / 3):
                    slice_skip = True
                    break

            if slice_skip:
                continue

            sliced_window = image[
                :, x - int(slice_window_width / 2) : x + int(slice_window_width / 2)
            ].copy()
            # Count black pixel
            black_pixels = []
            for row in range(sliced_window.shape[0]):
                for col in range(sliced_window.shape[1]):
                    value = sliced_window[row, col]
                    if value == 0:
                        black_pixels.append((row, col))

            # Localize the densed black area
            if len(black_pixels) > 1:
                kmeans = KMeans(n_clusters=1)
                kmeans.fit(black_pixels)
                cluster_centers = kmeans.cluster_centers_
                cluster_centers = np.copy(cluster_centers).tolist()

                dominant_point = cluster_centers[0]
                dominant_point = list(map(int, dominant_point))

                data_points.append(
                    [
                        int(x - int(slice_window_width / 2) + dominant_point[1]),
                        int(dominant_point[0]),
                    ]
                )

        data_points_list.append(data_points)

    return data_points_list


def detect_error_bar(data_points_list):
    error_bar_lines = []

    for line in data_points_list:
        edited_line = []
        for point in line:
            edited_line.append(
                {
                    "x": int(point[0]),
                    "y": int(point[1]),
                    "topBarPixelDistance": 0,
                    "bottomBarPixelDistance": 0,
                    "deviationPixelDistance": 0,
                }
            )
        error_bar_lines.append(edited_line)

    return error_bar_lines


async def gen_data_point_line_former(img_url: str):
    line_former_coroutine = get_lines_lineformer(img_url)
    get_image_coroutine = async_get_image_from_url(img_url, return_pil_image=True)
    lines_lineformer, image = await asyncio.gather(
        line_former_coroutine, get_image_coroutine
    )
    image = np.array(image)
    lines = [line["line_points"] for line in lines_lineformer]
    lines = extend_line(lines)
    line_tuple = convert_line_dict_tuple(lines)
    line_tuple = filter_lines(line_tuple)
    lines = convert_tuple_line_dict(line_tuple)
    claude_map_input = gen_claude_legend_map_img(
        image=image.copy(), extended_lines=lines
    )
    binary_image_list, _ = gen_cropped_line_img(line_tuple, image.copy())
    all_peaks_list = gen_black_pixel_peaks(image.copy(), binary_image_list)
    data_points_list = detect_points(binary_image_list, all_peaks_list)
    error_bar_lines = detect_error_bar(data_points_list)

    return image, error_bar_lines, claude_map_input
