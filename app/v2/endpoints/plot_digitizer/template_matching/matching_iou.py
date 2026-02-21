import colorsys
import math
from traceback import print_exc

import cv2
import numpy as np
from sklearn.cluster import KMeans

from app.v2.endpoints.plot_digitizer.tick_origin_detection.services import crop_image
from app.v2.endpoints.plot_digitizer.utils import get_image_from_url


def rgb_to_hsv(r, g, b):
    # Normalize the RGB values by dividing by 255
    r_normalized, g_normalized, b_normalized = r / 255.0, g / 255.0, b / 255.0

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r_normalized, g_normalized, b_normalized)

    # Convert Hue from range [0, 1] to [0, 360] and Saturation, Value to [0, 100]
    h = int(h * 360)
    s = int(s * 100)
    v = int(v * 100)

    return h, s, v


def focus_to_color(bb_tuple, reference_image):
    center_x = bb_tuple[0] + bb_tuple[2] / 2
    center_y = bb_tuple[1] + bb_tuple[3] / 2

    square_length = 4
    x_top = int(center_x - square_length)
    y_top = int(center_y - square_length)

    crop_bb = {"x": x_top, "y": y_top, "width": square_length, "height": square_length}
    return crop_image(reference_image.copy(), crop_bb)


def find_dominant_color(cv_image, n_clusters=4, print_dominant=False):
    """
    Finds the dominant color in an image using K-means clustering.

    Args:
    cv_image (numpy.ndarray): Input image in BGR format.
    n_clusters (int): Number of clusters to use in K-means.
    Defaults to 1 for dominant color.

    Returns:
    tuple: Dominant color in RGB format.
    """
    # Convert the image from BGR to RGB
    img_rgb = cv_image[:, :, ::-1]

    # Reshape the image to a 2D array of pixels
    pixels = img_rgb.reshape(-1, 3)

    # Use K-means to cluster pixel colors
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)

    kmeans.cluster_centers_

    # Find the most dominant cluster center (RGB value)
    dominant_colors = kmeans.cluster_centers_

    # check for black and white image
    # if all the channel of all color is of same value it is bw image
    test_color_list = np.copy(dominant_colors).tolist()
    test_color_list = test_color_list[1:]
    average = [(x[0] + x[1] + x[2]) / 3 for x in test_color_list]
    diff = [abs(x[0][0] - x[1]) < 0.3 for x in zip(test_color_list, average)]
    bw_image = len(diff) == sum(diff)

    # color filtering based on HSV color domain
    color_selected = None
    big_s = 0
    for center in kmeans.cluster_centers_:
        _, s, _ = rgb_to_hsv(*center.astype(int))
        if s > big_s:
            color_selected = center.astype(int)
            big_s = s

    return tuple(color_selected), bw_image


def filter_by_color(cv_image, target_color, tolerance=30):
    """
    Filter an image by a specific RGB color within a given tolerance.

    Args:
    cv_image (numpy.ndarray): Input image in BGR format.
    target_color (tuple): Target RGB color as (R, G, B).
    tolerance (int): Tolerance for color matching.

    Returns:
    numpy.ndarray: Filtered image with highlighted areas matching the target color.
    """
    # adjust tolerence value
    _, saturation, _ = rgb_to_hsv(*target_color)
    tolerance = 30 if saturation > 30 else 5

    image = cv_image.copy()
    img_rgb = image[:, :, ::-1]

    distances = np.sqrt(((img_rgb - np.array(target_color)) ** 2).sum(axis=2))

    mask = distances < tolerance

    mask_bw = (mask * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask_bw, 0, 255, cv2.THRESH_BINARY)

    inverted_binary = cv2.bitwise_not(binary)

    return inverted_binary


def non_max_suppression(bounding_boxes, iou_threshold=0.02):
    """
    Filter all the overlapped bounding boxes based on their
    IoU with the reference mask given by the user using non max supression (NMS)

    Args:
    bounding_boxes (list): List of predicted bounding boxes
    (x_top_left, y_top_left, x_bottom_right, y_bottom_right, iou_with_ref)
    overlapped with the reference mask.
    iou_threshold (float): Value of IoU to filter bounding boxes.

    Returns:
    bounding_boxes (list): Filtered bounding boxes
    (x_top_left, y_top_left, x_top_right, y_top_right, iou_with_ref) using NMS.
    """

    if len(bounding_boxes) == 0:
        return []

    boxes = np.array(bounding_boxes)

    picked_indices = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        picked_indices.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        indices = np.where(iou <= iou_threshold)[0]

        order = order[indices + 1]

    return boxes[picked_indices].tolist()


def draw_template_contour(cropped_template_img, color, plot=False):
    """
    Detect contour around the input user template

    Args:
    cropped_template_img (numpy.ndarray): User selected template
    color (list): Dominant color in the template
    plot (bool): Condition to show the coutour in plot

    Returns:
        contour_image (numpy.ndarray): Countour drawed image
    """
    template_gray = cv2.cvtColor(cropped_template_img, cv2.COLOR_BGR2GRAY)
    ret, template_binary = cv2.threshold(template_gray, 240, 255, 1)
    contours, hierarchy = cv2.findContours(
        template_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contour_image = np.zeros(
        shape=[cropped_template_img.shape[0], cropped_template_img.shape[1]]
    ).astype("uint8")
    cv2.drawContours(contour_image, contours, -1, 255, 1)

    return contour_image


def template_flood_fill(contour_image, plot=False):
    """
    Fill inside the countour of hollow template

    Args:
    contour_image (numpy.ndarray): countour of the template

    Returns:
        flood_img (numpy.ndarray): Filled image
    """
    mask = np.zeros(
        shape=[contour_image.shape[0] + 2, contour_image.shape[1] + 2]
    ).astype("uint8")
    seedPoint = (
        int((contour_image.shape[0] + 2) / 2),
        int((contour_image.shape[1] + 2) / 2),
    )
    retval, flood_img, mask, rect = cv2.floodFill(
        contour_image.copy(), mask, seedPoint=seedPoint, newVal=255
    )

    return flood_img


def cal_overlap(cropped_template_img, color, flood_img):
    """
    Calculate overlap between template and the flood filled image

    Args:
    cropped_template_img (numpy.ndarray): countour of the template
    flood_img (numpy.ndarray): countour of the template
    color (list): Dominant color in the template

    Returns:
        iou (float): overlap between template and the flood filled image
    """
    template_gray = cv2.cvtColor(cropped_template_img, cv2.COLOR_BGR2GRAY)
    _, template_binary = cv2.threshold(template_gray, 240, 255, 1)
    _, flood_img_binary = cv2.threshold(flood_img, 240, 255, 1)

    overlap = np.bitwise_and(~template_binary, flood_img_binary)
    union = np.bitwise_or(~template_binary, flood_img_binary)

    iou = np.sum(overlap) / np.sum(union)

    return iou


def iou_match(
    template_binary,
    filtered_binary,
    hollow=True,
    hollow_marker=False,
    bw_image=False,
    tick_based_boxes=None,
    sliding_window_boxes=None,
):
    """
    Determine the match for input template all over the image

    Args:
    template_binary (numpy.ndarray): template image
    filtered_binary (numpy.ndarray): input image of the plot
    hollow (bool): if the marker is hollow this is true
    hollow_marker (bool): if the marker is hollow this is True
    bw_image (bool): if the plot is black and white this is True

    Returns:
        bounding_boxes (list[Tuple]): List of template matched bounding boxes
        in this format (x_top_left, y_top_left, width, height)
    """
    if tick_based_boxes is None:
        tick_based_boxes = []

    template_window_height, template_window_width = template_binary.shape
    reference_image_height, reference_image_width = filtered_binary.shape

    stride = 1

    if bw_image:
        iou_threshold_marker = 0.8
        iou_threshold_outside = 0.9
        bounding_boxes_nms = []

        for y in range(0, reference_image_height - template_window_height + 1, stride):
            for x in range(
                0, reference_image_width - template_window_width + 1, stride
            ):
                # Define the box for the current crop
                box = (x, y, x + template_window_width, y + template_window_height)

                cropped_image = filtered_binary[box[1] : box[3], box[0] : box[2]]

                if np.sum(cropped_image) == 0:
                    continue

                intersection = np.bitwise_and(template_binary, cropped_image)
                union = np.bitwise_or(template_binary, cropped_image)
                iou_marker = np.sum(intersection) / np.sum(union)

                intersection = np.bitwise_and(
                    np.abs(template_binary - 1), np.abs(cropped_image - 1)
                )
                union = np.bitwise_or(
                    np.abs(template_binary - 1), np.abs(cropped_image - 1)
                )
                iou_outside = np.sum(intersection) / np.sum(union)

                box_nms = (
                    x,
                    y,
                    x + template_window_width,
                    y + template_window_height,
                    iou_marker,
                )

                if (
                    iou_marker > iou_threshold_marker
                    and iou_outside > iou_threshold_outside
                ):
                    bounding_boxes_nms.append(box_nms)

        filtered_box = non_max_suppression(bounding_boxes_nms, iou_threshold_marker)
        bounding_boxes = [
            (box[0], box[1], box[2] - box[0], box[3] - box[1]) for box in filtered_box
        ]
        return bounding_boxes

    else:
        # iou_threshold = 0.4 if hollow else 0.7
        # iou_start_threshold = 0.2
        # bounding_boxes_nms = []
        # start = 0

        # for y in range((0, reference_image_height -
        # template_window_height + 1, stride)):
        #     for x in range(
        #         0, reference_image_width - template_window_width + 1, stride
        #     ):
        #         # Define the box for the current crop
        #         box = (x, y, x + template_window_width, y + template_window_height)

        #         cropped_image = filtered_binary[box[1] : box[3], box[0] : box[2]]

        #         if np.sum(cropped_image) == 0:
        #             continue

        #         intersection = np.bitwise_and(template_binary, cropped_image)
        #         union = np.bitwise_or(template_binary, cropped_image)
        #         iou = np.sum(intersection) / np.sum(union)

        #         box_nms = (
        #             x,
        #             y,
        #             x + template_window_width,
        #             y + template_window_height,
        #             iou,
        #         )

        #         if iou > iou_threshold:
        #             bounding_boxes_nms.append(box_nms)

        #         if start == 0 and iou > iou_start_threshold:
        #             bounding_boxes_nms.append(box_nms)
        #             start = 1

        # iou_threshold = 0.02 if hollow else 0.35
        # bounding_boxes_nms = bounding_boxes_nms + tick_based_boxes

        # test for sliding window
        iou_threshold = 0.7
        bounding_boxes_nms = sliding_window_boxes
        filtered_box = non_max_suppression(bounding_boxes_nms, iou_threshold)
        # filtered_box = non_max_suppression(filtered_box, 0.02)
        bounding_boxes = [
            (box[0], box[1], box[2] - box[0], box[3] - box[1]) for box in filtered_box
        ]
        return bounding_boxes


def match_hollow(
    cropped_template_img, reference_image, template_binary, bw_image=False
):
    """
    Match hollow template with e possible region in the plot

    Args:
    cropped_template_img (numpy.ndarray): template image
    reference_image (numpy.ndarray): input image of the plot
    template_binary (numpy.ndarray): template image in binary pixel value
    bw_image (bool): if the plot is black and white this is True

    Returns:
        filtered_box (list[Tuple]): List of template matched bounding boxes
        in this format (x_top_left, y_top_left, width, height)
    """
    color, _ = find_dominant_color(cropped_template_img.copy())
    filtered_img = filter_by_color(reference_image.copy(), color, tolerance=30)

    in_img = ~filtered_img.copy()

    kernel = np.ones((3, 3), np.uint8)
    in_img = cv2.dilate(in_img, kernel, iterations=1)

    mask = np.zeros(shape=[in_img.shape[0] + 2, in_img.shape[1] + 2]).astype("uint8")
    retval, flood_img, mask, rect = cv2.floodFill(
        in_img.copy(), mask, seedPoint=(0, 0), newVal=255
    )

    mask_image = np.bitwise_or(~flood_img, in_img)

    hollow_marker = True
    filtered_box = iou_match(
        template_binary, mask_image, hollow_marker=hollow_marker, bw_image=bw_image
    )

    return filtered_box


def extract_axes_information(result):
    x_max = None
    y_max = None

    for point in result["points"]:
        if point["label"] == "ymax":
            y_max = point

        if point["label"] == "xmax":
            x_max = point

    return y_max, x_max


def tick_based_marker_detect(
    axes_information, intersection, ref_image, filtered_img, bb_tuple
):
    # detect x axis ticks
    gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    thresh_image = np.where(gray < 240, 1, 0).astype("uint8")

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 7)
    )  # optimum value 5 to 7
    detect_vertical = cv2.morphologyEx(
        thresh_image, cv2.MORPH_OPEN, vertical_kernel, iterations=1
    )

    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # point are in (width, height) position fashion
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    test_result = ref_image.copy()

    x_ticks_centroids = []

    for c in cnts:
        (cx, cy) = cal_contour_center(c)
        if (intersection[1] <= cy <= intersection[1] + 12) and cx >= intersection[
            0
        ]:  # condition for xmax
            x_ticks_centroids.append((cx, cy))
            cv2.drawContours(test_result, [c], -1, (255, 0, 0), 5)

    # detect points
    slice_window_width = bb_tuple[2]

    y_max, _ = extract_axes_information(axes_information)

    data_points = []

    for x_tick in x_ticks_centroids:
        x_temp_tick = x_tick[0]
        start_window = x_temp_tick - int(slice_window_width / 2)
        end_window = x_temp_tick + int(slice_window_width / 2)
        sliced_window = filtered_img[:, start_window:end_window].copy()

        sliced_window[0 : y_max["y"], :] = 255
        sliced_window[intersection[1] :, :] = 255

        black_pixels = []
        for row in range(sliced_window.shape[0]):
            for col in range(sliced_window.shape[1]):
                value = sliced_window[row, col]
                if value == 0:
                    black_pixels.append((row, col))
        if len(black_pixels) > 0:
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(black_pixels)
            cluster_centers = kmeans.cluster_centers_
            cluster_centers = np.copy(cluster_centers).tolist()

            dominant_point = cluster_centers[0]
            dominant_point = list(map(int, dominant_point))
            data_points.append((start_window + dominant_point[1], dominant_point[0]))

            cv2.circle(
                test_result,
                (start_window + dominant_point[1], dominant_point[0]),
                3,
                (0, 0, 255),
                3,
            )

    bb_boxes = [
        (
            point[0] - int(bb_tuple[2] / 2),
            point[1] - int(bb_tuple[3] / 2),
            point[0] + int(bb_tuple[2] / 2),
            point[1] + int(bb_tuple[3] / 2),
            0.6,
        )
        for point in data_points
    ]

    return bb_boxes


def line_length(x1, y1, x2, y2):
    length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return length


def shorten_line(line, short_amount):
    # Original line coordinates
    x1, y1, x2, y2 = line[0]

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

    return [[x1_new, y1_new, x2_new, y2_new]]


def calculate_average_slope(points):
    # Ensure points are sorted by x-values
    points = sorted(points, key=lambda point: point[0])

    # Calculate slopes between consecutive points
    slopes = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)

    # Calculate and return the average slope
    average_slope = sum(slopes) / len(slopes)
    return average_slope


def normal_distribution(slice):
    black_pixels = []

    for idx, col in enumerate(range(slice.shape[1])):
        value = slice[:, col]
        black_count = np.sum(value < 10)
        black_pixels.append((idx, black_count))

    left_average_slope = calculate_average_slope(
        black_pixels[: int(len(black_pixels) / 2)]
    )
    right_average_slope = calculate_average_slope(
        black_pixels[int(len(black_pixels) / 2) :]
    )

    if (left_average_slope > 0) and (right_average_slope < 0):
        return True
    return False


def remove_extra_pixel(slice, bb_tuple):
    black_pixels = []

    for _, row in enumerate(range(slice.shape[0])):
        value = slice[row, :]
        black_count = np.sum(value < 10)
        black_pixels.append(black_count)

    max_black_row = black_pixels.index(max(black_pixels))

    slice[: max_black_row - int(bb_tuple[3] / 2), :] = 255
    slice[max_black_row + int(bb_tuple[3] / 2) :, :] = 255

    return slice


def sliding_window_match(filtered_img, bb_tuple, reference_image):
    _, ref_width, _ = reference_image.shape

    thresh_image = np.where(filtered_img < 250, 0, 255).astype("uint8")
    blurred_image = cv2.blur(thresh_image, (5, 5), 0)
    edged = cv2.Canny(blurred_image.copy(), 50, 150, apertureSize=3)

    # color based error bars detection
    gray = thresh_image.copy()

    thresh_direct = np.where(gray < 240, 1, 0).astype("uint8")

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh_blur = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    def extract_contour(thresh_image, result):
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (7, 1)
        )  # optimum value 5 to 7
        detect_horizontal = cv2.morphologyEx(
            thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
        )
        # show_img(detect_horizontal, gray = True)

        cnts = cv2.findContours(
            detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # point are in (width, height) position fashion
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        ouput_contour = []

        for c in cnts:
            length_contour = cal_contour_length(c)
            height_contour = cal_contour_height(c)
            # height.append(height_contour)

            if (length_contour < 50) and height_contour < 10:  # 50,5
                ouput_contour.append(c)
                # if plot:
                cv2.drawContours(result, [c], -1, (0, 0, 255), 1)

        return result

    mask = np.ones_like(reference_image) * 255
    remove_error_bar = extract_contour(thresh_direct, mask)
    remove_error_bar = extract_contour(thresh_blur, remove_error_bar)
    mask_eror_bars = cv2.cvtColor(remove_error_bar, cv2.COLOR_BGR2GRAY)
    mask_eror_bars = np.where(mask_eror_bars < 240, 0, 255).astype("uint8")

    # data points connecting lines detection and removal
    lsd = cv2.createLineSegmentDetector(
        refine=cv2.LSD_REFINE_STD,
        # cv2.LSD_REFINE_NONE
        # log_eps = 1,
    )
    lines = lsd.detect(edged)[0]

    mask = np.ones_like(reference_image) * 255

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = list(map(int, line[0]))
            length_line = line_length(x1, y1, x2, y2)
            short_amount = 10
            # TODO slope = abs(y1-y2)/abs(x1-x2+0.001)
            if length_line <= bb_tuple[3] or length_line <= bb_tuple[2]:
                continue
            line = shorten_line(line, short_amount=short_amount)
            x1, y1, x2, y2 = list(map(int, line[0]))
            random_color = (0, 0, 0)
            cv2.line(mask, (x1, y1), (x2, y2), random_color, thickness=5)

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    line_removed = ~np.bitwise_and(~thresh_image, mask_gray)
    line_removed = ~np.bitwise_and(~line_removed, mask_eror_bars)
    line_removed = cv2.medianBlur(line_removed, 3)

    # Loop to move the slice across the image horizontally
    slice_window_width = bb_tuple[2]
    _, img_width = line_removed.shape

    data_points = []
    stride = 1
    for x in range(0, img_width - slice_window_width + 1, stride):
        slice_skip = False
        for point in data_points:
            distance_center = abs(point[0] - (x + bb_tuple[2] / 2))
            if distance_center <= bb_tuple[2] * (2 / 3):
                slice_skip = True
                break

        if slice_skip:
            continue

        sliced_window = line_removed[:, x : x + slice_window_width].copy()
        sliced_window = remove_extra_pixel(sliced_window, bb_tuple)
        # Count black pixel
        black_pixels = []
        for row in range(sliced_window.shape[0]):
            for col in range(sliced_window.shape[1]):
                value = sliced_window[row, col]
                if value == 0:
                    black_pixels.append((row, col))

        # Localize the densed black area
        if len(black_pixels) > 5 and normal_distribution(sliced_window):
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(black_pixels)
            cluster_centers = kmeans.cluster_centers_
            cluster_centers = np.copy(cluster_centers).tolist()

            dominant_point = cluster_centers[0]
            dominant_point = list(map(int, dominant_point))

            data_points.append([x + dominant_point[1], dominant_point[0]])

    bb_boxes = [
        (
            point[0] - int(bb_tuple[2] / 2),
            point[1] - int(bb_tuple[3] / 2),
            point[0] + int(bb_tuple[2] / 2),
            point[1] + int(bb_tuple[3] / 2),
            1,
        )
        for point in data_points
    ]

    return bb_boxes


def template_match_iou(url, bb_tuple, axes_information, intersection):
    """
    Matches a template image within a reference image using a IoU method.

    Args:
        figure_url (str): The URL of the reference image.
        bb_tuple (tuple): tuple of bounding box cordinates
        (x_top_left, y_top_left, width, height)

    Returns:
        filtered_box (list[Tuple]): List of template matched bounding boxes
        in this format (x_top_left, y_top_left, width, height)
    """
    try:
        reference_image = get_image_from_url(url, False)
        reference_image = np.frombuffer(reference_image, np.uint8)
        reference_image = cv2.imdecode(reference_image, cv2.IMREAD_COLOR)

        # color filter
        crop_bb = {
            "x": bb_tuple[0],
            "y": bb_tuple[1],
            "width": bb_tuple[2],
            "height": bb_tuple[3],
        }
        cropped_template_img = crop_image(reference_image.copy(), crop_bb)

        # cropeed image to only focus on the main color
        color_focused = focus_to_color(bb_tuple, reference_image.copy())
        # HSV based color filtering
        color, _ = find_dominant_color(color_focused)
        filtered_img = filter_by_color(reference_image.copy(), color)
        sliding_window_boxes = sliding_window_match(
            filtered_img, bb_tuple, reference_image
        )

        contour_image = draw_template_contour(
            cropped_template_img.copy(), color, plot=False
        )
        flood_img = template_flood_fill(contour_image.copy(), plot=False)
        overlapped_area = cal_overlap(
            cropped_template_img.copy(), color, flood_img.copy()
        )

        template_gray = filter_by_color(cropped_template_img.copy(), color)
        template_binary = np.where(template_gray < 240, 1, 0)

        filtered_binary = np.where(filtered_img < 240, 1, 0)

        overlapped_threshold = 0.95

        # bw check
        _, bw_image = find_dominant_color(reference_image.copy(), print_dominant=True)
        if bw_image:
            offset = 1
            crop_bb = {
                "x": bb_tuple[0],
                "y": bb_tuple[1],
                "width": bb_tuple[2] + offset,
                "height": bb_tuple[3] + offset,
            }
            cropped_template_img = crop_image(reference_image.copy(), crop_bb)

        if overlapped_area < overlapped_threshold:
            filtered_box = match_hollow(
                cropped_template_img.copy(),
                reference_image.copy(),
                flood_img.copy(),
                bw_image=bw_image,
            )
        else:
            if bw_image:
                tick_based_boxes = []
            else:
                tick_based_boxes = tick_based_marker_detect(
                    axes_information,
                    intersection,
                    reference_image,
                    filtered_img,
                    bb_tuple,
                )
            filtered_box = iou_match(
                template_binary.copy(),
                filtered_binary.copy(),
                hollow=False,
                bw_image=bw_image,
                tick_based_boxes=tick_based_boxes,
                sliding_window_boxes=sliding_window_boxes,
            )

        return filtered_box

    except Exception:
        print_exc()
        bounding_boxes = []
        return bounding_boxes


def cal_contour_length(c):
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


def cal_contour_height(c):
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


def load_image(url):
    reference_image = cv2.imread(url)
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    return reference_image


def detect_error_bars(url, local_source=False):
    """
    Detects all the error bars in a given given
    (image from the url)
    Args:
    url (str): link to the image to be detected
    local_source (bool): Whether the image should be
                        locally or in sever

    Returns:
    ouput_contour (list[list[(int,int)]]): List of contours of the
            detected contours.
    """
    if local_source:
        ref_image = cv2.imread(url)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    else:
        ref_image = get_image_from_url(url, False)
        ref_image = np.frombuffer(ref_image, np.uint8)
        ref_image = cv2.imdecode(ref_image, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255,
    #    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh_direct = np.where(gray < 240, 1, 0).astype("uint8")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_blur = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    def extract_contour(thresh_image):
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (7, 1)
        )  # optimum value 5 to 7
        detect_horizontal = cv2.morphologyEx(
            thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
        )

        cnts = cv2.findContours(
            detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # point are in (width, height) position fashion
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        ouput_contour = []

        for c in cnts:
            length_contour = cal_contour_length(c)
            height_contour = cal_contour_height(c)
            # height.append(height_contour)

            if (length_contour < 50) and height_contour < 10:  # 50,5
                ouput_contour.append(c)

        return ouput_contour

    ouput_contour_1 = extract_contour(thresh_direct)
    ouput_contour_2 = extract_contour(thresh_blur)

    return ouput_contour_1 + ouput_contour_2


def cal_contour_center(contour):
    """
    Calculate center of a given contour.
    Args:
    contour (list[(int,int)]): Input contour

    Returns:
    (tuple(int,int)): Center of the given contuor.
    """
    all_horizontal_end = contour[:, 0, 0]
    start_horizontal = min(all_horizontal_end)
    end_horizontal = max(all_horizontal_end)
    cX = (end_horizontal + start_horizontal) / 2

    all_horizontal_end = contour[:, 0, 1]
    start_horizontal = min(all_horizontal_end)
    end_horizontal = max(all_horizontal_end)
    cY = (end_horizontal + start_horizontal) / 2

    return (int(cX), int(cY))


def cal_bounding_box_center(bb_box):
    """
    Calculate center of a given bounding box.
    Args:
    contour (tuple(x_top, y_top, width, height)): Input bounding box

    Returns:
    (tuple(int,int)): Center of the given bounding box.
    """
    cX = int(bb_box[0] + bb_box[2] / 2)
    cY = int(bb_box[1] + bb_box[3] / 2)

    return (cX, cY)


def check_equal(distance_1, distance_2):
    if abs(distance_1 - distance_2) < 5:
        return True
    return False


def check_in_range(box_width, distance_1, distance_2):
    multiplier = 5
    if distance_1 > multiplier * box_width:
        return False
    if distance_2 > multiplier * box_width:
        return False
    return True


def filter_error_points(contour_error_bars, bb_boxes, bb_tuple, ref_image):
    """
    Attach detected error points to their corresponding datapoints.
    Args:
    contour_error_bars (list[list[(int,int)]]): List of contours of
                    the detected datapoint.
    bb_boxes (list(tuple(x_top, y_top, width, height))): List of
                    bounding boxes of the data points.
    bb_tuple tuple (int, int, int, int): Bounding box of the user given
        template box.

    Returns:
    data_list (list(int, int, int, int, int ,int ,int)): Returns
                detected bounding boxes in the following format,
                (x_top, y_top, height, width, deviationPixelDistance,
                topBarPixelDistance, bottomBarPixelDistance)
    """

    # Convert Line to Poly

    # Convert Error bar detection contour to point
    error_points = [cal_contour_center(contour) for contour in contour_error_bars]

    # Convert detected data point boxes to center point
    bounding_box_center_points = [cal_bounding_box_center(box) for box in bb_boxes]

    # for center in bounding_box_center_point:
    #     cv2.circle(ref_image, center, 7, (255, 0, 0), -1)
    # show_img(ref_image)

    # Apply sliding window
    window_width = bb_boxes[0][2]  # (x_top_left, y_top_left, width, height)
    window_height = bb_boxes[0][3]

    # Track the assigned data points
    assigned_bb_box = []

    _, ref_image_width = ref_image.shape[:2]  # (height, width)

    # width_list = list(range(0, ref_image_width, window_width))
    # if ref_image_width not in width_list:
    #     width_list.append(ref_image_width)

    start_width = 0

    data_list = []
    while True:
        for center_idx, bb_center in enumerate(bounding_box_center_points):
            if (bb_boxes[center_idx] not in assigned_bb_box) and (
                start_width <= bb_center[0] <= start_width + window_width
            ):
                # calculate vertical distance of error bars
                upper_error_bars = []
                lower_error_bars = []

                # split into top and bottom error bars
                for error_point in error_points:
                    # Check whether it is in the current sliding window
                    condition = (
                        start_width <= error_point[0] <= start_width + window_width
                        and abs(error_point[1] - bb_center[1]) > 2 + window_height / 2
                    )

                    if condition:
                        # selected_error_bars.append(error_point)
                        diff = error_point[1] - bb_center[1]
                        if diff < 0:
                            upper_error_bars.append(error_point)

                        else:
                            lower_error_bars.append(error_point)

                if len(upper_error_bars) == 0 or len(lower_error_bars) == 0:
                    upper_error_bars = []
                    lower_error_bars = []

                    for error_point in error_points:
                        condition = (
                            min(start_width, abs(start_width - window_width * 1.5))
                            <= error_point[0]
                            <= start_width + window_width * 1.5
                        ) and abs(error_point[1] - bb_center[1]) > 2 + window_height / 2

                        if condition:
                            diff = error_point[1] - bb_center[1]
                            if diff < 0:
                                upper_error_bars.append(error_point)
                            else:
                                lower_error_bars.append(error_point)

                # select the error bars that are at equal distance
                selected_error_bars = []
                if len(upper_error_bars) != 0 and len(lower_error_bars) != 0:
                    for point_top in upper_error_bars:
                        topBarPixelDistance = abs(point_top[1] - bb_center[1])
                        for point_bottom in lower_error_bars:
                            bottomBarPixelDistance = abs(point_bottom[1] - bb_center[1])
                            if check_equal(
                                topBarPixelDistance, bottomBarPixelDistance
                            ) and check_in_range(
                                bb_boxes[center_idx][3],
                                topBarPixelDistance,
                                bottomBarPixelDistance,
                            ):
                                bar_pair = {
                                    "topBarPixelDistance": topBarPixelDistance,
                                    "bottomBarPixelDistance": bottomBarPixelDistance,
                                    "distance": max(
                                        bottomBarPixelDistance, topBarPixelDistance
                                    ),
                                }
                                selected_error_bars.append(bar_pair)
                                assigned_bb_box.append(bb_boxes[center_idx])

                if len(selected_error_bars) != 0:
                    selected_error_bars = sorted(
                        selected_error_bars, key=lambda x: x["distance"]
                    )
                    selected_error_bar_pair = selected_error_bars[-1]
                    topBarPixelDistance = selected_error_bar_pair["topBarPixelDistance"]
                    bottomBarPixelDistance = selected_error_bar_pair[
                        "bottomBarPixelDistance"
                    ]

                    deviationPixelDistance = max(
                        topBarPixelDistance, bottomBarPixelDistance
                    )

                    point_dis_data = bb_boxes[center_idx] + (
                        deviationPixelDistance,
                        topBarPixelDistance,
                        bottomBarPixelDistance,
                    )
                    data_list.append(point_dis_data)

                else:
                    # print(len(upper_error_bars))
                    # print(len(lower_error_bars))
                    point_dis_data = bb_boxes[center_idx] + (
                        int(bb_boxes[center_idx][3] / 2),
                        int(bb_boxes[center_idx][3] / 2),
                        int(bb_boxes[center_idx][3] / 2),
                    )
                    data_list.append(point_dis_data)

        start_width += window_width
        # Add condition for unevenly distribution
        # if (ref_image_width - start_width) < window_width:
        if start_width > ref_image_width:
            break

    return data_list


def detect_bb_box_error_bar(url, bb_tuple, axes_information, intersection):
    """
    Detects bounding box for the user given template marker,
    with their correspoing error bars if present.

    Args:
    url (str): Link to the image
    bb_tuple (int, int, int, int): Bounding box for the user
        given template marker int the format (x_top, y_top,
        width, height)

    Returns:
    data_list (list(int, int, int, int, int ,int ,int)): Returns
                detected bounding boxes in the following format,
                (x_top, y_top, height, width, deviationPixelDistance,
                topBarPixelDistance, bottomBarPixelDistance)

    """
    try:
        # Load image
        ref_image = get_image_from_url(url, False)
        ref_image = np.frombuffer(ref_image, np.uint8)
        ref_image = cv2.imdecode(ref_image, cv2.IMREAD_COLOR)

        contour_error_bars = detect_error_bars(url)
        bb_boxes = template_match_iou(url, bb_tuple, axes_information, intersection)
        data_list = filter_error_points(
            contour_error_bars, bb_boxes, bb_tuple, ref_image
        )

    except Exception:
        print_exc()
        data_list = []
        return data_list

    return data_list
