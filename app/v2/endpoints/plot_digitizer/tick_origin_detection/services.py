import cv2
import numpy as np

from app.v2.endpoints.plot_digitizer.utils import get_image_from_url


def crop_image(image, bounding_box):
    """
    Crop an image based on the provided bounding box coordinates.

    Args:
        image (numpy.ndarray): The input image to be cropped.
        bounding_box (dict): A dictionary containing the bounding box coordinates.
            - 'x': The x-coordinate of the top-left corner of the bounding box.
            - 'y': The y-coordinate of the top-left corner of the bounding box.
            - 'width': The width of the bounding box.
            - 'height': The height of the bounding box.

    Returns:
        numpy.ndarray: The cropped image.
    """
    y_start = bounding_box["y"]
    y_end = y_start + bounding_box["height"]
    x_start = bounding_box["x"]
    x_end = x_start + bounding_box["width"]

    cropped_image = image[y_start:y_end, x_start:x_end]
    return cropped_image


def y_gap_filter(black_pixel_counts, gap_threshold=10):
    """
    Detects gap in counts to know when to stop.

    Args:
        black_pixel_counts (list): List of pixel counts.
        gap_threshold (int): Threshold for the gap size (default is 10).

    Returns:
        list: List of counts with counts beyond gap flatted to 0

    """
    black_pixel_list = black_pixel_counts.tolist()

    # Iterate through points starting from origin to identify the first gap
    white_pixel_count = 0
    gap_stop_index = None

    for index, pixel_count in enumerate(reversed(black_pixel_list)):
        if pixel_count == 0:
            white_pixel_count += 1
        else:
            white_pixel_count = 0

        if white_pixel_count > gap_threshold:
            gap_stop_index = len(black_pixel_list) - index
            break

    if gap_stop_index is None:
        return black_pixel_list

    # Create a new list with pixels beyond the gap set to 0
    gap_filtered_list = [0] * gap_stop_index + black_pixel_list[gap_stop_index:]

    return gap_filtered_list


def x_gap_filter(black_pixel_counts, gap_threshold=10):
    """
    Detects gap in counts to know when to stop.

    Args:
        black_pixel_counts (list): List of pixel counts.
        gap_threshold (int): Threshold for the gap size (default is 10).

    Returns:
        list: List of counts with counts beyond gap flatted to 0

    """
    black_pixel_list = black_pixel_counts.tolist()

    # Iterate through points starting from origin to identify the first gap
    white_pixel_count = 0
    gap_stop_index = None

    for index, pixel_count in enumerate(black_pixel_list):
        if pixel_count == 0:
            white_pixel_count += 1
        else:
            white_pixel_count = 0

        if white_pixel_count > gap_threshold:
            gap_stop_index = index
            break

    # If there are no gaps found then return original list
    if gap_stop_index is None:
        return black_pixel_list

    # Create a new list with pixels beyond the gap set to 0
    gap_filtered_list = black_pixel_list[0:gap_stop_index] + [0] * (
        len(black_pixel_list) - gap_stop_index
    )

    return gap_filtered_list


def threshold_filter(scaled_counts, threshold):
    threshold_list = [
        index
        for index, scaled_count in enumerate(scaled_counts)
        if scaled_count > threshold
    ]

    return threshold_list


def find_row_with_most_active_pixels(image, threshold=50):
    """
    Converts an image to binary based on a threshold and identifies
    the row with the most '1's (active pixels).

    :param image: The input image as a numpy array.
    :param threshold: The threshold for binarization (default is 128).
    :return: A tuple containing:
        - The index of the row with the most '1's.
        - The count of '1's in that row.
    """
    # Convert to grayscale if the image is colored
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply threshold to convert image to binary
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to binary format in terms of 0s and 1s for easier counting
    binary_image = (binary_image > 0).astype(int)

    # Count the number of '1's in each row
    counts = np.sum(binary_image, axis=1)
    best_row = np.argmin(counts)

    return best_row


def find_column_with_most_active_pixels(image, threshold=40):
    """
    Converts an image to binary based on a threshold and identifies
    the column with the most '1's (active pixels).

    :param image: The input image as a numpy array.
    :param threshold: The threshold for binarization (default is 20).
    :return: A tuple containing:
        - The index of the column with the most '1's.
        - The count of '1's in that column.
    """
    # Convert to grayscale if the image is colored
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply threshold to convert image to binary
    # _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    binary_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Convert to binary format in terms of 0s and 1s for easier counting
    binary_image = (binary_image > 0).astype(int)

    # Count the number of '1's in each column
    counts = np.sum(binary_image, axis=0)
    best_column = np.argmin(counts)

    return best_column


def calculate_x_tick_marks(
    figure_img, global_origin, crop_threshold=0.03, tick_detection_threshold=0.3
):
    """
    Calculate Tick marks on the X axis

    Args:
        figure_img (numpy.ndarray): The input figure image as a NumPy array.
        global_origin (dict): A dictionary containing the global row and column values.
        crop_threshold (float,optional): % Threshold to crop X axis image
        from below origin
        tick_detection_threshold (float, optional): The threshold value
        for detecting scaled tick marks.

    Returns:
        list: A list of global X positions of the detected tick marks.
    """

    global_row = global_origin["global_row"]
    global_column = global_origin["global_column"]

    y_crop_distance = int(figure_img.shape[0] * crop_threshold)
    y_start = max(0, global_row)
    y_end = min(figure_img.shape[0], global_row + y_crop_distance)
    y_cropped_img = figure_img[y_start:y_end, global_column:]

    grayscale_image = cv2.cvtColor(y_cropped_img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    black_pixel_counts = np.sum(binary_image == 0, axis=0)

    # Identify any gaps and filter out any pixels beyond the gap
    gap_filtered_ct = x_gap_filter(black_pixel_counts)

    max_count = np.max(gap_filtered_ct)
    scaled_counts = gap_filtered_ct / max_count if max_count > 0 else gap_filtered_ct

    # Tick Detection
    horizontal_tick_threshold_list = threshold_filter(
        scaled_counts, tick_detection_threshold
    )

    x_pos_list_ct = len(horizontal_tick_threshold_list)
    while x_pos_list_ct > 50:
        tick_detection_threshold += 0.05
        horizontal_tick_threshold_list = threshold_filter(
            scaled_counts, tick_detection_threshold
        )
        x_pos_list_ct = len(horizontal_tick_threshold_list)

    # Global positioning
    global_x_pos_list = [
        x_pos + global_column for x_pos in horizontal_tick_threshold_list
    ]

    return global_x_pos_list


def calculate_y_tick_marks(
    figure_img, global_origin, crop_threshold=0.01, tick_detection_threshold=0.4
):
    """
    Calculate tick marks in the Y (vertical axis)

    Args:
        figure_img (numpy.ndarray): The input figure image as a NumPy array.

        global_origin (dict): A dictionary containing the global row and column
        values.

        crop_threshold (float,optional): Threshold on cropping image

        tick_detection_threshold (float, optional): The threshold
        value for detecting tick marks.

    Returns:
        list: A list of global X positions of the detected tick marks.
    """

    global_row = global_origin["global_row"]
    global_column = global_origin["global_column"]

    # Crop and turn picture to binary
    x_crop_distance = int(figure_img.shape[1] * crop_threshold)
    x_start = global_column - x_crop_distance
    x_cropped_img = figure_img[0:global_row, x_start : global_column + 10]

    grayscale_image = cv2.cvtColor(x_cropped_img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Count all the pixels in the row if they are black
    black_pixel_counts = np.sum(binary_image == 0, axis=1)

    # Identify any gaps and filter out any pixels beyond the gap
    gap_filtered_ct = y_gap_filter(black_pixel_counts)

    max_count = np.max(gap_filtered_ct)
    scaled_counts = gap_filtered_ct / max_count if max_count > 0 else gap_filtered_ct

    # Initial Tick Detection
    vertical_tick_threshold_list = threshold_filter(
        scaled_counts, tick_detection_threshold
    )

    # Iteratively increase sensitivity if too many detected tickmarks
    y_pos_list_ct = len(vertical_tick_threshold_list)
    while y_pos_list_ct > 50:
        tick_detection_threshold += 0.05
        vertical_tick_threshold_list = threshold_filter(
            scaled_counts, tick_detection_threshold
        )
        y_pos_list_ct = len(vertical_tick_threshold_list)

    global_y_pos_list = list(vertical_tick_threshold_list)

    return global_y_pos_list


def find_axes_intersection_advanced(image, debug=False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    edges = cv2.Canny(thresh, 30, 120)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(
        dilated, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=5
    )  # More sensitive settings

    vertical_lines = []
    horizontal_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(x1 - x2) < 5:  # Tighter vertical line check
                    vertical_lines.append((x1, y1, x2, y2))
                    if debug:
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                elif abs(y1 - y2) < 5:  # Tighter horizontal line check
                    horizontal_lines.append((x1, y1, x2, y2))
                    if debug:
                        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    if vertical_lines:
        vertical_line = sorted(
            vertical_lines, key=lambda line: abs(line[1] - line[3]), reverse=True
        )[0]
    else:
        vertical_line = None

    if horizontal_lines:
        horizontal_line = sorted(
            horizontal_lines, key=lambda line: abs(line[0] - line[2]), reverse=True
        )[0]
    else:
        horizontal_line = None

    intersection_point = None

    if vertical_line and horizontal_line:
        vx1, vy1, vx2, vy2 = vertical_line
        hx1, hy1, hx2, hy2 = horizontal_line
        d = (vx1 - vx2) * (hy1 - hy2) - (vy1 - vy2) * (hx1 - hx2)
        if d:
            xi = (
                (vx1 * vy2 - vy1 * vx2) * (hx1 - hx2)
                - (vx1 - vx2) * (hx1 * hy2 - hy1 * hx2)
            ) / d
            yi = (
                (vx1 * vy2 - vy1 * vx2) * (hy1 - hy2)
                - (vy1 - vy2) * (hx1 * hy2 - hy1 * hx2)
            ) / d
            intersection_point = (int(xi), int(yi))
            if debug:
                cv2.circle(image, intersection_point, 5, (0, 0, 255), -1)

    return intersection_point


def origin_based_axes_detection(
    figure_url: str, x: int, y: int, width: int, height: int
):
    """
    Given a bounding box around the origin in reference image,
    calculates the x and y tick positions

    Args:
        figure_url (str): The URL of the reference image.
        x (int): The x-coordinate of the top-left corner of the bounding box.
        y (int): The y-coordinate of the top-left corner of the bounding box.
        width (int): The width of the bounding box.
        height (int): The height of the bounding box.

    Return:
        dict: Dictionary of pixel position of x min/max and y min/max

    Raises:
        Any exceptions that may occur during image processing.

    """
    # Origin bounding box
    origin_bounding_box = {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }

    figure_img = get_image_from_url(figure_url, False)
    figure_img = np.frombuffer(figure_img, np.uint8)
    figure_img = cv2.imdecode(figure_img, cv2.IMREAD_COLOR)

    cropped_image = crop_image(figure_img, origin_bounding_box)

    vertical_fit = find_column_with_most_active_pixels(cropped_image)
    horizontal_fit = find_row_with_most_active_pixels(cropped_image)

    global_column = x + vertical_fit
    global_row = y + horizontal_fit

    global_origin = {"global_column": global_column, "global_row": global_row}

    # Iterative Sensitivity testing
    global_x_pos_list = calculate_x_tick_marks(figure_img, global_origin, 0.02, 0.3)

    global_y_pos_list = calculate_y_tick_marks(figure_img, global_origin, 0.01, 0.3)

    if not global_x_pos_list:
        raise ValueError("No X tick marks detected.")

    if not global_y_pos_list:
        raise ValueError("No Y tick marks detected.")

    x_min_x = int(global_column)
    x_min_y = int(global_row)
    x_max_x = int(global_x_pos_list[-1])
    x_max_y = int(global_row)
    y_min_x = int(global_column)
    y_min_y = int(global_row)
    y_max_x = int(global_column)
    y_max_y = int(global_y_pos_list[0])

    output = {
        "points": [
            {
                "label": "xmin",
                "x": int(x_min_x),
                "y": int(x_min_y),
            },
            {
                "label": "xmax",
                "x": int(x_max_x),
                "y": int(x_max_y),
            },
            {
                "label": "ymin",
                "x": int(y_min_x),
                "y": int(y_min_y),
            },
            {
                "label": "ymax",
                "x": int(y_max_x),
                "y": int(y_max_y),
            },
        ]
    }

    return output


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


def crop_lower_intersection(image):
    """
    Given an input image, it will return the
    origin focusing on the lower left part
    of the image.
    Args:
    image (numpy.ndarray): input image

    Return:
    intersection (tuple): (x,y) co-ordinate
    of the origin
    """
    temp_img = image.copy()
    image_shape = temp_img.shape
    height = image_shape[0]
    width = image_shape[1]
    # Divide the image into equal four parts
    # Extract the lower left part
    cropped_image = temp_img[int(height / 2) : height, 0 : int(width / 2), :]
    x, y = find_axes_intersection_advanced(cropped_image)
    intersection = (x, int(y + height / 2))
    return intersection


def morphological_detection(ref_image):
    """
    Detect origin and x_max, y_max based on morphological operation

    Args:
    ref_image (numpy.ndarray): input image

    Return:
    morph_origin (tuple): (x,y) value of the origin
    horizontal_right_x (int): x_max value of the axes
    vertical_top_y (int): y_max value of the axes
    horizontal_left_x (int): x_min value of the axes
    vertical_bottom_y (int): y_min value of the axes
    """
    image_shape = ref_image.shape
    height = image_shape[0]
    width = image_shape[1]

    gray = cv2.cvtColor(ref_image.copy(), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # show_img(thresh, gray=True)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (70, 1)
    )  # optimum value 5 to 7
    detect_horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 70)
    )  # optimum value 5 to 7
    detect_vertical = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1
    )
    # show_img(detect_vertical)
    cnts = cv2.findContours(
        detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    cnts = sorted(cnts, key=cal_contour_height, reverse=True)

    # for y axis callibaration
    temp_point_x = []
    temp_point_y = []

    for contour in cnts:
        for point in contour:
            point = point[0]
            # print(point[1])
            temp_point_x.append(point[0])
            temp_point_y.append(point[1])

        vertical_x = min(temp_point_x)
        vertical_top_y = min(temp_point_y)
        vertical_bottom_y = max(temp_point_y)
        if vertical_x < width / 2:
            break
    # vertical_bottom_y = max(temp_point_y)

    # for x axis callibaration
    cnts = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    cnts = sorted(cnts, key=cal_contour_length, reverse=True)
    max_length = cal_contour_length(cnts[0])

    temp_point_x = []
    temp_point_y = []

    for contour in cnts:
        contour_length = cal_contour_length(contour)
        if contour_length > max_length * 0.75:
            for point in contour:
                point = point[0]
                temp_point_x.append(point[0])
                temp_point_y.append(point[1])

            if max(temp_point_y) > height / 2 or contour_length == max_length:
                horizontal_y = max(temp_point_y)
                horizontal_right_x = max(temp_point_x)
                horizontal_left_x = min(temp_point_x)
                break

    morph_origin = (vertical_x, horizontal_y)

    return (
        morph_origin,
        horizontal_right_x,
        vertical_top_y,
        horizontal_left_x,
        vertical_bottom_y,
    )
