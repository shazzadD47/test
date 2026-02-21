import colorsys

import cv2
import numpy as np
from sklearn.cluster import KMeans


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


def focus_to_color(center_x, center_y, reference_image):

    square_length = 4
    x_top = int(center_x - square_length)
    y_top = int(center_y - square_length)

    crop_bb = {"x": x_top, "y": y_top, "width": square_length, "height": square_length}
    return crop_image(reference_image.copy(), crop_bb)


def find_dominant_color(cv_image, n_clusters=4, print_dominant=False):
    """
    Finds the dominant color in an image using K-means clustering.

    Args:
    cv_image (numpy.ndarray): Input image in RGB format.
    n_clusters (int): Number of clusters to use in K-means.
    Defaults to 1 for dominant color.

    Returns:
    tuple: Dominant color in RGB format.
    """
    # Reshape the image to a 2D array of pixels
    pixels = cv_image.reshape(-1, 3)

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
    big_s = 0
    for center in kmeans.cluster_centers_:
        _, s, _ = rgb_to_hsv(*center.astype(int))
        if s >= big_s:
            color_selected = center.astype(int)
            big_s = s

    return tuple(color_selected), bw_image


def filter_by_color(cv_image, target_color, tolerance=30):
    """
    Filter an image by a specific RGB color within a given tolerance.

    Args:
    cv_image (numpy.ndarray): Input image in RGB format.
    target_color (tuple): Target RGB color as (R, G, B).
    tolerance (int): Tolerance for color matching.

    Returns:
    numpy.ndarray: Filtered image with highlighted areas matching the target color.
    """
    # adjust tolerence value
    _, saturation, _ = rgb_to_hsv(*target_color)
    tolerance = 30 if saturation > 30 else 5

    image = cv_image.copy()

    distances = np.sqrt(((image - np.array(target_color)) ** 2).sum(axis=2))

    mask = distances < tolerance

    mask_bw = (mask * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask_bw, 0, 255, cv2.THRESH_BINARY)

    inverted_binary = cv2.bitwise_not(binary)

    return inverted_binary
