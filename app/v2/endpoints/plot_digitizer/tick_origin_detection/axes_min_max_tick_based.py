import copy

import cv2
import numpy as np


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


def detect_x_y_ticks(main_image, result, intersection):
    """
    Correct min max of the axes based on the tick marks
    Args:
    main_image (numpy.ndarray): input plot image
    result (dict): dictionary containing values of the min max
    intersection (tuple): (x,y) coordinate of the origin

    Returns:
    result (dict): updated dictionary containing values of the min max
    """
    height, _, _ = main_image.shape

    def flip_vertical(point, height):
        return (point[0], abs(height - point[1]))

    FLIP = False
    if intersection[1] < height / 2:
        FLIP = True
        main_image = cv2.flip(main_image, 0)
        intersection = flip_vertical(intersection, height)
        for data in result["points"]:
            # flip the y value of the point only
            data["y"] = int(abs(height - data["y"]))

    ref_image = main_image.copy()
    # remove x axis line
    gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    thresh_image = np.where(gray < 240, 1, 0).astype("uint8")
    thresh_vertical = thresh_image.copy()
    thresh_vertical[: intersection[1] - 10, :] = 0

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    detect_horizontal = cv2.morphologyEx(
        thresh_vertical, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
    )
    cnts_horizontal = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts_horizontal = (
        cnts_horizontal[0] if len(cnts_horizontal) == 2 else cnts_horizontal[1]
    )

    # max_contour_idx = len_list.index(max(len_list))
    for c in cnts_horizontal:
        # height = cal_contour_height(c)
        cv2.drawContours(ref_image, [c], -1, (255, 255, 255), thickness=cv2.FILLED)

    # check upside down axis
    gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    thresh_image = np.where(gray < 240, 1, 0).astype("uint8")

    thresh_vertical = thresh_image.copy()
    thresh_vertical[: intersection[1] - 10, :] = 0

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 2)
    )  # optimum value 5 to 7

    detect_vertical = cv2.morphologyEx(
        thresh_vertical, cv2.MORPH_OPEN, vertical_kernel, iterations=1
    )

    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # point are in (width, height) position fashion
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    x_ticks_centroids = []

    cnts_sorted = []

    ymax_y = None
    xmax_x = None
    for data in result["points"]:
        if FLIP:
            if data["label"] == "ymin":
                ymax_y = int(data["y"])
        else:
            if data["label"] == "ymax":
                ymax_y = int(data["y"])
        if data["label"] == "xmax":
            xmax_x = int(data["x"])

    lower_range = 12
    break_loop = False
    while True:
        for c in cnts:
            (cx, cy) = cal_contour_center(c)
            # add condition between 12 and 20
            if (intersection[1] <= cy <= intersection[1] + lower_range) and (
                intersection[0] <= cx <= xmax_x + 1
            ):  # condition for xmax
                cnts_sorted.append(c)
                x_ticks_centroids.append((cx, cy))
                # cv2.drawContours(test_result, [c], -1, (255,0,0), 5)

        cnts_sorted = sorted(cnts_sorted, key=lambda x: cal_contour_center(x)[0])
        if break_loop:
            break
        if len(cnts_sorted) == 0:
            break_loop = True
            lower_range = 20
            continue
        break

    # check number in the bottom of origin
    offset_origin = 20
    x_intersection_temp = copy.copy(intersection[0])
    y_intersection_temp = copy.copy(intersection[1]) + offset_origin
    # make sure box_size < offset_origin
    box_size = 5
    temp_gray = gray.copy()
    box_image = temp_gray[
        y_intersection_temp - box_size : y_intersection_temp + box_size,
        x_intersection_temp - box_size : x_intersection_temp + box_size,
    ]
    black_count = np.sum(box_image != 255)

    # Update x cord of x min
    if black_count > 0:
        if FLIP:
            for data in result["points"]:
                if data["label"] == "xmin":
                    data["y"] = int(abs(height - data["y"]))

    else:
        for data in result["points"]:
            if data["label"] == "xmin":
                data["x"] = int(cal_contour_center(cnts_sorted[0])[0])
                if FLIP:
                    data["y"] = int(abs(height - data["y"]))

    # Update x cord of x max
    for data in result["points"]:
        if data["label"] == "xmax":
            data["x"] = int(cal_contour_center(cnts_sorted[-1])[0])
            if FLIP:
                data["y"] = int(abs(height - data["y"]))

    # --------------------------------------y ticks detection-----------------#
    ref_image = main_image.copy()
    # remove y axis line
    gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    thresh_image = np.where(gray < 240, 1, 0).astype("uint8")
    thresh_vertical = thresh_image.copy()
    thresh_vertical[:, intersection[0] + 8 :] = 0

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    detect_vertical = cv2.morphologyEx(
        thresh_vertical, cv2.MORPH_OPEN, vertical_kernel, iterations=1
    )
    cnts_vertical = cv2.findContours(
        detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts_vertical = cnts_vertical[0] if len(cnts_vertical) == 2 else cnts_vertical[1]

    for c in cnts_vertical:
        cv2.drawContours(ref_image, [c], -1, (255, 255, 255), thickness=cv2.FILLED)

    # detect ticks
    gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    thresh_image = np.where(gray < 240, 1, 0).astype("uint8")
    thresh_horizontal = thresh_image.copy()
    thresh_horizontal[:, intersection[0] + 8 :] = 0

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2, 1)
    )  # optimum value 5 to 7
    detect_horizontal = cv2.morphologyEx(
        thresh_horizontal, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
    )

    cnts = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # point are in (width, height) position fashion
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    x_ticks_centroids = []

    cnts_sorted = []

    left_range = 12
    break_loop = False
    while True:
        for c in cnts:
            (cx, cy) = cal_contour_center(c)
            # if ((intersection[1] <= cy <= intersection[1] + 12)
            # and cx >= intersection[0]): # condition for xmax
            if (intersection[0] - left_range <= cx <= intersection[0]) and (
                ymax_y - 5 <= cy <= intersection[1] + 12
            ):
                cnts_sorted.append(c)
                x_ticks_centroids.append((cx, cy))
                # cv2.drawContours(test_result, [c], -1, (255,0,0), 5)

        cnts_sorted = sorted(cnts_sorted, key=lambda x: cal_contour_center(x)[1])
        if break_loop:
            break
        if len(cnts_sorted) == 0:
            break_loop = True
            left_range = 20
            continue
        break

    # update y co-ordinate of ymin
    for data in result["points"]:
        if data["label"] == "ymin":
            if FLIP:
                temp_point = cal_contour_center(cnts_sorted[0])
                temp_point = flip_vertical(temp_point, height)
                data["y"] = int(temp_point[1])
            else:
                data["y"] = int(cal_contour_center(cnts_sorted[-1])[1])

    # update y co-ordinate of ymax
    for data in result["points"]:
        if data["label"] == "ymax":
            if FLIP:
                temp_point = cal_contour_center(cnts_sorted[-1])
                temp_point = flip_vertical(temp_point, height)
                data["y"] = int(temp_point[1])
            else:
                data["y"] = int(cal_contour_center(cnts_sorted[0])[1])

    return result
