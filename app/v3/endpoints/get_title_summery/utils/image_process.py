import cv2


def compute_gtc(ground_truth, prediction):
    """
    Compute the Ground Truth Coverage (GTC) metric.

    Parameters:
    - ground_truth: Tuple (x_min, y_min, x_max, y_max) for ground truth bounding box.
    - prediction: Tuple (x_min, y_min, x_max, y_max) for predicted bounding box.

    Returns:
    - GTC value.
    """
    # Calculate intersection
    x_min_inter = max(ground_truth[0], prediction[0])
    y_min_inter = max(ground_truth[1], prediction[1])
    x_max_inter = min(ground_truth[2], prediction[2])
    y_max_inter = min(ground_truth[3], prediction[3])

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection = inter_width * inter_height

    # Calculate ground truth area
    gt_width = ground_truth[2] - ground_truth[0]
    gt_height = ground_truth[3] - ground_truth[1]
    gt_area = gt_width * gt_height

    if gt_area == 0:
        return 0.0

    gtc = intersection / gt_area

    return gtc


def crop_image(image_array, bounding_box):
    """
    Crops an image using OpenCV.

    Parameters:
    - image_array: NumPy array representing the image.
    - bounding_box: Tuple or list defining the crop as (x1, y1, x2, y2).

    Returns:
    - Cropped image as a NumPy array.
    """
    x1, y1, x2, y2 = bounding_box

    cropped_image = image_array[y1:y2, x1:x2]

    return cropped_image


def draw_bounding_box(image, box, label, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box with a label on an image.

    :param image: The image on which to draw (numpy array).
    :param box: Tuple of (x1, y1, x2, y2) for the bounding box coordinates.
    :param label: Text label to place at the top of the bounding box.
    :param color: Color of the bounding box and label background (BGR tuple).
    :param thickness: Thickness of the bounding box border.
    """
    x1, y1, x2, y2 = box

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    top_left = (x1, y1 - text_height - 5)
    bottom_right = (x1 + text_width + 10, y1)

    if top_left[1] < 0:
        top_left = (x1, y2 + 5)
        bottom_right = (x1 + text_width + 10, y2 + text_height + 5)

    if bottom_right[0] > image.shape[1]:
        overflow = bottom_right[0] - image.shape[1]
        top_left = (top_left[0] - overflow, top_left[1])
        bottom_right = (bottom_right[0] - overflow, bottom_right[1])

    if bottom_right[1] > image.shape[0]:
        top_left = (
            x1,
            y1 - text_height - 5,
        )
        bottom_right = (x1 + text_width + 10, y1)

    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)

    text_x = top_left[0] + (bottom_right[0] - top_left[0] - text_width) // 2
    text_y = top_left[1] + (bottom_right[1] - top_left[1] + text_height - baseline) // 2

    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )

    return image


def check_overlap(bbox_1: list, bbox_list: list, id: int, legend_mapping: list):
    merge_bbox = None
    overall_list = []
    if bbox_list == []:
        overall_list.append(bbox_1)
        legend_mapping.append(id + 1)
        return overall_list, legend_mapping
    for bbox in bbox_list:
        overlap_score = compute_gtc(bbox_1, bbox)
        if overlap_score > 0.0:
            bbox_1_x1, bbox_1_y1, bbox_1_x2, bbox_1_y2, score_bbox_1 = bbox_1
            bbox_list_x1, bbox_list_y1, bbox_list_x2, bbox_list_y2, score_bbox = bbox
            mege_bbox_x1 = min(bbox_1_x1, bbox_list_x1)
            mege_bbox_y1 = min(bbox_1_y1, bbox_list_y1)
            mege_bbox_x2 = max(bbox_1_x2, bbox_list_x2)
            mege_bbox_y2 = max(bbox_1_y2, bbox_list_y2)
            merge_bbox = (
                mege_bbox_x1,
                mege_bbox_y1,
                mege_bbox_x2,
                mege_bbox_y2,
                score_bbox_1,
            )
            bbox_1 = merge_bbox
        else:
            overall_list.append(bbox)

    overall_list.append(bbox_1)
    legend_mapping.append(id + 1)

    return overall_list, legend_mapping
