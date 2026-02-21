from app.v3.endpoints.get_title_summery.configs import settings as project_settings
from app.v3.endpoints.get_title_summery.services.chart_dete import chart_dete_completion
from app.v3.endpoints.get_title_summery.utils.image_process import compute_gtc


def detect_main_image(total_image_path, total_image):

    total_result = chart_dete_completion(total_image_path)

    score_list = []
    bounding_box_list = []
    """
        get the legend areas in the main image
        """
    if project_settings.LEGEND_AREA in total_result and isinstance(
        total_result[project_settings.LEGEND_AREA], list
    ):
        for data in total_result[project_settings.LEGEND_AREA]:
            x1, y1, x2, y2, score = (
                data["x1"],
                data["y1"],
                data["x2"],
                data["y2"],
                data["score"],
            )
            if score > project_settings.MINIMUM_THRESHOLD:
                score_list.append(score)
                bounding_box_list.append([x1, y1, x2, y2])

    top_box = None
    bbox_list = []
    overlap_bbox = []
    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []

    if len(bounding_box_list) > 0:
        top_box = bounding_box_list[0]

        """
            check for overlap of other areas with the top legend area in the main image
            """
        for bbox in bounding_box_list:
            if bbox != top_box:
                gtc_score = compute_gtc(top_box, bbox)

                if gtc_score > 0:
                    overlap_bbox.append(bbox)
            """
            merge the overlap bounding boxes
            """
        if len(overlap_bbox) > 0:
            for bbox in overlap_bbox:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)
            x1 = min(x1_list)
            y1 = min(y1_list)
            x2 = max(x2_list)
            y2 = max(y2_list)

            if x1 < top_box[0]:
                top_box[0] = x1

            if x2 > top_box[2]:
                top_box[2] = x2

            """
            choose the other legend areas which has no overlap
            with the top legend area but has high confidence

            """

        for count, bbox in enumerate(
            bounding_box_list
        ):  # Use enumerate to get the index and the bbox
            if bbox != top_box:
                gtc_score = compute_gtc(top_box, bbox)

                if (
                    gtc_score == 0.0
                    and score_list[count] > project_settings.SECOND_CATEGORY_THRESHOLD
                ):
                    bbox_list.append(bbox)

            count += 1
        height, width = total_image.shape[:2]

        """
            add padding to the top box
        """
        padding = 60
        top_box[0] -= padding
        top_box[2] += padding
        top_box[0] = max(0, top_box[0])
        top_box[2] = min(width - 1, top_box[2])

    return top_box, bbox_list, bounding_box_list, total_result
