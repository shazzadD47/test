from io import BytesIO

import cv2

from app.v3.endpoints.get_title_summery.configs import settings as project_settings
from app.v3.endpoints.get_title_summery.logging import logger
from app.v3.endpoints.get_title_summery.services.chart_dete import chart_dete_completion
from app.v3.endpoints.get_title_summery.services.check_claude import (
    area_compare,
    legend_map,
)
from app.v3.endpoints.get_title_summery.services.crud import (
    update_legend_bounding_boxes,
)
from app.v3.endpoints.get_title_summery.utils.detect_main_images import (
    detect_main_image,
)
from app.v3.endpoints.get_title_summery.utils.get_legend_labels import get_legend_labels
from app.v3.endpoints.get_title_summery.utils.image_process import (
    compute_gtc,
    crop_image,
    draw_bounding_box,
)
from app.v3.endpoints.get_title_summery.utils.utils import upload_legend


def crop_legend(
    extract_img,
    total_image,
    extracted_img_path,
    total_img_path,
    figure_id=None,
    top_box=None,
    bbox_list=None,
    bounding_box_list=None,
    total_result=None,
    client=None,
    flag_id=None,
):

    cropped_image = None

    if top_box is None:
        top_box, bbox_list, bounding_box_list, total_result = detect_main_image(
            total_img_path, total_image
        )

    if top_box is None:
        return top_box, bbox_list, bounding_box_list, total_result, cropped_image

    """
    merging bounding boxes of alternative legend areas where required
    """
    if len(bbox_list) > 0:
        height, width = total_image.shape[:2]
        padding = 60

        bbox_image = total_image.copy()
        x1, y1, x2, y2 = top_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        start_bbox = (x1, y1, x2, y2)
        bbox_image = draw_bounding_box(bbox_image, start_bbox, project_settings.AREA_1)
        area = 2

        merged_list = []
        for i in range(len(bbox_list)):
            if i in merged_list:
                continue
            for j in range(i + 1, len(bbox_list)):
                if j in merged_list:
                    continue
                gtc_score = compute_gtc(bbox_list[i], bbox_list[j])
                if gtc_score > 0:

                    merged_list.append(j)
                    bbox_list[i][0] = min(bbox_list[i][0], bbox_list[j][0])
                    bbox_list[i][1] = min(bbox_list[i][1], bbox_list[j][1])
                    bbox_list[i][2] = max(bbox_list[i][2], bbox_list[j][2])
                    bbox_list[i][3] = max(bbox_list[i][3], bbox_list[j][3])

        new_bbox_list = []
        for i in range(len(bbox_list)):
            if i not in merged_list:
                new_bbox_list.append(bbox_list[i])
        bbox_list = new_bbox_list

    """
    get the legend label map for the total image
    """
    total_image_legend_map = get_legend_labels(
        total_result, bounding_box_list, bbox_list
    )

    result = chart_dete_completion(extracted_img_path)

    score_list = []
    extracted_bbox = []
    """
    get the legend area for the second image
    """
    if project_settings.LEGEND_AREA in result and isinstance(
        result[project_settings.LEGEND_AREA], list
    ):

        for extracted_data in result[project_settings.LEGEND_AREA]:
            bbox = (
                extracted_data["x1"],
                extracted_data["y1"],
                extracted_data["x2"],
                extracted_data["y2"],
                extracted_data["score"],
            )
            if extracted_data["score"] > project_settings.INITIAL_THRESHOLD:
                extracted_bbox.append(bbox)
                score_list.append(extracted_data["score"])

    patches_list = []
    patches_check = False
    legend_label_check = False
    if len(extracted_bbox) > 0:

        """
        look for patch where the legend is present in the extracted image
        """

        if project_settings.LEGEND_PATCH in result and isinstance(
            result[project_settings.LEGEND_PATCH], list
        ):
            for extracted_data in result[project_settings.LEGEND_PATCH]:
                x1, y1, x2, y2, score = (
                    extracted_data["x1"],
                    extracted_data["y1"],
                    extracted_data["x2"],
                    extracted_data["y2"],
                    extracted_data["score"],
                )
                bbox = (x1, y1, x2, y2, score)
                if extracted_data["score"] > project_settings.INITIAL_THRESHOLD:
                    gtc_score = compute_gtc(extracted_bbox[0], bbox)
                    if gtc_score > 0:
                        patches_list.append([x1, y1, x2, y2])
                        patches_check = True

        extracted_image_legend_map = get_legend_labels(result, extracted_bbox, [])

        """
        check if legend labels in the original image and the extracted
        image are the same number for any of the cases
        """
        for value in extracted_image_legend_map.items():
            for value1 in total_image_legend_map.items():
                if len(value) >= len(value1):

                    legend_label_check = True
                    break

    is_legend_present = False

    """
    if legend patch is found and legend labels are equal
    then check the confidence of the legend area.
    If confidence is low check with claude
    """
    logger.debug("testing structured output")

    if patches_check and legend_label_check:

        if score_list and score_list[0] < project_settings.LEGEND_DETECTION_THRESHOLD:

            response = legend_map([total_image, extract_img], client=client)

            # json_response = json.loads(response.content)
            result = response.answer
            if result.lower() == "yes":
                is_legend_present = True

        else:

            is_legend_present = True

    cropped_image = None
    if not is_legend_present:

        """
        for multiple areas draw the bounding boxes and ask claude which
        one is most suitable for the extracted image
        """

        if len(bbox_list) > 0:

            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox
                x1 -= padding
                x2 += padding
                x1 = max(0, x1)
                x2 = min(width - 1, x2)

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                other_bbox = (x1, y1, x2, y2)
                bbox_image = draw_bounding_box(
                    bbox_image, other_bbox, f"""AREA: {area}"""
                )
                area += 1
            logger.debug("comparing area")
            response = area_compare([bbox_image, extract_img], client=client)

            # json_data = json.loads(response.content)
            area_number = response.area_number_to_choose

            area_number = int(area_number)
            if area_number == 1:
                final_bounding_box = list(map(int, bounding_box_list[0]))
            else:
                final_bounding_box = list(map(int, bbox_list[area_number - 2]))

            cropped_image = crop_image(total_image, final_bounding_box)
            success, buffer = cv2.imencode(".png", cropped_image)
            if success:
                cropped_image_io = BytesIO(buffer.tobytes())
                if flag_id is not None:
                    upload_legend(cropped_image_io, figure_id, flag_id)
                    logger.debug("upload_legend bbox")
                    update_legend_bounding_boxes(figure_id, final_bounding_box)

        else:
            if len(bounding_box_list) > 0:
                final_bounding_box = list(map(int, bounding_box_list[0]))

                cropped_image = crop_image(total_image, final_bounding_box)
                success, buffer = cv2.imencode(".png", cropped_image)
                if success:
                    cropped_image_io = BytesIO(buffer.tobytes())
                    if flag_id is not None:
                        upload_legend(cropped_image_io, figure_id, flag_id)
                        logger.debug("upload_legend bbox")
                        update_legend_bounding_boxes(figure_id, final_bounding_box)

    return top_box, bbox_list, bounding_box_list, total_result, cropped_image
