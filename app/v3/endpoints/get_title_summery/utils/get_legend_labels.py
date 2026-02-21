from app.v3.endpoints.get_title_summery.configs import settings as project_settings
from app.v3.endpoints.get_title_summery.utils.image_process import (
    check_overlap,
    compute_gtc,
)


def get_legend_labels(total_result, bounding_box_list: list, bbox_list: list):

    legend_labels = []
    legend_mapping = []

    """
        first check if legend label is inside legend area
        then go for check_overlap
        """
    if project_settings.LEGEND_LABEL in total_result and isinstance(
        total_result[project_settings.LEGEND_LABEL], list
    ):
        for data in total_result[project_settings.LEGEND_LABEL]:
            x1, y1, x2, y2, score = (
                data["x1"],
                data["y1"],
                data["x2"],
                data["y2"],
                data["score"],
            )
            bbox = (x1, y1, x2, y2, score)
            if score > project_settings.SECOND_CATEGORY_THRESHOLD:
                if compute_gtc(bounding_box_list[0], bbox) > 0:
                    legend_labels, legend_mapping = check_overlap(
                        bbox, legend_labels, -1, legend_mapping
                    )
                else:
                    for i in range(len(bbox_list)):
                        gtc_score = compute_gtc(bbox_list[i], bbox)
                        if gtc_score > 0:
                            legend_labels, legend_mapping = check_overlap(
                                bbox, legend_labels, i, legend_mapping
                            )

    """
        create a map of legend labels for each legend area

        """
    total_legend_map = {}
    for i in range(len(legend_labels)):
        if total_legend_map.get(legend_mapping[i]) is None:
            total_legend_map[legend_mapping[i]] = [legend_labels[i]]
        else:
            part_legend_list = total_legend_map.get(legend_mapping[i])

            part_legend_list.append(legend_labels[i])
            total_legend_map[legend_mapping[i]] = part_legend_list

    return total_legend_map
