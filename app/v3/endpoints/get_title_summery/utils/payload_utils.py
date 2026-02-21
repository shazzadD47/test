from app.core.database.models import FigureDetails, TableDetails
from app.utils.texts import convert_data_to_string
from app.v3.endpoints.get_title_summery.schemas import AnnotationItem
from app.v3.endpoints.get_title_summery.utils.s3_utils import build_public_s3_url
from app.v3.endpoints.projects.constants import AutoConnection


def fix_paper_metadata_response(response: dict) -> dict:
    fixed_response = {}
    for key, value in response.items():
        if isinstance(value, dict):
            fixed_response[key] = fix_paper_metadata_response(value)
        else:
            fixed_response[key] = convert_data_to_string(value)

    return fixed_response


def build_ai_annotation_payload(
    figure_details_list: list[FigureDetails], table_details_list: list[TableDetails]
) -> list[AnnotationItem]:

    all_data = []

    parent_figure_ids = {
        fig.parent_figure_id for fig in figure_details_list if fig.parent_figure_id
    }

    for fig in figure_details_list:
        if fig.figure_id in parent_figure_ids:
            continue

        x1, y1, x2, y2 = fig.converted_bbox_coordinates
        x = round(x1)
        y = round(y1)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        caption = fig.caption
        if caption.strip().lower() in ["caption"]:
            caption = None

        footnote = fig.footnote
        if footnote.strip().lower() in ["footnote"]:
            footnote = None

        all_data.append(
            {
                "id": str(fig.figure_id),
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "pageNo": fig.page_number,
                "imgSrc": build_public_s3_url(fig.bucket_path),
                "type": "plot",
                "caption": caption,
                "description": fig.summary,
                "footnote": footnote,
                "number": fig.figure_number,
                "legends": [
                    {"legend_bbox": bbox, "legend_path": path}
                    for bbox, path in zip(
                        fig.legend_converted_bbox_coordinates or [],
                        fig.legend_paths or [],
                    )
                ],
            }
        )

    for tbl in table_details_list:
        x1, y1, x2, y2 = tbl.converted_bbox_coordinates
        x = round(x1)
        y = round(y1)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        caption = tbl.caption
        if caption.strip().lower() in AutoConnection.CAPTION_FILTER_WORDS:
            caption = None

        footnote = tbl.footnote
        if footnote.strip().lower() in AutoConnection.FOOTNOTE_FILTER_WORDS:
            footnote = None

        all_data.append(
            {
                "id": str(tbl.table_id),
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "pageNo": tbl.page_number,
                "imgSrc": build_public_s3_url(tbl.bucket_path),
                "type": "table",
                "caption": caption,
                "description": tbl.summary,
                "footnote": footnote,
                "number": None,
            }
        )

    return all_data
