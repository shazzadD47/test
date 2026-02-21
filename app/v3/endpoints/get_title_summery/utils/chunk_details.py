from uuid import UUID, uuid4

from celery.utils.log import get_task_logger

from app.core.database.models import FigureDetails, TableDetails

celery_logger = get_task_logger(__name__)


def prepare_figure_details(
    flag_id: str,
    project_id: str,
    page_details: dict,
    figure_chunk: dict,
    parent_figure_id: UUID | None = None,
) -> list[FigureDetails]:

    def _convert_bbox(norm_bbox: list[float], width: int, height: int) -> list[int]:
        """Converts normalized [x1, y1, x2, y2] to absolute pixel coordinates."""
        dimensions = [width, height, width, height]
        return [round(coord * dim) for coord, dim in zip(norm_bbox, dimensions)]

    def _get_all_legend_data(
        figure_chunk: dict, img_width: int, img_height: int
    ) -> tuple[list[str] | None, list[list[int]] | None]:
        """Extract ALL legends from the legends list format."""
        legends = figure_chunk.get("legends") or []

        if not legends or len(legends) == 0:
            return None, None

        legend_paths = []
        legend_bboxes = []

        for legend in legends:
            path = legend.get("legend_path")
            if path:
                legend_paths.append(path)
            bbox = legend.get("legend_bbox")
            if bbox:
                converted_bbox = _convert_bbox(bbox, img_width, img_height)
                legend_bboxes.append(converted_bbox)

        return (
            legend_paths if legend_paths else None,
            legend_bboxes if legend_bboxes else None,
        )

    figure_id = uuid4()
    figure_details_list = []

    page_number = figure_chunk.get("metadata", {}).get("page_id") + 1
    resolution = page_details.get(page_number)

    if not resolution:
        celery_logger.warning(
            f"[flag_id: {flag_id}] No resolution for page {page_number}"
        )
        resolution = list(page_details.values())[0]

    img_width = resolution.get("width", 0)
    img_height = resolution.get("height", 0)

    norm_bbox = figure_chunk.get("bbox", [])
    converted_bbox = (
        _convert_bbox(norm_bbox, img_width, img_height) if norm_bbox else []
    )

    legend_paths, legend_bboxes = _get_all_legend_data(
        figure_chunk, img_width, img_height
    )

    bucket_path = figure_chunk.get("img_s3_uri")
    bucket_path = bucket_path.replace("s3://", "")
    bucket_path = "/".join(bucket_path.split("/")[1:])

    figure_details = FigureDetails(
        flag_id=flag_id,
        project_id=project_id,
        page_number=page_number,
        figure_id=figure_id,
        parent_figure_id=parent_figure_id,
        figure_number=figure_chunk.get("figure_number"),
        caption=figure_chunk.get("img_caption"),
        summary=figure_chunk.get("value"),
        footnote=figure_chunk.get("img_footnote"),
        normalized_bbox_coordinates=norm_bbox,
        converted_bbox_coordinates=converted_bbox,
        legend_converted_bbox_coordinates=legend_bboxes,
        legend_paths=legend_paths,
        bucket_path=bucket_path,
    )

    figure_details_list.append(figure_details)

    for subfigure in figure_chunk.get("subfigures", []):
        subfigure_details = prepare_figure_details(
            flag_id, project_id, page_details, subfigure, figure_id
        )
        figure_details_list.extend(subfigure_details)

    return figure_details_list


def prepare_table_details(
    flag_id: str,
    project_id: str,
    page_details: dict,
    table_chunk: dict,
) -> TableDetails:
    table_id = uuid4()

    page_number = table_chunk.get("metadata").get("page_id") + 1
    resolution = page_details.get(page_number)

    if not resolution:
        celery_logger.warning(
            f"[flag_id: {flag_id}] No resolution for page {page_number}"
        )
        resolution = list(page_details.values())[0]

    image_height = resolution.get("height")
    image_width = resolution.get("width")

    normalized_bbox_coordinates = table_chunk.get("bbox")
    x1, y1, x2, y2 = normalized_bbox_coordinates

    x1 = round(x1 * image_width)
    x2 = round(x2 * image_width)
    y1 = round(y1 * image_height)
    y2 = round(y2 * image_height)

    converted_bbox_coordinates = [x1, y1, x2, y2]

    bucket_path = table_chunk.get("img_s3_uri")
    bucket_path = bucket_path.replace("s3://", "")
    bucket_path = "/".join(bucket_path.split("/")[1:])

    table_details = TableDetails(
        flag_id=flag_id,
        project_id=project_id,
        page_number=page_number,
        table_id=table_id,
        table_number=table_chunk.get("table_number"),
        caption=table_chunk.get("table_caption"),
        summary=table_chunk.get("value"),
        digitized=table_chunk.get("table_body"),
        footnote=table_chunk.get("table_footnote"),
        normalized_bbox_coordinates=normalized_bbox_coordinates,
        converted_bbox_coordinates=converted_bbox_coordinates,
        bucket_path=bucket_path,
    )

    return table_details
