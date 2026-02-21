from uuid import uuid4

from sqlalchemy import select

from app.core.database.crud import select_with_retry
from app.core.database.models import FigureDetails
from app.utils.subfigure_matching import get_matching_figure
from app.v3.endpoints.get_title_summery.utils.utils import secure_file_path
from app.v3.endpoints.plot_digitizer.helpers import (
    download_files_with_retries,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger


def find_matching_figure(
    paper_id: str,
    page_no: int = None,
    bounding_box: dict = None,
) -> dict:
    figure_details, paths = {}, {}
    if page_no is not None and bounding_box is not None and bounding_box != {}:
        bbox_coords = [
            bounding_box["top_left_x"],
            bounding_box["top_left_y"],
            bounding_box["bottom_right_x"],
            bounding_box["bottom_right_y"],
        ]
        matching_figure = get_matching_figure(
            flag_id=paper_id,
            page=page_no,
            bbox=bbox_coords,
        )

        if matching_figure is not None:
            figure_details["figure_number"] = matching_figure.figure_number
            figure_details["caption"] = matching_figure.caption
            figure_details["footnote"] = matching_figure.footnote
            figure_details["summary"] = matching_figure.summary

            parent_figure_id = matching_figure.parent_figure_id
            if parent_figure_id is not None:
                parent_figure_id = parent_figure_id.hex
                try:
                    query = select(FigureDetails).where(
                        FigureDetails.figure_id == parent_figure_id,
                    )

                    parent_figure_details = select_with_retry(query)
                    parent_figure_details = parent_figure_details[0]

                    figure_details["parent_figure_summary"] = (
                        parent_figure_details.summary
                    )
                    figure_details["parent_figure_caption"] = (
                        parent_figure_details.caption
                    )
                    figure_details["parent_figure_footnote"] = (
                        parent_figure_details.footnote
                    )

                    logger.info(
                        f"Parent figure details extracted: {parent_figure_details}"
                    )
                except Exception as e:
                    logger.info(f"Error fetching parent figure details: {e}")

            matched_figure_path = secure_file_path(
                base_dir="./", filename=f"{uuid4()}.png"
            )
            _, download_success = download_files_with_retries(
                matching_figure.bucket_path, matched_figure_path
            )
            if download_success:
                paths["matched_figure_path"] = matched_figure_path
                logger.info("Downloaded matched figure successfully")

            if matching_figure.legend_paths:
                for i, legend_path in enumerate(matching_figure.legend_paths):
                    legend_save_path = secure_file_path(
                        base_dir="./", filename=f"{uuid4()}.png"
                    )
                    _, download_success = download_files_with_retries(
                        legend_path, legend_save_path
                    )
                    if download_success:
                        if "legend_paths" not in paths:
                            paths["legend_paths"] = [legend_save_path]
                        else:
                            paths["legend_paths"].append(legend_save_path)
                        logger.info(f"Downloaded legend {i+1} successfully")

    return figure_details, paths
