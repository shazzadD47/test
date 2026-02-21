import os
from uuid import UUID

import fitz
from celery.utils.log import get_task_logger
from sqlalchemy import select, update

from app.configs import settings
from app.core.database.base import get_db_session
from app.core.database.crud import (
    ResultType,
    configure_retry,
    insert_batch_with_retry,
    insert_single_with_retry,
    select_with_retry,
)
from app.core.database.models import (
    FigureDetails,
    FileDetails,
    PageDetails,
    TableDetails,
)
from app.v3.endpoints.get_title_summery.logging import logger
from app.v3.endpoints.get_title_summery.utils.file_utils import _read_notebook_file
from app.v3.endpoints.get_title_summery.utils.s3_utils import extract_page_dimensions

celery_logger = get_task_logger(__name__)


def insert_file_details(file_details: FileDetails, max_retries: int = 3):
    insert_single_with_retry(file_details, max_retries)


def get_file_details(flag_id: str, supplementary_id: str = None) -> FileDetails | None:
    query = select(FileDetails).where(FileDetails.flag_id == flag_id)

    if supplementary_id:
        query = query.where(FileDetails.supplementary_id == supplementary_id)

    try:
        result = select_with_retry(query, result_type=ResultType.SCALAR_ONE)
    except Exception:
        logger.exception(
            f"[flag_id: {flag_id}] Unexpected error getting file details"
            f"[supplementary_id: {supplementary_id}]"
        )
        raise

    return result


def insert_page_details(
    page_details: PageDetails | list[PageDetails],
    batch_size: int = 5,
    max_retries: int = 3,
):
    """
    Insert page details into the database, with batch processing and retry logic.

    Args:
        page_details: A single PageDetails object or a list of PageDetails objects
        batch_size: Number of records to insert in each batch
        max_retries: Maximum number of retry attempts for failed operations
    """
    if isinstance(page_details, PageDetails):
        insert_single_with_retry(page_details, max_retries)
    else:
        for i in range(0, len(page_details), batch_size):
            batch = page_details[i : i + batch_size]
            insert_batch_with_retry(batch, max_retries)


def get_page_details(
    file_path: str, project_id: str, flag_id: str, max_retries: int = 3
):
    """
    Get page details for a specific project and flag.

    Args:
        project_id: The project ID to filter by
        flag_id: The flag ID to filter by
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary of page details indexed by page number
    """

    @configure_retry(max_retries=max_retries)
    def _get_page_details_with_retry():
        with get_db_session() as session:
            result = (
                session.query(PageDetails)
                .filter(
                    PageDetails.project_id == project_id, PageDetails.flag_id == flag_id
                )
                .all()
            )

            page_details = {}
            for page_detail in result:
                page_details[page_detail.page_number] = {
                    "width": page_detail.image_width,
                    "height": page_detail.image_height,
                }

            return page_details

    try:
        page_details = _get_page_details_with_retry()
        total_pages_from_db = len(page_details)

        file_extension = os.path.splitext(file_path)[-1].lower()
        if file_extension == ".ipynb":
            file_path = _read_notebook_file(file_path)
        pdf_document = fitz.open(file_path)

        if len(pdf_document) == total_pages_from_db:
            return page_details
        else:
            for page_number in range(len(pdf_document)):
                page_index = page_number + 1
                if page_index not in page_details:
                    dims = extract_page_dimensions(
                        pdf_document,
                        page_number,
                        settings.PDF_PAGE_IMAGE_RESOLUTION,
                        flag_id,
                    )
                    if dims:
                        page_details[page_index] = dims
                    else:
                        celery_logger.warning(
                            f"[{flag_id}] No dimensions for page {page_index}"
                        )

            pdf_document.close()

            return page_details

    except Exception as e:
        celery_logger.error(
            f"Error getting page dimensions: {e}"
            f"[project_id: {project_id}, flag_id: {flag_id}]"
        )
        return {}


def insert_figure_details(
    figure_details: FigureDetails | list[FigureDetails],
    batch_size: int = 4,
    max_retries: int = 3,
):
    """
    Insert figure details into the database, with batch processing and retry logic.

    Args:
        figure_details: A single FigureDetails object or a list of FigureDetails objects
        batch_size: Number of records to insert in each batch
        max_retries: Maximum number of retry attempts for failed operations
    """
    if isinstance(figure_details, FigureDetails):
        insert_single_with_retry(figure_details, max_retries)
    else:
        for i in range(0, len(figure_details), batch_size):
            batch = figure_details[i : i + batch_size]
            insert_batch_with_retry(batch, max_retries)


def insert_table_details(
    table_details: TableDetails | list[TableDetails],
    batch_size: int = 4,
    max_retries: int = 3,
):
    """
    Insert table details into the database, with batch processing and retry logic.

    Args:
        table_details: A single TableDetails object or a list of TableDetails objects
        batch_size: Number of records to insert in each batch
        max_retries: Maximum number of retry attempts for failed operations
    """
    if isinstance(table_details, TableDetails):
        insert_single_with_retry(table_details, max_retries)
    else:
        for i in range(0, len(table_details), batch_size):
            batch = table_details[i : i + batch_size]
            insert_batch_with_retry(batch, max_retries)


def update_legend_paths(figure_id: UUID, legend_paths: list[str], max_retries: int = 3):
    """
    Updates the legend_paths for a given figure_id in the FigureDetails table.

    Args:
        figure_id: The figure_id to filter rows
        legend_paths: The new legend paths to set
        max_retries: Maximum number of retry attempts for failed operations
    """
    if not isinstance(legend_paths, list):
        raise ValueError("legend_paths must be a list of strings.")

    @configure_retry(max_retries=max_retries)
    def _update_legend_paths_with_retry():
        with get_db_session() as session:
            query = (
                update(FigureDetails)
                .where(FigureDetails.figure_id == figure_id)
                .values(legend_paths=legend_paths)
            )
            session.execute(query)
            session.commit()

    _update_legend_paths_with_retry()


def update_legend_bounding_boxes(
    figure_id: UUID, bounding_boxes: list[int], max_retries: int = 3
):
    """
    Updates the legend_paths for a given figure_id in the FigureDetails table.

    Args:
        figure_id: The figure_id to filter rows
        legend_bounding_boxes: The new converted bounding box for legends
        max_retries: Maximum number of retry attempts for failed operations
    """
    if not isinstance(bounding_boxes, list):
        raise ValueError("legend_paths must be a list of strings.")

    @configure_retry(max_retries=max_retries)
    def _update_legend_bounding_boxes_with_retry():
        logger.debug("updating")
        with get_db_session() as session:
            query = (
                update(FigureDetails)
                .where(FigureDetails.figure_id == figure_id)
                .values(legend_converted_bbox_coordinates=bounding_boxes)
            )
            session.execute(query)
            session.commit()

    _update_legend_bounding_boxes_with_retry()


def delete_existing_figure_details(flag_id: str, max_retries: int = 3):
    """
    Delete all figure details for a given flag_id with retry logic.
    """

    @configure_retry(max_retries=max_retries)
    def _delete_with_retry():
        logger.debug(f"Deleting figure_details for flag_id: {flag_id}")
        with get_db_session() as db:
            db.query(FigureDetails).filter(FigureDetails.flag_id == flag_id).delete()
            db.commit()

    _delete_with_retry()


def delete_existing_table_details(flag_id: str, max_retries: int = 3):
    """
    Delete all table details for a given flag_id with retry logic.
    """

    @configure_retry(max_retries=max_retries)
    def _delete_with_retry():
        logger.debug(f"Deleting table_details for flag_id: {flag_id}")
        with get_db_session() as db:
            db.query(TableDetails).filter(TableDetails.flag_id == flag_id).delete()
            db.commit()

    _delete_with_retry()
