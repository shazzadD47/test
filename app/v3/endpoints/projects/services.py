import asyncio

import boto3
from pydantic import BaseModel
from sqlalchemy import select

from app.configs import settings
from app.core.database.crud import ResultType, delete_with_retry, select_with_retry
from app.core.database.models import (
    FigureDetails,
    FileDetails,
    MinerUResponses,
    PageDetails,
    TableDetails,
)
from app.core.vector_store import VectorStore
from app.v3.endpoints.projects.constants import AutoConnection
from app.v3.endpoints.projects.exceptions import (
    DatabaseError,
    DataFetchFailed,
    StorageError,
    UnexpectedError,
)
from app.v3.endpoints.projects.logging import logger
from app.v3.endpoints.projects.schemas import AutoFigureConnection
from app.v3.endpoints.projects.utils import delete_s3_files

s3_client = boto3.client(
    "s3",
    endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
    aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
    aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
)


async def delete_project_and_storage_service(project_id: str):
    """
    Delete all documents and storage files associated with a project.

    Args:
        project_id: The ID of the project to delete

    Raises:
        HTTPException: If database or storage operations fail
    """
    logger.debug(f"Starting deletion for project_id: {project_id}")

    try:
        # Get flag_ids using SQLAlchemy
        flag_ids_query = select(FileDetails.flag_id).where(
            FileDetails.project_id == project_id
        )
        flag_ids_result = select_with_retry(flag_ids_query, ResultType.SCALAR_ALL)
        flag_ids_result = [str(res) for res in flag_ids_result]

        if not flag_ids_result:
            logger.info(f"No documents found for project_id: {project_id}")
            return

        flag_ids = list(set(flag_ids_result))

        def _delete_db_entries():
            """Run synchronous SQLAlchemy deletions inside a worker thread."""
            try:
                delete_success = True

                if not delete_with_retry(FigureDetails, {"project_id": project_id}):
                    delete_success = False

                if not delete_with_retry(PageDetails, {"project_id": project_id}):
                    delete_success = False

                if not delete_with_retry(TableDetails, {"project_id": project_id}):
                    delete_success = False

                if not delete_with_retry(FileDetails, {"project_id": project_id}):
                    delete_success = False

                if not delete_success:
                    logger.error(
                        f"Some database deletions failed for project {project_id}"
                    )
                    raise DatabaseError()

            except Exception as e:
                logger.exception(
                    f"Database deletion failed for project {project_id}: {str(e)}"
                )
                raise DatabaseError()

        await asyncio.gather(
            asyncio.to_thread(_delete_db_entries),
            delete_s3_files(flag_ids, s3_client, settings.S3_SPACES_BUCKET),
            asyncio.to_thread(VectorStore.delete_by_project_id, project_id),
        )
    except DatabaseError as e:
        logger.exception(f"Database error during deletion: {str(e)}")
        raise DatabaseError()
    except StorageError as e:
        logger.exception(f"Storage error during deletion: {str(e)}")
        raise StorageError()
    except Exception as e:
        logger.exception(f"Unexpected error during deletion: {str(e)}")
        raise UnexpectedError()


async def delete_flag_and_storage_service(project_id: str, flag_id: str):
    logger.debug(f"Starting deletion for flag_id: {flag_id}")

    try:

        def _delete_db_entries():
            try:
                delete_success = True

                if not delete_with_retry(
                    FigureDetails, {"project_id": project_id, "flag_id": flag_id}
                ):
                    delete_success = False

                if not delete_with_retry(
                    PageDetails, {"project_id": project_id, "flag_id": flag_id}
                ):
                    delete_success = False

                if not delete_with_retry(
                    TableDetails, {"project_id": project_id, "flag_id": flag_id}
                ):
                    delete_success = False

                if not delete_with_retry(
                    FileDetails, {"project_id": project_id, "flag_id": flag_id}
                ):
                    delete_success = False

                if not delete_success:
                    logger.error(
                        f"Some database deletions failed for flag_id {flag_id}"
                    )
                    raise DatabaseError()

            except Exception as e:
                logger.exception(f"Database deletion failed: {str(e)}")
                raise DatabaseError()

        await asyncio.gather(
            asyncio.to_thread(_delete_db_entries),
            delete_s3_files([flag_id], s3_client, settings.S3_SPACES_BUCKET),
            asyncio.to_thread(VectorStore.delete_by_flag_id, flag_id),
        )
    except DatabaseError as e:
        logger.exception(f"Database error during deletion: {str(e)}")
        raise DatabaseError()
    except StorageError as e:
        logger.exception(f"Storage error during deletion: {str(e)}")
        raise StorageError()
    except Exception as e:
        logger.exception(f"Unexpected error during deletion: {str(e)}")
        raise UnexpectedError()


class MineruResponseCheck(BaseModel):
    response_checking_fail: bool
    response_status: bool | None


async def check_mineru_response_status(flag_id: str) -> MineruResponseCheck:
    try:
        response_query = select(MinerUResponses.response_type).where(
            MinerUResponses.flag_id == flag_id
        )
        response_result = select_with_retry(response_query, ResultType.SCALAR_ALL)

        if response_result:
            response_types = set(response_result)
            if "final" in response_types:
                return {
                    "response_checking_fail": False,
                    "response_status": True,
                }

        return {
            "response_checking_fail": False,
            "response_status": False,
        }

    except Exception as e:
        logger.exception(f"Error checking response status: {str(e)}")
        return {
            "response_checking_fail": True,
            "response_status": None,
        }


async def get_sub_figure_autodetect_data(
    project_id: str, flag_id: str, mineru_status: bool
) -> AutoFigureConnection:
    if mineru_status:
        bucket_path = settings.S3_SPACES_PUBLIC_BASE_URL
        if not bucket_path.endswith("/"):
            bucket_path += "/"

        figure_details_query = select(FigureDetails).where(
            FigureDetails.project_id == project_id, FigureDetails.flag_id == flag_id
        )
        figure_details_result = select_with_retry(
            figure_details_query, ResultType.SCALAR_ALL
        )

        table_details_query = select(TableDetails).where(
            TableDetails.project_id == project_id, TableDetails.flag_id == flag_id
        )
        table_details_result = select_with_retry(
            table_details_query, ResultType.SCALAR_ALL
        )

        all_data = []
        parent_figure_ids = {
            data.parent_figure_id
            for data in figure_details_result
            if data.parent_figure_id
        }
        for data in figure_details_result:
            if data.figure_id in parent_figure_ids:
                continue
            id = data.figure_id
            x_top, y_top, x_bottom, y_bottom = data.converted_bbox_coordinates
            x = int(x_top)
            y = int(y_top)
            width = abs(int(x_bottom) - int(x_top))
            height = abs(int(y_bottom) - int(y_top))
            pageNo = data.page_number
            imgSrc = data.bucket_path
            type = "plot"

            caption = data.caption
            caption_filter_words = ["caption"]
            if caption in caption_filter_words:
                caption = None

            summary = data.summary
            footnote = data.footnote
            footnote_filter_words = ["footnote"]
            if footnote in footnote_filter_words:
                footnote = None

            temp_data = {
                "id": id,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "pageNo": pageNo,
                "imgSrc": f"{bucket_path}{imgSrc}",
                "type": type,
                "caption": caption,
                "description": summary,
                "footnote": footnote,
                "aiSuggested": {"status": False, "description": []},
            }
            all_data.append(temp_data)

        # extract table data
        for data in table_details_result:
            id = data.table_id
            x_top, y_top, x_bottom, y_bottom = data.converted_bbox_coordinates
            x = int(x_top)
            y = int(y_top)
            width = abs(int(x_bottom) - int(x_top))
            height = abs(int(y_bottom) - int(y_top))
            pageNo = data.page_number
            imgSrc = data.bucket_path
            type = "table"

            caption = data.caption
            caption_filter_words = AutoConnection.CAPTION_FILTER_WORDS
            if caption in caption_filter_words:
                caption = None

            summary = data.summary
            footnote = data.footnote
            footnote_filter_words = AutoConnection.FOOTNOTE_FILTER_WORDS
            if footnote in footnote_filter_words:
                footnote = None

            temp_data = {
                "id": id,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "pageNo": pageNo,
                "imgSrc": f"{bucket_path}{imgSrc}",
                "type": type,
                "caption": caption,
                "description": summary,
                "footnote": footnote,
                "aiSuggested": None,
            }
            all_data.append(temp_data)

        return_data = {"isAiProcessDone": True, "annotations": all_data}

        return return_data

    else:
        return_data = {"isAiProcessDone": False, "annotations": []}

        return return_data


def get_paper_summaries_by_project(project_id: str) -> list[str] | None:
    """
    Retrieves all paper_summary values for a given project_id.

    Args:
        project_id: The project ID to filter the FileDetails rows

    Returns:
        A list of paper_summary strings, or None if an error occurs
    """
    query = select(
        FileDetails.flag_id,
        FileDetails.supplementary_id,
        FileDetails.paper_summary,
        FileDetails.summary,
    ).where(FileDetails.project_id == project_id)

    try:
        results = select_with_retry(query, result_type=ResultType.ROW_ALL)

        return [
            {
                "flag_id": result[0],
                "supplementary_id": result[1],
                "paper_summary": result[2],
                "abstract": result[3],
            }
            for result in results
        ]
    except Exception as e:
        logger.exception(f" failed  for project_id: {project_id}] : {str(e)}")
        raise DataFetchFailed()
