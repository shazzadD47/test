import time

from celery import shared_task
from celery.utils.log import get_task_logger

from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.utils.decorators.cost_tracker import track_all_llm_costs
from app.v3.endpoints import Status
from app.v3.endpoints.dosing_table.configs import settings as dosing_settings
from app.v3.endpoints.dosing_table.constants import DOSING_TABLE_ERROR_MESSAGE
from app.v3.endpoints.dosing_table.services.with_figure import (
    prepare_dosing_table,
)
from app.v3.endpoints.dosing_table.services.without_figure import (
    prepare_dosing_table_no_figure,
)

logger = get_task_logger("dosing_table")


@track_all_llm_costs
def execute_dosing_table_with_figure_task(
    project_id: str, flag_id: str, image_url: str, metadata: dict
):
    result = prepare_dosing_table(project_id, flag_id, image_url, metadata)
    logger.info(f"Dosing table prepared: {result}")
    return result


@shared_task(
    name="prepare_dosing_table_with_figure",
    bind=True,
    max_retries=0,
    default_retry_delay=10,
)
def prepare_dosing_table_with_figure_task(
    self, project_id: str, flag_id: str, image_url: str, metadata: dict
):
    """
    Celery task to prepare dosing table with figure.
    """
    retry_count = 0
    while retry_count < dosing_settings.MAX_RETRIES:
        try:
            output = execute_dosing_table_with_figure_task(
                project_id, flag_id, image_url, metadata
            )
            if (
                "metadata" in output
                and "status" in output["metadata"]
                and output["metadata"]["status"] == Status.SUCCESS.value
            ):
                send_to_backend(
                    BackendEventEnumType.PRESET_DOSING_TABLE_WITH_FIGURE, output
                )
                return output
            else:
                retry_count += 1
                retry_delay = dosing_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
                time.sleep(retry_delay)
                if "metadata" in output and isinstance(output["metadata"], dict):
                    metadata.update(output["metadata"])
                if retry_count >= dosing_settings.MAX_RETRIES:
                    send_to_backend(
                        BackendEventEnumType.PRESET_DOSING_TABLE_WITH_FIGURE, output
                    )
                    return output
                continue
        except Exception as exc:
            logger.exception(f"Error preparing dosing table: {str(exc)}")
            retry_count += 1
            retry_delay = dosing_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
            time.sleep(retry_delay)
            if retry_count >= dosing_settings.MAX_RETRIES:
                output = {
                    "payload": {},
                    "metadata": metadata,
                }
                output["metadata"]["status"] = Status.FAILED.value
                output["metadata"]["message"] = DOSING_TABLE_ERROR_MESSAGE
                send_to_backend(
                    BackendEventEnumType.PRESET_DOSING_TABLE_WITH_FIGURE, output
                )
                return output
            continue


@track_all_llm_costs
def execute_dosing_table_no_figure_task(project_id: str, flag_id: str, metadata: dict):
    """
    Celery task to prepare dosing table without figure.

    Args:
        project_id (str): The project ID
        flag_id (str): The flag ID
        metadata (dict): Additional metadata for the task

    Returns:
        dict: The prepared dosing table data

    Raises:
        Exception: If the task fails after all retries
    """
    result = prepare_dosing_table_no_figure(project_id, flag_id, metadata)

    logger.info(f"Dosing table prepared: {result}")
    return result


@shared_task(
    name="prepare_dosing_table_no_figure",
    bind=True,
    max_retries=0,
    default_retry_delay=10,
)
def prepare_dosing_table_no_figure_task(
    self, project_id: str, flag_id: str, metadata: dict
):
    """
    Celery task to prepare dosing table without figure.

    Args:
        project_id (str): The project ID
        flag_id (str): The flag ID
        metadata (dict): Additional metadata for the task

    Returns:
        dict: The prepared dosing table data

    Raises:
        Exception: If the task fails after all retries
    """
    retry_count = 0
    while retry_count < dosing_settings.MAX_RETRIES:
        try:
            output = execute_dosing_table_no_figure_task(project_id, flag_id, metadata)
            if (
                "metadata" in output
                and "status" in output["metadata"]
                and output["metadata"]["status"] == Status.SUCCESS.value
            ):
                send_to_backend(
                    BackendEventEnumType.PRESET_DOSING_TABLE_NO_FIGURE, output
                )
                return output
            else:
                retry_count += 1
                retry_delay = dosing_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
                time.sleep(retry_delay)
                if "metadata" in output and isinstance(output["metadata"], dict):
                    metadata.update(output["metadata"])
                if retry_count >= dosing_settings.MAX_RETRIES:
                    send_to_backend(
                        BackendEventEnumType.PRESET_DOSING_TABLE_NO_FIGURE, output
                    )
                    return output
                continue
        except Exception as exc:
            logger.exception(f"Error preparing dosing table: {str(exc)}")
            retry_count += 1
            retry_delay = dosing_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
            time.sleep(retry_delay)
            if retry_count >= dosing_settings.MAX_RETRIES:
                output = {
                    "payload": {},
                    "metadata": metadata,
                }
                output["metadata"]["status"] = Status.FAILED.value
                output["metadata"]["message"] = DOSING_TABLE_ERROR_MESSAGE
                send_to_backend(
                    BackendEventEnumType.PRESET_DOSING_TABLE_NO_FIGURE, output
                )
                return output
            continue
