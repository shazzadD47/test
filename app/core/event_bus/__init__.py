from enum import Enum

from celery.utils.log import get_task_logger

from app.core.event_bus.producer_mq import ProducerMQ
from app.logging import logger

celery_logger = get_task_logger("delineate.event_bus")


class BackendEventEnumType(Enum):
    PRESET_AUTOFILL_COVERIATE = "PRESET_AUTOFILL_COVERIATE"
    PRESET_AUTOFILL_DOSING = "PRESET_AUTOFILL_DOSING"
    PRESET_AUTOFILL_LINE = "PRESET_AUTOFILL_LINE"
    PRESET_AUTOFILL_PLOT = "PLOT_AUTOFILL"
    PRESET_AUTOFILL_DIGITIZATION = "PLOT_DIGITIZATION"
    PRESET_AUTOFILL_ITERATIVE = "PRESET_AUTOFILL_ITERATIVE"
    PRESET_DOSING_TABLE_WITH_FIGURE = "PRESET_DOSING_TABLE_WITH_FIGURE"
    PRESET_DOSING_TABLE_NO_FIGURE = "PRESET_DOSING_TABLE_NO_FIGURE"
    PRESET_GENERAL_EXTRACTION = "PRESET_GENERAL_EXTRACTION"
    PRESET_GENERAL_EXTRACTION_ROOT_LABELS = "PRESET_GENERAL_EXTRACTION_ROOT_LABELS"
    AUTO_INPUT_SUGGESTION = "AUTO_INPUT_SUGGESTION"
    PDF_SUMMARIZATION = "PDF_SUMMARIZATION"
    CONVERT_PDF_TO_IMAGE = "CONVERT_PDF_TO_IMAGE"
    MINERU_OUTPUT_STATUS = "MINERU_OUTPUT_STATUS"
    UNIT_STANDARDIZATION = "UNIT_STANDARDIZATION"
    COLUMN_STANDARDIZATION = "COLUMN_STANDARDIZATION"
    TAG_EXTRACTION = "TAG_EXTRACTION"


def send_to_backend(
    event: BackendEventEnumType,
    message: dict,
    queue_name: str = None,
):
    """
    Send a message to an event bus synchronously in a Celery task.

    Args:
        event (BackendEventEnumType): The event to send.
        message (dict): The message to send.

    Returns:
        None

    Examples:
        >>> from app.core.event_bus import (
        ...     send_to_backend,
        ...     BackendEventEnumType
        ... )
        >>> data = {
        ...     "test": "hello world",
        ... }
        >>> send_to_backend(BackendEventEnumType.PRESET_AUTOFILL_COVERIATE, data)
    """
    pattern = event.value
    payload = {"pattern": pattern, "data": message}

    try:
        producer = ProducerMQ(queue_name=queue_name)
        with producer:
            producer.send_message(payload)

            logger.info(f"[RMQ] Sending message with pattern: {pattern}")
            celery_logger.info(f"[RMQ] Sending message with pattern: {pattern}")
    except Exception as e:
        logger.exception(e)
        celery_logger.exception(e)
