import json
from datetime import datetime, timezone
from enum import Enum

from celery.utils.log import get_task_logger

from app.core.event_bus.producer_mq import ProducerMQ
from app.logging import logger

celery_logger = get_task_logger("delineate.event_bus")

BACKEND_EVENT_LOG_TTL = 12 * 60 * 60  # 12 hours

_LOGGED_EVENT_TYPES = {
    "MINERU_OUTPUT_STATUS",
    "CONVERT_PDF_TO_IMAGE",
    "PDF_SUMMARIZATION",
}


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

    if event.name in _LOGGED_EVENT_TYPES:
        _log_backend_event(event, message)


def _log_backend_event(event: BackendEventEnumType, message: dict) -> None:
    """Log backend event to Redis keyed by file_id for debugging event ordering."""
    file_id = message.get("file_id")
    if not file_id:
        return

    try:
        from app.redis import redis_client

        annotations = message.get("annotations")
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event.value,
            "flag_id": message.get("flag_id"),
            "status": message.get("status"),
            "response_type": message.get("response_type"),
            "has_annotations": annotations is not None,
            "annotations_count": len(annotations) if annotations else 0,
            "supplementary_id": message.get("supplementary_id"),
            "message": message.get("message"),
        }

        key = f"backend_events:{file_id}"
        redis_client.rpush(key, json.dumps(log_entry))
        redis_client.expire(key, BACKEND_EVENT_LOG_TTL)
    except Exception as e:
        logger.warning(f"Failed to log backend event to Redis: {e}")
