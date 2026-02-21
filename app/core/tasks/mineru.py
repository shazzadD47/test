from typing import Literal

from app.core.celery.app import celery_app
from app.core.database.crud import insert_single_with_retry
from app.core.database.models import MinerUResponses
from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.logging import logger
from app.utils.utils import sanitize_flag_id
from app.v3.endpoints.get_title_summery.schemas import MinerUOutputStatusPayload


@celery_app.task(name="log_mineru_output")
def log_mineru_output(
    flag_id: str,
    response_type: Literal["initial", "final"],
    response: dict,
    file_id: str | None = None,
):
    flag_id, supplementary_id = sanitize_flag_id(
        flag_id, return_supplimentary_info=True
    )

    try:
        insert_single_with_retry(
            MinerUResponses(
                flag_id=flag_id,
                supplementary_id=supplementary_id,
                response_type=response_type,
                response=response,
            )
        )
        if response_type == "initial":
            event_payload = MinerUOutputStatusPayload(
                flag_id=flag_id,
                file_id=file_id,
                supplementary_id=supplementary_id,
                status="1ST_PASS_COMPLETED",
                response_type=response_type,
                message="Initial processing completed successfully.",
            )
            send_to_backend(
                BackendEventEnumType.MINERU_OUTPUT_STATUS,
                event_payload.model_dump(),
            )
        elif response_type == "final":
            event_payload = MinerUOutputStatusPayload(
                flag_id=flag_id,
                file_id=file_id,
                supplementary_id=supplementary_id,
                status="2ND_PASS_COMPLETED",
                response_type=response_type,
                message="Final processing completed successfully.",
            )
            send_to_backend(
                BackendEventEnumType.MINERU_OUTPUT_STATUS,
                event_payload.model_dump(),
            )
    except Exception as e:
        logger.exception(f"Error logging mineru output: {e}")
