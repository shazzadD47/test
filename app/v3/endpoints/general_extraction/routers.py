from fastapi import APIRouter

from app.v3.endpoints.general_extraction.logging import logger
from app.v3.endpoints.general_extraction.schemas import (
    GeneralExtractionRequest,
    update_ctype,
    validate_payload,
)
from app.v3.endpoints.general_extraction.services import (
    general_extraction_task,
)

router = APIRouter(tags=["General Extraction"])


@router.post("/extractions/general/")
async def general_extraction(
    payload: GeneralExtractionRequest,
):
    validate_payload(payload)
    payload = payload.model_dump()
    payload["table_structure"] = update_ctype(payload["table_structure"])
    logger.info(f"General Extraction Payload: {payload}")
    task = general_extraction_task.apply_async(
        kwargs={
            "inputs": payload,
        },
    )

    metadata = payload["metadata"]
    if metadata is None:
        metadata = {}

    metadata["message"] = "General extraction process started in Background"
    metadata["ai_metadata"] = {"task_id": task.id}

    return metadata
