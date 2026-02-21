import json

from fastapi import APIRouter

from app.v3.endpoints.get_paper_labels import dynamic_paper_metadata_extraction_service
from app.v3.endpoints.get_paper_labels.logging import logger
from app.v3.endpoints.get_paper_labels.schemas import DynamicPaperLabelsRequest

router = APIRouter(tags=["paper_labels"])


@router.post("/meta-analysis/paper-labels")
async def get_dynamic_paper_metadata(
    payload: DynamicPaperLabelsRequest,
):
    logger.debug(f"Received payload: {payload.model_dump_json()}")

    response = await dynamic_paper_metadata_extraction_service(
        payload.paper_id,
        payload.project_id or "",
        payload.table_structure,
    )
    logger.debug(f"Response: {json.dumps(response, ensure_ascii=False)}")
    return response
