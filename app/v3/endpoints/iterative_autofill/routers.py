from fastapi import APIRouter

from app.v3.endpoints.iterative_autofill import iterative_metadata_extraction_service
from app.v3.endpoints.iterative_autofill.logging import logger
from app.v3.endpoints.iterative_autofill.schemas import IterativeAutofillRequest

router = APIRouter(tags=["Iterative Autofill"])


@router.post("/meta-analysis/iterative-autofill/")
async def get_meta_analysis_iterative_autofill(
    payload: IterativeAutofillRequest,
):
    logger.debug(f"Received payload: {payload.model_dump_json()}")

    return iterative_metadata_extraction_service(data=payload)
