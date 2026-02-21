from fastapi import APIRouter

from app.logging import logger
from app.utils.token_usage import async_track_openai_usage
from app.v3.endpoints.autofill.exceptions import (
    NoRootChoiceProvided,
    RootChoiceNotAllowedException,
)
from app.v3.endpoints.autofill.schemas import MetaAnalysisAutofillRequest
from app.v3.endpoints.autofill.services import autofill_meta_analysis_data

logger = logger.getChild("autofill")

router = APIRouter(tags=["Autofill"])


@router.post("/meta-analysis/autofill/")
@async_track_openai_usage
async def get_meta_analysis_autofill(payload: MetaAnalysisAutofillRequest):
    if payload.is_root and payload.root_choices is not None:
        raise RootChoiceNotAllowedException()

    logger.debug(
        f"Autofill meta-analysis data with payload: {payload.model_dump_json(indent=2)}"
    )

    if (not payload.is_root) and (not payload.root_choices):
        raise NoRootChoiceProvided()

    result, sources = await autofill_meta_analysis_data(payload)

    logger.debug(f"Autofill meta-analysis data result: {result}")

    return {
        "message": "Autofill meta-analysis data",
        "data": result,
        "sources": sources,
    }
