from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.auth.S2S_Client import S2SClient, S2SSecurityModel
from app.logging import logger
from app.v3.endpoints.tag_extraction.exceptions import TagExtractionTaskFailedException
from app.v3.endpoints.tag_extraction.schemas import TagExtractionRequest
from app.v3.endpoints.tag_extraction.services import start_tag_extraction

router = APIRouter(tags=["Tag Extraction"])


@router.post(
    "/tag-extraction/",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Extract tags by flag_ids",
)
async def tag_extraction(
    data: TagExtractionRequest,
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["tag-extraction"]))],
):
    logger.info(f"Received tag extraction request for {len(data.flag_ids)} flag_ids")
    try:
        result = start_tag_extraction(data)
        return result
    except Exception as e:
        logger.exception(f"Failed to start tag extraction task: {e}")
        raise TagExtractionTaskFailedException()
