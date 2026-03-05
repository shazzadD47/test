from fastapi import APIRouter

from app.v3.endpoints.dynamic_dosing.logging import logger
from app.v3.endpoints.dynamic_dosing.schemas import DynamicDosingRequest
from app.v3.endpoints.dynamic_dosing.services import (
    extract_dynamic_dosing_from_tables_or_paper,
)

router = APIRouter(tags=["Dynamic Dosing"], include_in_schema=True)


@router.post(
    "/meta-analysis/dosing-table/dynamic/",
    summary="Extract Dosing Data with Dynamic Template",
)
async def get_meta_analysis_dynamic_dosing_table(
    data: DynamicDosingRequest,
):
    logger.info(f"[Dynamic Dosing API] Received payload: {data.model_dump_json()}")

    return await extract_dynamic_dosing_from_tables_or_paper(
        data=data,
    )
