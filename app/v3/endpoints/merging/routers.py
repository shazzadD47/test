from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.auth.S2S_Client import S2SClient, S2SSecurityModel
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.pipeline import (
    run_merge,
    run_merge_and_qc,
    run_standardization_only,
)
from app.v3.endpoints.merging.schemas import (
    MergeRequest,
    MergeResponse,
    StandardizeSchema,
)

router = APIRouter(tags=["Merging"])


@router.post("/merge-tables")
def merge_tables(
    payload: MergeRequest,
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["merging"]))],
) -> MergeResponse:
    """
    Merge tables from payload. v0: merge+QC flow; v1 (or any other version): load
    from URLs, transform, merge by group/FILE_NAME, return CSV.
    """
    try:
        table_dict = payload.model_dump()
        if payload.version == "v0":
            return run_merge_and_qc(table_dict)
        return run_merge(table_dict)
    except Exception as e:
        logger.exception(f"Failed to merge tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/standardize-tables")
def standardize(
    payload: StandardizeSchema,
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["merging"]))],
):
    """
    Runs the standardization only process for a given table dictionary.
    """
    logger.info(f"Received payload for standardize-only: {payload.model_dump_json()}")
    try:
        return run_standardization_only(payload.table_values, payload.table_structure)
    except Exception as e:
        logger.exception(f"Failed to standardize tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))
