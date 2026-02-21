from typing import Annotated

from fastapi import APIRouter, Depends

from app.auth.S2S_Client import S2SClient, S2SSecurityModel
from app.v3.endpoints.covariate_extraction.end_to_end_adverse_digitizer import (
    end_to_end_adverse_digitizer,
)
from app.v3.endpoints.covariate_extraction.helpers.helpers import (
    check_if_covariate_table,
)
from app.v3.endpoints.covariate_extraction.logging import logger
from app.v3.endpoints.covariate_extraction.schemas import CovariateAutofillRequest
from app.v3.endpoints.covariate_extraction.services import (
    extract_covariate_from_tables_or_paper,
)

router = APIRouter(tags=["Covariate Extraction"], include_in_schema=True)


@router.post(
    "/meta-analysis/covariate-extraction/",
    summary="Extract Covariate from Tables",
)
async def get_meta_analysis_covariate_table(
    data: CovariateAutofillRequest,
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["meta-analysis"]))],
):

    logger.info(
        f"[Covariate API] Received payload: {data.model_dump_json()} "
        f"from client {client.client_id}"
    )
    logger.debug(f"table_definition: {data}")
    # see if it is not a covariate table
    is_cov_table = check_if_covariate_table(data.payload.table_definition)
    if is_cov_table:
        return await extract_covariate_from_tables_or_paper(
            data=data,
        )
    else:
        logger.info("Routing to adverse event extraction pipeline")
        logger.info(f"image_url: {data.payload.image_url}")
        if isinstance(data.payload.image_url, list):
            data.payload.image_url = data.payload.image_url[0]
        data.payload.image_url = str(data.payload.image_url)

        return await end_to_end_adverse_digitizer(
            data.payload.image_url,
            data.payload.paper_id,
            data.payload.project_id or "",
            data.payload.table_definition,
            metadata=data.metadata,
        )
