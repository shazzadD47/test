from fastapi import APIRouter
from pydantic import HttpUrl

from app.utils.token_usage import track_openai_usage
from app.v3.endpoints.dosing_table.logging import logger
from app.v3.endpoints.dosing_table.schemas import (
    DosingTableRequest,
    NoFigureDosingTableRequest,
)
from app.v3.endpoints.dosing_table.services import (
    prepare_dosing_table,
)
from app.v3.endpoints.dosing_table.services.tasks import (
    prepare_dosing_table_no_figure_task,
    prepare_dosing_table_with_figure_task,
)

router = APIRouter(tags=["Dosing Table"])


@router.get("/meta-analysis/dosing-table/")
@track_openai_usage
def get_meta_analysis_dosing_table(
    project_id: str,
    paper_id: str,
    image_url: HttpUrl,
):
    logger.debug(f"GOT IMAGE URL: {image_url}")
    response = prepare_dosing_table(project_id, paper_id, image_url.unicode_string())
    logger.debug(f"RESPONSE: {response}")

    return response


@router.post("/meta-analysis/dosing-table/")
async def get_meta_analysis_dosing_table_with_figure(
    payload: DosingTableRequest,
):
    task = prepare_dosing_table_with_figure_task.apply_async(
        args=[
            payload.project_id,
            payload.flag_id,
            payload.image_url.unicode_string(),
            payload.metadata,
        ],
    )

    metadata = payload.metadata

    metadata["message"] = "Dosing table with figure process started in Background"
    metadata["ai_metadata"] = {"task_id": task.id}

    return metadata


@router.post("/meta-analysis/dosing-table/no-figure/")
async def get_meta_analysis_dosing_table_no_figure(
    payload: NoFigureDosingTableRequest,
):
    task = prepare_dosing_table_no_figure_task.apply_async(
        args=[payload.project_id, payload.flag_id, payload.metadata],
    )

    metadata = payload.metadata

    metadata["message"] = "Dosing table no figure process started in Background"
    metadata["ai_metadata"] = {"task_id": task.id}

    return metadata
