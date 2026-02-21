from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, status

from app.auth.S2S_Client import S2SClient, S2SSecurityModel
from app.core.dependencies import validate_flag_id
from app.v3.endpoints.projects.exceptions import NoSummariesFound
from app.v3.endpoints.projects.logging import logger
from app.v3.endpoints.projects.schemas import AutoFigureConnection
from app.v3.endpoints.projects.services import (
    check_mineru_response_status,
    delete_flag_and_storage_service,
    delete_project_and_storage_service,
    get_paper_summaries_by_project,
    get_sub_figure_autodetect_data,
)

router = APIRouter(tags=["projects"])


@router.delete("/projects/{project_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_and_storage(project_id: str):
    await delete_project_and_storage_service(project_id)


@router.delete(
    "/projects/{project_id}/file/{flag_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_flag_and_storage(
    project_id: str,
    flag_id: Annotated[str, Depends(validate_flag_id)],
):
    await delete_flag_and_storage_service(project_id, flag_id)


@router.get("/sub-figure-data")
async def get_sub_figure_data(
    project_id: str, flag_id: Annotated[str, Depends(validate_flag_id)]
):
    logger.info(f"project_id : {project_id}")
    response_out = await check_mineru_response_status(flag_id)
    response_checking_fail = response_out["response_checking_fail"]
    mineru_status = response_out.get("response_status")

    if response_checking_fail:
        return_data = {"isAiProcessDone": True, "annotations": []}

        return return_data

    sub_figure_data = await get_sub_figure_autodetect_data(
        project_id, flag_id, mineru_status
    )

    return sub_figure_data


@router.post("/sub-figure-data-ai-suggested", status_code=status.HTTP_200_OK)
async def get_sub_figure_data_ai_suggested(
    project_id: Annotated[str, Form()],
    flag_id: Annotated[str, Depends(validate_flag_id), Form()],
    meta_data: Annotated[Any, Form()],
) -> AutoFigureConnection:
    logger.info(f"project_id : {project_id}")
    response_out = await check_mineru_response_status(flag_id)
    response_checking_fail = response_out["response_checking_fail"]
    mineru_status = response_out.get("response_status")

    if response_checking_fail:
        return_data = {"isAiProcessDone": True, "annotations": []}

        return return_data

    sub_figure_data = await get_sub_figure_autodetect_data(
        project_id, flag_id, mineru_status
    )

    return sub_figure_data


@router.get("/projects/{project_id}/summaries")
async def get_paper_summaries(
    project_id: str,
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["projects"]))],
):
    logger.info(
        f"Fetching paper summaries for project_id: {project_id}, client: {client}"
    )
    summaries = get_paper_summaries_by_project(project_id)

    if not summaries:
        raise NoSummariesFound()

    return summaries
