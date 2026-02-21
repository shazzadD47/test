from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.auth.S2S_Client import S2SClient, S2SSecurityModel
from app.v3.endpoints.column_standardization.exceptions import (
    ColumnStandardizationTaskFailedException,
)
from app.v3.endpoints.column_standardization.logging import logger
from app.v3.endpoints.column_standardization.schemas import (
    ColumnStandardizationRequest,
    ColumnStandardizationTaskResponse,
)
from app.v3.endpoints.column_standardization.services.tasks import (
    column_standardization_task,
)

router = APIRouter(tags=["column_standardization"])


@router.post(
    "/standardization/columns/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ColumnStandardizationTaskResponse,
)
async def standardize_columns_background(
    request: ColumnStandardizationRequest,
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["column-standardization"]))],
):
    """
    Start column standardization task for the given table column in background
    """
    logger.info(
        f"Received column standardization request for \n\
            table_id: {request.table_id}, \n\
            column_name: {request.column_name}, \n\
            column_description: {request.column_description}, \n\
            user_prompt: {request.usr_instruction},"
    )

    try:
        # Start the background task
        task = column_standardization_task.delay(
            table_id=request.table_id,
            col_name=request.column_name,
            col_desc=request.column_description,
            usr_instruction=request.usr_instruction,
        )

        return ColumnStandardizationTaskResponse(
            message="Column standardization task has been started in the background",
            task_id=task.id,
            table_id=request.table_id,
            column_name=request.column_name,
            column_description=request.column_description,
            usr_instruction=request.usr_instruction,
        )
    except Exception as e:
        logger.exception(f"Failed to start column standardization task: {e}")
        raise ColumnStandardizationTaskFailedException()
