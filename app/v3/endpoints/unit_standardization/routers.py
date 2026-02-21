from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.auth.S2S_Client import S2SClient, S2SSecurityModel
from app.v3.endpoints.unit_standardization.exceptions import (
    UnitStandardizationTaskFailedException,
)
from app.v3.endpoints.unit_standardization.logging import logger
from app.v3.endpoints.unit_standardization.schemas import (
    UnitStandardizationRequest,
    UnitStandardizationTaskResponse,
)
from app.v3.endpoints.unit_standardization.services.tasks import (
    unit_standardization_task,
)

router = APIRouter(tags=["unit_standardization"])


@router.post(
    "/standardization/units/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UnitStandardizationTaskResponse,
)
async def standardize_units_background(
    request: UnitStandardizationRequest,
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["unit-standardization"]))],
):
    """
    Start unit standardization task for the given table_id in background
    """
    table_id = request.table_id
    logger.info(f"Received unit standardization request for table_id: {table_id}")

    try:
        # Start the background task
        task = unit_standardization_task.delay(table_id)

        return UnitStandardizationTaskResponse(
            message="Unit standardization task has been started in the background",
            task_id=task.id,
            table_id=table_id,
        )
    except Exception as e:
        logger.exception(f"Failed to start unit standardization task: {e}")
        raise UnitStandardizationTaskFailedException()
