from fastapi import APIRouter, HTTPException, status
from fastapi.security import APIKeyHeader

from app.core.celery.app import celery_app
from app.v3.endpoints.tasks.schemas import TaskStatus, TaskStatusResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


@router.get(
    "/{task_id}/status",
    response_model=TaskStatusResponse,
    summary="Get Celery Task Status",
    description="Retrieve the status of a Celery task by its task ID",
)
async def get_task_status(task_id: str):
    """
    Get the status of a Celery task by its task ID.

    Args:
        task_id: The ID of the Celery task to check

    Returns:
        TaskStatusResponse: The status of the task

    Raises:
        HTTPException: If the task is not found
    """
    try:
        task_result = celery_app.AsyncResult(task_id)

        response = TaskStatusResponse(
            task_id=task_id,
            status=TaskStatus(task_result.status),
            result=task_result.result if task_result.successful() else None,
            error=str(task_result.result) if task_result.failed() else None,
            traceback=task_result.traceback if task_result.failed() else None,
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving task status: {str(e)}",
        )
