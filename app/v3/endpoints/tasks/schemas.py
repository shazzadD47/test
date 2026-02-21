from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REVOKED = "REVOKED"
    RETRY = "RETRY"
    RECEIVED = "RECEIVED"


class TaskStatusResponse(BaseModel):
    task_id: str = Field(..., description="The ID of the Celery task")
    status: TaskStatus = Field(..., description="The current status of the task")
    result: Any | None = Field(None, description="The result of the task if completed")
    error: str | None = Field(None, description="Error message if the task failed")
    traceback: str | None = Field(None, description="Traceback if the task failed")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "SUCCESS",
                "result": {"some_key": "some_value"},
                "error": None,
                "traceback": None,
            }
        }
