from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, status

router = APIRouter(tags=["Delete"])


@router.post("/delete_project_id/", deprecated=True)
async def delete_project_id(
    project_id: Annotated[str, Form()], flag_id: Annotated[str, Form()]
):
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="This endpoint is deprecated",
    )
