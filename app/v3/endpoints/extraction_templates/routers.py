from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from app.auth.user_auth import AuthenticatedUser, UserAuthClient
from app.v3.endpoints.extraction_templates.schemas import (
    CreateExtractionTemplateRequest,
)
from app.v3.endpoints.extraction_templates.services import chat_extraction_template

router = APIRouter(
    prefix="/meta-analysis",
    tags=["extraction-templates"],
)


@router.post(
    "/templates/chat",
    description="Create a new extraction template with human in the loop chat",
)
async def create_extraction_template(
    request: CreateExtractionTemplateRequest,
    user: AuthenticatedUser = Depends(UserAuthClient()),
):
    """
    Stream extraction template chat responses using SSE.
    """
    return StreamingResponse(
        chat_extraction_template(
            request.user_query,
            str(request.thread_id),
            user.token,
            str(request.project_id),
            task_type=request.task_type,
            table_name=request.table_name,
            table_description=request.table_description,
            input_rows=request.input_rows,
            output_rows=request.output_rows,
        ),
        media_type="text/event-stream",
    )
