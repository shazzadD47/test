from typing import Annotated

from fastapi import APIRouter, Form
from starlette.responses import StreamingResponse

from app.v3.endpoints.rag_chat.services import generate_response_chunks

router = APIRouter(tags=["V3_Rag_app"])


@router.post("/improved-rag/")
async def improved_rag_endpoint(
    message: Annotated[str, Form()],
    project_id: Annotated[str | None, Form()] = None,
    flag_id: Annotated[str | None, Form()] = None,
    user_id: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
    description: Annotated[str | None, Form()] = None,
    system_prompt: Annotated[str | None, Form()] = None,
):
    """
    FastAPI endpoint to generate response chunks based on user's query using LLM model.
    """
    return StreamingResponse(
        generate_response_chunks(
            message, project_id, flag_id, user_id, title, description, system_prompt
        )
    )
