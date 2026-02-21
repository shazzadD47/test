from fastapi import APIRouter, Form
from starlette.responses import StreamingResponse

from app.v3.endpoints.case_study_rag.services import generate_response_chunks

router = APIRouter(tags=["V3_Rag_app"])


@router.post("/case-study-rag/")
async def case_study_rag_endpoint(
    message: str = Form(...),
    case_study_id: str | None = Form(None),
    project_id: str | None = Form(None),
):
    return StreamingResponse(
        generate_response_chunks(message, case_study_id, project_id)
    )
