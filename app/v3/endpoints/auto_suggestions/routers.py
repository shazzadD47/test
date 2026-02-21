from fastapi import APIRouter, status

from app.v3.endpoints.auto_suggestions.schemas import AutoFigureSuggestionRequest
from app.v3.endpoints.auto_suggestions.services.auto_figure_selection import (
    auto_figure_selection,
)
from app.v3.endpoints.projects.logging import logger

router = APIRouter(tags=["auto_suggestions"])


@router.post("/suggestions/figures", status_code=status.HTTP_200_OK)
async def get_auto_figure_suggestion(input_data: AutoFigureSuggestionRequest):
    logger.debug(f"Received payload for: {input_data.model_dump_json()}")

    metadata = await auto_figure_selection(
        table_id=input_data.payload.table_id,
        file_id=input_data.payload.file_id,
        flag_id=input_data.payload.flag_id,
        table_structure=input_data.payload.table_structure,
        project_id=input_data.payload.project_id,
        metadata=input_data.metadata,
    )
    return metadata
