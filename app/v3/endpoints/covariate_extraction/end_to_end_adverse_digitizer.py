from app.v3.endpoints.covariate_extraction.adverse_event import (
    adverse_event_extraction_service,
)
from app.v3.endpoints.covariate_extraction.schemas import MetaAnalysisTableField


async def end_to_end_adverse_digitizer(
    figure_url: str,
    paper_id: str,
    project_id: str,
    table_structure: list[MetaAnalysisTableField],
    metadata: dict = None,
):
    table_structure = [ts.model_dump() for ts in table_structure]

    task_chain = adverse_event_extraction_service.apply_async(
        kwargs={
            "figure_url": figure_url,
            "paper_id": paper_id,
            "project_id": project_id,
            "table_structure": table_structure,
            "metadata": metadata,
        }
    )
    if metadata is None:
        metadata = {}
    metadata["ai_metadata"] = {"task_id": task_chain.id}
    metadata["message"] = "Adverse Event Extraction Started in background"
    return metadata
