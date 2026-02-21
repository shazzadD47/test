from uuid import uuid4

from app.configs import settings
from app.v3.endpoints.plot_digitizer.schemas import (
    PlotBounds,
    PlotDigitizerTableField,
)
from app.v3.endpoints.plot_digitizer.services.digitizer_autofill_tasks import (
    dynamic_metadata_extraction_service,
)

REDIS_LIVE_TIME = settings.CACHE_DAY * 24 * 60 * 60


async def end_to_end_autofill_digitizer(
    figure_url: list[str] | str,
    paper_id: str,
    project_id: str,
    table_structure: list[PlotDigitizerTableField],
    page_no: int = None,
    bounding_box: PlotBounds = None,
    legend_urls: list[str] = None,
    bounding_box_legends: list[PlotBounds] = None,
    run_autofill: bool = True,
    run_digitization: bool = True,
    line_names_to_extract: list[dict] = None,
    generate_labels: list[str] = None,
    chart_type: str = None,
    metadata: dict = None,
):

    table_structure = [ts.model_dump() for ts in table_structure]
    bounding_box = bounding_box.model_dump() if bounding_box else None
    bounding_box_legends = (
        [bb.model_dump() for bb in bounding_box_legends]
        if bounding_box_legends and len(bounding_box_legends) > 0
        else None
    )
    if isinstance(legend_urls, list) and len(legend_urls) == 0:
        legend_urls = None

    if metadata is None:
        metadata = {}

    task = dynamic_metadata_extraction_service.apply_async(
        kwargs={
            "figure_url": figure_url,
            "paper_id": paper_id,
            "project_id": project_id,
            "table_structure": table_structure,
            "page_no": page_no,
            "bounding_box": bounding_box,
            "legend_urls": legend_urls,
            "bounding_box_legends": bounding_box_legends,
            "run_autofill": run_autofill,
            "run_digitization": run_digitization,
            "line_names_to_extract": line_names_to_extract,
            "generate_labels": generate_labels,
            "chart_type": chart_type,
            "langfuse_session_id": uuid4().hex,
            "metadata": metadata,
        },
    )
    metadata["ai_metadata"] = {"task_id": task.id}
    metadata["message"] = "Plot Label and Line Points Extraction Started in background"
    return metadata
