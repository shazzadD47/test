from app.v3.endpoints.auto_suggestions.services.tasks import (
    auto_figure_suggestion_task,
)


async def auto_figure_selection(
    table_id: str,
    file_id: str,
    flag_id: str,
    table_structure: dict,
    project_id: str,
    metadata: dict,
):

    task = auto_figure_suggestion_task.apply_async(
        kwargs={
            "table_id": table_id,
            "file_id": file_id,
            "flag_id": flag_id,
            "table_structure": table_structure,
            "project_id": project_id,
            "metadata": metadata,
        }
    )
    metadata = {
        "ai_metadata": {"task_id": task.id},
        "input_metadata": metadata,
        "message": "Auto Figure Suggestion Started in background",
    }
    return metadata
