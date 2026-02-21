from app.core.celery.app import celery_app
from app.v3.endpoints.iterative_autofill.schemas import IterativeAutofillRequest


def iterative_metadata_extraction_service(data: IterativeAutofillRequest) -> dict:
    (
        paper_id,
        project_id,
        table_structure,
        prev_response,
    ) = (
        data.payload.paper_id,
        data.payload.project_id,
        data.payload.table_structure,
        data.payload.prev_response,
    )
    table_definition = [
        (
            {
                "name": field.name,
                "description": field.description,
                "d_type": field.d_type,
                "generate": field.generate,
                "relationships": [
                    {
                        "related_label": relationship.related_label,
                        "description": relationship.description,
                    }
                    for relationship in field.relationships
                ],
            }
            if field.relationships is not None
            else {
                "name": field.name,
                "description": field.description,
                "d_type": field.d_type,
                "generate": field.generate,
                "relationships": None,
            }
        )
        for field in table_structure
    ]
    async_result = celery_app.send_task(
        "iterative autofill metadata extraction task",
        kwargs={
            "paper_id": paper_id,
            "project_id": project_id,
            "table_structure": table_definition,
            "prev_response": prev_response,
            "metadata": data.metadata,
        },
    )
    task_id = async_result.id
    ai_metadata = {
        "task_id": task_id,
    }
    metadata = data.metadata
    metadata["message"] = "Iterative metadata extraction process started in Background"
    metadata["ai_metadata"] = ai_metadata
    return metadata
