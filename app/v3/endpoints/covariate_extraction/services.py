from uuid import uuid4

from app.v3.endpoints.covariate_extraction.schemas import CovariateAutofillRequest
from app.v3.endpoints.covariate_extraction.tasks import extract_covariate_task


async def extract_covariate_from_tables_or_paper(
    data: CovariateAutofillRequest,
    langfuse_session_id: str = None,
) -> dict:
    """
    Extracts covariate data using the general extraction pipeline.
    Supports extraction from paper text with optional images.

    Args:
    - payload (CovariateAutofillRequest): The request payload.

    Returns:
    - dict: The processed covariate table data in a specific schema
    """
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex

    project_id = data.payload.project_id
    paper_id = data.payload.paper_id
    image_url = data.payload.image_url
    table_definition = [
        {
            "name": field.name,
            "description": field.description,
            "d_type": field.d_type,
            "c_type": field.c_type,
            "literal_options": field.literal_options,
        }
        for field in data.payload.table_definition
    ]

    # Normalize image_url to list or None
    if image_url is not None:
        image_urls = [image_url] if isinstance(image_url, str) else image_url
    else:
        image_urls = None

    workflow = extract_covariate_task.apply_async(
        kwargs={
            "paper_id": paper_id,
            "project_id": project_id,
            "image_url": image_urls,
            "table_definition": table_definition,
            "langfuse_session_id": langfuse_session_id,
            "request_metadata": data.metadata,
        }
    )

    metadata = data.metadata
    metadata["message"] = "Covariate extraction process started in Background"
    metadata["ai_metadata"] = {"task_id": workflow.id}

    return metadata
