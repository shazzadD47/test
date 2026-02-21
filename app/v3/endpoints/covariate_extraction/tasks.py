import time
from copy import deepcopy
from uuid import uuid4

from langfuse import observe

from app.core.celery.app import celery_app
from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.utils.decorators.cost_tracker import track_all_llm_costs
from app.utils import check_if_null
from app.v3.endpoints import Status
from app.v3.endpoints.covariate_extraction.constants import COVARIATE_ERROR_MESSAGE
from app.v3.endpoints.covariate_extraction.helpers.utils import (
    identify_trial_arm_column,
)
from app.v3.endpoints.covariate_extraction.logging import celery_logger as logger
from app.v3.endpoints.covariate_extraction.prompts import COVARIATE_INSTRUCTIONS
from app.v3.endpoints.general_extraction.services.tasks import (
    execute_general_extraction,
)


@observe()
def extract_covariate(
    project_id: str,
    paper_id: str,
    table_definition: list[dict],
    langfuse_session_id: str = None,
    metadata: dict = None,
    image_url: str | list[str] | None = None,
) -> dict:
    """
    Extract covariates using the general extraction pipeline.
    Supports extraction from paper text alone or with additional images.
    """
    start_time = time.time()
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    if metadata is None:
        metadata = {}

    ge_table_definition = deepcopy(table_definition)
    table_definition_trial_arm = identify_trial_arm_column(
        [covariate_info["name"] for covariate_info in ge_table_definition]
    )
    has_root_labels = any(label["c_type"] == "root" for label in ge_table_definition)

    for label in ge_table_definition:
        # if no root labels are present, then add the trial arm column as a root label
        if not has_root_labels and label["name"] == table_definition_trial_arm:
            label["c_type"] = "root"
        else:
            if label["c_type"] is None:
                label["c_type"] = "general"
        label["description"] += (
            ". Be very specific. "
            "Provide only the specific answer. "
            "Do not provide any other text or explanation."
        )
    logger.info(f"Table definition: {ge_table_definition}")

    # Prepare inputs for general extraction
    ge_inputs = {
        "flag_id": paper_id,
        "project_id": project_id,
        "table_structure": ge_table_definition,
        "metadata": metadata,
        "custom_instruction": COVARIATE_INSTRUCTIONS,
    }

    # Add image inputs if provided
    if image_url is not None:
        image_urls = [image_url] if isinstance(image_url, str) else image_url
        ge_inputs["inputs"] = [
            {
                "type": "image",
                "data": [{"figure_url": url} for url in image_urls],
            }
        ]

    logger.info(f"GE inputs: {ge_inputs}")

    result = execute_general_extraction(ge_inputs)
    total_time = time.time() - start_time
    logger.info(f"Time : covariate extraction: {total_time} seconds")

    if result["metadata"]["status"] == Status.FAILED.value:
        return {
            "data": None,
            "status": Status.FAILED.value,
            "message": "Covariate extraction failed.",
            "runtime": total_time,
            "metadata": metadata,
        }
    metadata.update(result["metadata"])
    return {
        "data": result["payload"],
        "message": "Covariate Extracted successfully",
        "status": Status.SUCCESS.value,
        "runtime": total_time,
        "metadata": metadata,
    }


@celery_app.task(name="extract_covariate_task")
@observe()
@track_all_llm_costs
def extract_covariate_task(
    paper_id: str,
    project_id: str,
    image_url: list[str] | str | None,
    table_definition: list[dict],
    langfuse_session_id: str = None,
    request_metadata: dict = None,
) -> dict:
    """
    Celery task for covariate extraction using the general extraction pipeline.
    """
    logger.info(f"Extracting covariate for paper {paper_id} and project {project_id}")

    metadata = {"ai_metadata": {"cost_metadata": {}}}

    # Merge request metadata if provided
    if request_metadata is not None:
        for key, value in request_metadata.items():
            if key not in metadata:
                metadata[key] = value

    try:
        if langfuse_session_id is None:
            langfuse_session_id = uuid4().hex

        # Normalize image_url - check for null/empty
        normalized_image_url = None
        if image_url is not None:
            if isinstance(image_url, list):
                # Filter out null/empty URLs
                valid_urls = [url for url in image_url if not check_if_null(url)]
                normalized_image_url = valid_urls if valid_urls else None
            elif not check_if_null(image_url):
                normalized_image_url = image_url

        # Route everything to general extraction pipeline
        result = extract_covariate(
            project_id=project_id,
            paper_id=paper_id,
            table_definition=table_definition,
            langfuse_session_id=langfuse_session_id,
            metadata=metadata,
            image_url=normalized_image_url,
        )

        # Format and order the output
        if result["status"] == Status.SUCCESS.value and result["data"] is not None:
            ordered_data = []
            for row in result["data"]:
                row_data = {}
                for field in table_definition:
                    row_data[field["name"]] = row.get(field["name"])
                ordered_data.append(row_data)
            result["data"] = ordered_data

        # Build final message
        final_metadata = result.get("metadata", metadata)
        final_metadata["message"] = result["message"]
        final_metadata["status"] = result["status"]
        final_metadata["ai_metadata"]["runtime"] = result["runtime"]

        message = {
            "payload": {"data": result["data"]} if result["data"] else {},
            "metadata": final_metadata,
        }

        send_to_backend(BackendEventEnumType.PRESET_AUTOFILL_COVERIATE, message)
        return message

    except Exception as e:
        logger.exception(f"Error occurred when extracting covariate: {e}")
        metadata["message"] = COVARIATE_ERROR_MESSAGE
        metadata["status"] = Status.FAILED.value
        metadata["ai_metadata"]["runtime"] = 0
        message = {
            "payload": {},
            "metadata": metadata,
        }
        send_to_backend(BackendEventEnumType.PRESET_AUTOFILL_COVERIATE, message)
        return message
