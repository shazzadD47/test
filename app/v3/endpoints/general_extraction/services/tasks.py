import time

from celery import shared_task

from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.utils.decorators.cost_tracker import track_all_llm_costs
from app.v3.endpoints import Status
from app.v3.endpoints.general_extraction.configs import settings as ge_settings
from app.v3.endpoints.general_extraction.constants import ErrorCode
from app.v3.endpoints.general_extraction.logging import celery_logger as logger
from app.v3.endpoints.general_extraction.services.graph import get_graph
from app.v3.endpoints.general_extraction.services.helpers.task_helpers import (
    modify_inputs_for_root_extraction,
    modify_response_for_non_root_extraction,
)


@track_all_llm_costs
def execute_general_extraction(
    inputs: dict,
):
    """
    Execute general extraction task
    Args:
        inputs: dict: Inputs for general extraction
    Returns:
        dict: Output of general extraction
    """

    if "metadata" not in inputs or not inputs["metadata"]:
        inputs["metadata"] = {}
    metadata = inputs["metadata"]
    if "ai_metadata" not in metadata:
        metadata["ai_metadata"] = {}

    send_root_labels_first = inputs.get("metadata", {}).get(
        "send_root_labels_first", False
    )
    start_time = time.time()
    try:
        final_workflow = get_graph()
        response = final_workflow.invoke(
            {
                "workflow_input": inputs,
                "table_structure": inputs["table_structure"],
            }
        )

        result = response.get("workflow_input").get("final_table")
        result_with_citations = response.get("workflow_input").get(
            "final_table_with_citations"
        )
        logger.info("General extraction process completed.")
        metadata["message"] = "General extraction process completed"
        metadata["status"] = Status.SUCCESS.value
        total_time = time.time() - start_time
        metadata["ai_metadata"]["runtime"] = total_time
        metadata["ai_metadata"]["result_with_citations"] = result_with_citations
        if send_root_labels_first:
            message = {
                "payload": result,
                "metadata": metadata,
                "workflow_response": response,
            }
        else:
            message = {
                "payload": result,
                "metadata": metadata,
            }
        return message

    except Exception as e:
        logger.exception(f"General extraction process failed. Error: {e}")
        error_message = ErrorCode.GENERAL_EXTRACTION_ERROR_MESSAGE
        error_message += "Failed to extract table labels after all retries"
        metadata["message"] = error_message
        metadata["status"] = Status.FAILED.value
        total_time = time.time() - start_time
        metadata["ai_metadata"]["runtime"] = total_time
        message = {
            "payload": {},
            "metadata": metadata,
            "workflow_response": {},
        }
        return message


@shared_task(
    name="general_extraction_task",
    bind=True,
    max_retries=0,
    default_retry_delay=30,
)
def general_extraction_task(
    self,
    inputs: dict,
):
    """
    Celery task for general extraction. It extracts table labels
    from the given inputs.
    """
    if ("metadata" not in inputs) or (not inputs["metadata"]):
        inputs["metadata"] = {}

    retry_count = 0
    while retry_count < ge_settings.MAX_RETRIES:
        try:
            send_root_labels_first = inputs.get("metadata", {}).get(
                "send_root_labels_first", False
            )
            if send_root_labels_first:
                output = execute_general_extraction(
                    modify_inputs_for_root_extraction(inputs)
                )
            else:
                output = execute_general_extraction(inputs)

            if (
                "metadata" in output
                and "status" in output["metadata"]
                and output["metadata"]["status"] == Status.SUCCESS.value
            ):
                if send_root_labels_first:
                    send_to_backend(
                        BackendEventEnumType.PRESET_GENERAL_EXTRACTION_ROOT_LABELS,
                        output,
                    )
                    response = output["workflow_response"].get("workflow_input")
                    prev_generate_labels = inputs["metadata"].get("generate_labels", [])
                    prev_table_structure = inputs["table_structure"]
                    response = modify_response_for_non_root_extraction(
                        response,
                        prev_generate_labels,
                        prev_table_structure,
                    )
                    # Instead of recursive call, loop with new inputs
                    # (avoids holding both phases in memory simultaneously)
                    del output  # free memory from phase 1
                    inputs = response
                    if ("metadata" not in inputs) or (not inputs["metadata"]):
                        inputs["metadata"] = {}
                    retry_count = 0
                    continue
                else:
                    send_to_backend(
                        BackendEventEnumType.PRESET_GENERAL_EXTRACTION, output
                    )
                return output
            else:
                retry_count += 1
                retry_delay = ge_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
                time.sleep(retry_delay)
                if "metadata" in output and isinstance(output["metadata"], dict):
                    inputs["metadata"].update(output["metadata"])
                if retry_count >= ge_settings.MAX_RETRIES:
                    send_to_backend(
                        BackendEventEnumType.PRESET_GENERAL_EXTRACTION, output
                    )
                    return output
                continue
        except Exception as e:
            logger.exception(f"General extraction process failed. Error: {e}")
            retry_count += 1
            retry_delay = ge_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
            time.sleep(retry_delay)
            if retry_count >= ge_settings.MAX_RETRIES:
                output = {
                    "payload": {},
                    "metadata": inputs["metadata"],
                }
                output["metadata"]["status"] = Status.FAILED.value
                output["metadata"][
                    "message"
                ] = ErrorCode.GENERAL_EXTRACTION_ERROR_MESSAGE
                send_to_backend(BackendEventEnumType.PRESET_GENERAL_EXTRACTION, output)
                return output
            continue
