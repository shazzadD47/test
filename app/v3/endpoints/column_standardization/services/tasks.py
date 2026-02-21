import time
from collections import defaultdict
from typing import Any

import httpx
from celery import shared_task
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.configs import settings
from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.utils.decorators.cost_tracker import track_all_llm_costs
from app.v3.endpoints.column_standardization.configs import settings as cs_settings
from app.v3.endpoints.column_standardization.exceptions import (
    BackendAPIError,
    ProcessingError,
)
from app.v3.endpoints.column_standardization.logging import celery_logger as logger
from app.v3.endpoints.column_standardization.schemas import (
    ColumnChange,
    ColumnStandardizationCompletedData,
    ColumnStandardizationFailedData,
    ExtractionMetadata,
    ProcessedExtraction,
    StandardizationSummary,
)
from app.v3.endpoints.column_standardization.services.column_standardization import (
    ColumnStandardizer,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def _fetch_backend_data(backend_url: str) -> list[dict[str, Any]]:
    """Fetch data from backend with retry logic."""
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.get(backend_url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise BackendAPIError(
            f"Backend API error {e.response.status_code}: {e.response.text}"
        )
    except httpx.RequestError as e:
        raise BackendAPIError(f"Backend request failed: {str(e)}")
    except Exception as e:
        raise BackendAPIError(f"Unexpected error while fetching backend data: {str(e)}")


def _send_backend_message(data: dict[str, Any], table_id: str) -> None:
    """Send message to backend via RabbitMQ."""
    try:
        send_to_backend(
            event=BackendEventEnumType.COLUMN_STANDARDIZATION,
            message=data,
            queue_name=settings.RMQ_QUEUE,
        )
    except Exception as e:
        logger.error(f"Failed to send backend message for {table_id}: {e}")


def _extract_cost_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Extract cost metadata from decorator result."""
    if "result" in result and "metadata" in result:
        return result["metadata"].get("ai_metadata", {}).get("cost_metadata", {})
    elif "metadata" in result and "ai_metadata" in result["metadata"]:
        return result["metadata"]["ai_metadata"].get("cost_metadata", {})
    return {}


def process_column_standardization_data(
    data: list[dict[str, Any]],
    column_name: str,
    column_description: str | None,
    user_instruction: str | None,
) -> list[ProcessedExtraction]:
    """Process column standardization data."""
    if not data:
        return []

    logger.info(f"Processing {len(data)} extractions for column '{column_name}'")

    try:
        standardizer = ColumnStandardizer(
            column_name=column_name,
            column_description=column_description,
            user_instruction=user_instruction,
        )

        column_values: dict[str, set[str]] = {}
        value_locations = []

        for extraction in data:
            extraction_id = extraction.get("metadata", {}).get("extractionId")
            if not extraction_id:
                logger.warning("Skipping extraction without extractionId")
                continue

            for label_idx, label_data in enumerate(extraction.get("labels", [])):
                if column_name in label_data:
                    value = label_data[column_name]

                    value_locations.append(
                        {
                            "extraction_id": extraction_id,
                            "label_index": label_idx,
                            "label_key": column_name,
                            "original_value": value,
                        }
                    )

                    if value is not None and str(value).strip():
                        column_values.setdefault(column_name, set()).add(str(value))

        if not column_values:
            logger.info(
                f"No non-empty values found in column '{column_name}' "
                f"for standardization"
            )
            return []

        logger.info(
            f"Found {len(column_values.get(column_name, set()))} unique values "
            f"in column '{column_name}'"
        )

        standardization_mapping = standardizer.standardize_column_values(column_values)

        extraction_changes = defaultdict(lambda: defaultdict(set))

        for extraction in data:
            extraction_id = extraction.get("metadata", {}).get("extractionId")
            if not extraction_id:
                continue

            for loc in value_locations:
                if loc["extraction_id"] != extraction_id:
                    continue

                original_value = loc["original_value"]

                if original_value is None or str(original_value).strip() == "":
                    standardized_value = "N/A"
                else:
                    original_value_str = str(original_value)
                    standardized_value = standardization_mapping.get(
                        column_name, {}
                    ).get(original_value_str, original_value_str)

                if str(standardized_value) != str(original_value):
                    extraction["labels"][loc["label_index"]][
                        column_name
                    ] = standardized_value

                    extraction_changes[extraction_id][column_name].add(
                        (str(original_value), str(standardized_value))
                    )

        result = []
        for extraction in data:
            extraction_id = extraction.get("metadata", {}).get("extractionId")
            if extraction_id in extraction_changes:
                metadata = ExtractionMetadata(**extraction["metadata"])
                changed_parameter = {
                    param: [
                        ColumnChange(prev=prev, current=curr) for prev, curr in changes
                    ]
                    for param, changes in extraction_changes[extraction_id].items()
                }
                result.append(
                    ProcessedExtraction(
                        metadata=metadata, changed_parameter=changed_parameter
                    )
                )

        logger.info(
            f"Processed {len(result)} extractions with changes "
            f"for column '{column_name}'"
        )
        return result

    except Exception as e:
        logger.exception(f"Processing failed for column '{column_name}': {e}")
        raise ProcessingError(f"Column standardization failed: {str(e)}")


@track_all_llm_costs
def execute_column_standardization(
    table_id: str,
    column_name: str,
    column_description: str | None,
    user_instruction: str | None,
    task_id: str,
) -> dict[str, Any]:
    """Execute column standardization with cost tracking."""

    start_time = time.time()

    logger.info(
        f"Starting column standardization for table_id: {table_id}, "
        f"column: {column_name}"
    )

    endpoint = cs_settings.BACKEND_COLUMN_STD_URL.format(
        table_id=table_id, col_name=column_name
    )
    backend_url = f"{settings.BACKEND_BASE_URL}/{endpoint}"
    logger.info(f"Calling backend API: {backend_url}")

    try:
        data = _fetch_backend_data(backend_url)
        logger.info(f"Retrieved {len(data)} extractions from backend")
    except BackendAPIError as e:
        logger.error(
            f"Failed to fetch data for table_id {table_id}, "
            f"column {column_name}: {str(e)}"
        )
        processing_time = time.time() - start_time
        return {
            "status": "FAILED",
            "payload": ColumnStandardizationFailedData(
                table_id=table_id,
                column_name=column_name,
                column_description=column_description or "N/A",
                usr_instruction=user_instruction or "N/A",
                task_id=task_id,
                status="failed",
                error=str(e),
            ),
            "processing_time": processing_time,
        }

    try:
        processed_data = process_column_standardization_data(
            data=data,
            column_name=column_name,
            column_description=column_description,
            user_instruction=user_instruction,
        )
    except ProcessingError as e:
        logger.error(f"Processing error: {str(e)}")
        processing_time = time.time() - start_time
        return {
            "status": "FAILED",
            "payload": ColumnStandardizationFailedData(
                table_id=table_id,
                column_name=column_name,
                column_description=column_description or "N/A",
                usr_instruction=user_instruction or "N/A",
                task_id=task_id,
                status="failed",
                error=str(e),
            ),
            "processing_time": processing_time,
        }

    total_column_changes = sum(
        len(changes)
        for extraction in processed_data
        for changes in extraction.changed_parameter.values()
    )

    summary = StandardizationSummary(
        total_extractions=len(data),
        total_standardized_extractions=len(processed_data),
        total_column_changes=total_column_changes,
    )

    payload = ColumnStandardizationCompletedData(
        table_id=table_id,
        task_id=task_id,
        status="completed",
        processed_data=processed_data,
        summary=summary,
    )

    processing_time = time.time() - start_time
    logger.info(
        f"Column standardization completed in {processing_time:.2f}s "
        f"({total_column_changes} changes)"
    )

    return {
        "status": "SUCCESS",
        "payload": payload,
        "processing_time": processing_time,
    }


@shared_task(
    name="column_standardization_task",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(BackendAPIError,),
)
def column_standardization_task(
    self,
    table_id: str,
    col_name: str,
    col_desc: str | None,
    usr_instruction: str | None,
) -> dict[str, Any]:
    """
    Celery task for column standardization.

    Args:
        table_id: ID of the table to process
        col_name: Name of the column to standardize
        col_desc: Optional description of the column
        usr_instruction: Optional user-specific standardization instructions
    """
    try:
        result = execute_column_standardization(
            table_id=table_id,
            column_name=col_name,
            column_description=col_desc,
            user_instruction=usr_instruction,
            task_id=self.request.id,
        )

        cost_metadata = _extract_cost_metadata(result)
        payload = (
            result["result"]["payload"] if "result" in result else result["payload"]
        )

        backend_message = payload.model_dump()
        if cost_metadata:
            backend_message["cost_metadata"] = cost_metadata

        _send_backend_message(backend_message, table_id)

        logger.info(
            f"Column standardization completed successfully for "
            f"table_id: {table_id}, column: {col_name}"
        )

    except BackendAPIError as e:
        logger.error(
            f"Backend API error for table_id {table_id}, column {col_name} "
            f"(attempt {self.request.retries + 1}/{self.max_retries}): {e}"
        )

        if self.request.retries >= self.max_retries:
            failed_data = ColumnStandardizationFailedData(
                table_id=table_id,
                column_name=col_name,
                column_description=col_desc or "N/A",
                usr_instruction=usr_instruction or "N/A",
                task_id=self.request.id,
                status="failed",
                error=f"Backend API error after {self.max_retries} retries: {str(e)}",
            )
            _send_backend_message(failed_data.model_dump(), table_id)

        raise

    except Exception as e:
        logger.error(
            f"Unexpected error for table_id {table_id}, column {col_name}: {e}"
        )

        failure_data = ColumnStandardizationFailedData(
            table_id=table_id,
            column_name=col_name,
            column_description=col_desc or "N/A",
            usr_instruction=usr_instruction or "N/A",
            task_id=self.request.id,
            status="failed",
            error=f"Unexpected error: {str(e)}",
        )
        _send_backend_message(failure_data.model_dump(), table_id)
