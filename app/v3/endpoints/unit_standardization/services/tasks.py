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
from app.v3.endpoints.unit_standardization.configs import settings as uc_settings
from app.v3.endpoints.unit_standardization.exceptions import (
    BackendAPIError,
    ProcessingError,
)
from app.v3.endpoints.unit_standardization.logging import celery_logger as logger
from app.v3.endpoints.unit_standardization.schemas import (
    ExtractionMetadata,
    ProcessedExtraction,
    StandardizationSummary,
    UnitChange,
    UnitStandardizationCompletedData,
    UnitStandardizationFailedData,
)
from app.v3.endpoints.unit_standardization.services.unit_standardization import (
    UnitStandardizer,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def _fetch_backend_data(backend_url: str) -> list[dict[str, Any]]:
    """Fetch data from backend with retry logic."""
    try:
        with httpx.Client(timeout=30.0) as client:
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
    """Send message to backend"""
    try:
        send_to_backend(
            event=BackendEventEnumType.UNIT_STANDARDIZATION,
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


@track_all_llm_costs
def execute_unit_standardization(table_id: str, task_id: str) -> dict[str, Any]:
    """Execute unit standardization with cost tracking."""

    start_time = time.time()

    logger.info(f"Starting unit standardization for table_id: {table_id}")

    # Fetch data
    endpoint = uc_settings.BACKEND_UNIT_STD_URL.format(table_id=table_id)
    backend_url = f"{settings.BACKEND_BASE_URL}/{endpoint}"
    logger.info(f"Calling backend API: {backend_url}")

    try:
        data = _fetch_backend_data(backend_url)
    except BackendAPIError as e:
        logger.error(f"Failed to fetch data for table_id {table_id}: {str(e)}")
        processing_time = time.time() - start_time
        return {
            "status": "FAILED",
            "payload": UnitStandardizationFailedData(
                table_id=table_id,
                task_id=task_id,
                status="failed",
                error=str(e),
            ),
            "processing_time": processing_time,
        }

    logger.info(
        f"Retrieved data for table_id: {table_id}, found {len(data)} extractions"
    )

    # Process data
    processed_data = process_unit_standardization_data(data)

    # Calculate Summary
    total_unit_changes = sum(
        len(changes)
        for extraction in processed_data
        for changes in extraction.changed_parameter.values()
    )

    summary = StandardizationSummary(
        total_extractions=len(data),
        total_standardized_extractions=len(processed_data),
        total_unit_changes=total_unit_changes,
    )

    payload = UnitStandardizationCompletedData(
        table_id=table_id,
        task_id=task_id,
        status="completed",
        processed_data=processed_data,
        summary=summary,
    )
    processing_time = time.time() - start_time
    logger.info(f"Unit standardization completed in {processing_time:.2f}s")

    return {
        "status": "SUCCESS",
        "payload": payload,
        "processing_time": processing_time,
    }


@shared_task(
    name="unit_standardization_task",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(BackendAPIError,),
)
def unit_standardization_task(self, table_id: str) -> dict[str, Any]:
    try:
        # Execute with cost tracking
        result = execute_unit_standardization(table_id, self.request.id)

        # Extract cost metadata
        cost_metadata = _extract_cost_metadata(result)
        payload = (
            result["result"]["payload"] if "result" in result else result["payload"]
        )

        backend_message = payload.model_dump()

        if cost_metadata:
            backend_message["cost_metadata"] = cost_metadata

        _send_backend_message(backend_message, table_id)

        logger.info(
            f"Unit standardization completed successfully for table_id: {table_id}"
        )

        return {
            "status": "SUCCESS",
            "table_id": table_id,
            "task_id": self.request.id,
            "cost_metadata": cost_metadata,  # Include in task response
        }

    except BackendAPIError as e:
        logger.error(
            f"Backend API error for table_id {table_id} "
            f"try {self.request.retries + 1}/{self.max_retries}: {e}"
        )

        if self.request.retries >= self.max_retries:
            # Send failure message
            failed_data = UnitStandardizationFailedData(
                table_id=table_id,
                task_id=self.request.id,
                status="failed",
                error=f"Backend API error: {str(e)}",
            )
            _send_backend_message(failed_data.model_dump(), table_id)
        raise

    except Exception as e:
        logger.error(f"Unexpected error for table_id {table_id}: {e}")

        # Send failure message
        failure_data = UnitStandardizationFailedData(
            table_id=table_id,
            task_id=self.request.id,
            status="failed",
            error=f"Unexpected error: {str(e)}",
        )
        _send_backend_message(failure_data.model_dump(), table_id)
        # Don't retry processing errors
        return {
            "status": "FAILED",
            "table_id": table_id,
            "task_id": self.request.id,
            "error": str(e),
        }


def process_unit_standardization_data(
    data: list[dict[str, Any]],
) -> list[ProcessedExtraction]:
    """Process unit standardization data."""
    if not data:
        return []

    logger.info(f"Processing {len(data)} extractions")

    try:
        standardizer = UnitStandardizer()
        column_units: dict[str, set[str]] = {}
        unit_locations = []

        # Extract units
        for extraction in data:
            extraction_id = extraction.get("metadata", {}).get("extractionId")
            if not extraction_id:
                continue

            for label_idx, label in enumerate(extraction.get("labels", [])):
                for key, value in label.items():
                    if key.endswith("_UNIT") and value:
                        unit_locations.append(
                            {
                                "extraction_id": extraction_id,
                                "label_index": label_idx,
                                "label_key": key,
                                "original_unit": value,
                            }
                        )
                        column_units.setdefault(key, set()).add(value)

        if not column_units:
            logger.info("No units found for standardization")
            return []

        # Standardize units
        standardization_mapping = standardizer.standardize_units(column_units)

        # Apply changes and track them
        extraction_changes = defaultdict(lambda: defaultdict(set))

        for extraction in data:
            extraction_id = extraction.get("metadata", {}).get("extractionId")
            if not extraction_id:
                continue

            for loc in unit_locations:
                if loc["extraction_id"] != extraction_id:
                    continue

                original_unit = loc["original_unit"]
                standardized_unit = standardization_mapping.get(
                    loc["label_key"], {}
                ).get(original_unit, original_unit)

                if standardized_unit != original_unit:
                    # Update extraction
                    extraction["labels"][loc["label_index"]][
                        loc["label_key"]
                    ] = standardized_unit
                    extraction_changes[extraction_id][loc["label_key"]].add(
                        (original_unit, standardized_unit)
                    )

        # Build response
        result = []
        for extraction in data:
            extraction_id = extraction.get("metadata", {}).get("extractionId")
            if extraction_id in extraction_changes:
                metadata = ExtractionMetadata(**extraction["metadata"])
                changed_parameter = {
                    param: [
                        UnitChange(prev=prev, current=curr) for prev, curr in changes
                    ]
                    for param, changes in extraction_changes[extraction_id].items()
                }
                result.append(
                    ProcessedExtraction(
                        metadata=metadata, changed_parameter=changed_parameter
                    )
                )

        logger.info(f"Processed {len(result)} extractions with changes")
        return result

    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        raise ProcessingError(f"Unit standardization failed: {str(e)}")
