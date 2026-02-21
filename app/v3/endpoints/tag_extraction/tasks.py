import json
import time
from uuid import uuid4

from celery.exceptions import SoftTimeLimitExceeded
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from pydantic import BaseModel, Field, ValidationError

from app.configs import settings
from app.core.auto.chat_model import AutoChatModel
from app.core.celery.app import celery_app
from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.utils.decorators.cost_tracker import track_all_llm_costs
from app.exceptions.system import (
    AnthropicBadRequestError,
    AnthropicInternalServerError,
    OpenAIBadRequestError,
    OpenAIServerError,
)
from app.logging import logger
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints import Status
from app.v3.endpoints.covariate_extraction.helpers.utils import (
    extract_contexts_from_paper,
)
from app.v3.endpoints.tag_extraction.prompts import SYSTEM_INSTRUCTION, TAG_EVAL_PROMPT
from app.v3.endpoints.tag_extraction.schemas import (
    CostMetadata,
    TagDefinition,
    TagExtractionResponse,
)


class SingleTagResult(BaseModel):
    name: str
    reasoning: str
    is_relevant: bool
    relevance_score: int = Field(ge=0, le=100)


class TagsEvalSchema(BaseModel):
    tags: list[SingleTagResult]


def _extract_cost_metadata(result: dict) -> dict:
    """Extract cost metadata from the decorator's result structure."""
    if "metadata" in result and "ai_metadata" in result["metadata"]:
        return result["metadata"]["ai_metadata"].get("cost_metadata", {})
    return {}


@track_all_llm_costs
def execute_tag_extraction(
    flag_id: str,
    project_id: str,
    tag_definitions: list[dict],
    meta_data: dict,
    langfuse_session_id: str | None = None,
):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    _langfuse_handler = setup_langfuse_handler(
        langfuse_session_id, name="tag_extraction_task"
    )

    # Retrieve contexts by flag_id (synchronously)
    context_text = extract_contexts_from_paper(
        paper_id=flag_id,
        project_id=project_id,
        langfuse_session_id=langfuse_session_id,
    )

    # Validate that we have meaningful context to work with
    if not context_text or not context_text.strip():
        logger.warning(
            f"No context found for flag_id={flag_id}, skipping tag extraction"
        )
        raise ValueError(
            f"No context available for flag_id={flag_id}. "
            "Cannot perform tag extraction without paper content."
        )

    model = AutoChatModel.from_model_name(
        model_name=settings.GPT_4_TEXT_MODEL,
        temperature=settings.TAG_EXTRACTION_TEMPARAUTURE,
    )

    schema = TagsEvalSchema
    structured_llm = model.with_structured_output(schema=schema)

    tags_payload = [TagDefinition(**t).model_dump() for t in tag_definitions]

    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(
            content=[
                {"type": "text", "text": TAG_EVAL_PROMPT},
                {"type": "text", "text": f"CONTEXTS:\n{context_text}"},
                {
                    "type": "text",
                    "text": (f"TAGS:\n{json.dumps(tags_payload, ensure_ascii=False)}"),
                },
            ]
        ),
    ]

    result = structured_llm.invoke(messages, config={"callbacks": [_langfuse_handler]})

    response: TagExtractionResponse = TagExtractionResponse(
        flag_id=flag_id,
        tags=[tag.model_dump() for tag in result.tags],
        cost_metadata=None,
        meta_data=meta_data,
    )

    final_payload = {
        "payload": response.model_dump(),
        "metadata": {
            "status": Status.SUCCESS.value,
            "ai_metadata": {},
        },
    }
    return final_payload


@celery_app.task(
    name="tag_extraction_task",
    bind=True,
    max_retries=3,
    autoretry_for=(
        # Network and transient errors - should retry
        OpenAIServerError,
        AnthropicInternalServerError,
        ConnectionError,
        TimeoutError,
    ),
    retry_kwargs={"max_retries": 3, "countdown": 60},
    time_limit=300,  # Hard timeout after 5 minutes
    soft_time_limit=240,  # Soft timeout warning after 4 minutes
)
@observe()
def tag_extraction_task(
    self,
    flag_id: str,
    project_id: str,
    tag_definitions: list[dict],
    meta_data: dict,
    langfuse_session_id: str | None = None,
):
    task_id = self.request.id
    start_time = time.time()
    logger.info(
        f"Starting tag extraction task {task_id} for flag_id={flag_id} "
        f"with {len(tag_definitions)} tags"
    )

    try:
        result_with_cost = execute_tag_extraction(
            flag_id=flag_id,
            project_id=project_id,
            tag_definitions=tag_definitions,
            meta_data=meta_data,
            langfuse_session_id=langfuse_session_id,
        )

        # Extract the original result and cost metadata
        if "result" in result_with_cost:
            # Cost decorator wrapped the result
            final_payload = result_with_cost["result"]
            cost_metadata = _extract_cost_metadata(result_with_cost)
        else:
            # No cost tracking or no cost data
            final_payload = result_with_cost
            cost_metadata = {}

        # Attach cost metadata to the response if available
        if (
            cost_metadata
            and "payload" in final_payload
            and isinstance(final_payload["payload"], dict)
        ):
            final_payload["payload"]["cost_metadata"] = CostMetadata(
                total_cost=cost_metadata.get("total_cost", 0.0),
                llm_cost_details=cost_metadata.get("llm_cost_details", {}),
            ).model_dump()

        # Calculate metrics
        processing_time = time.time() - start_time
        num_tags = len(final_payload.get("payload", {}).get("tags", []))
        total_cost = cost_metadata.get("total_cost", 0.0) if cost_metadata else 0.0

        logger.info(
            f"Successfully completed tag extraction task {task_id} for "
            f"flag_id={flag_id} | Processing time: {processing_time:.2f}s | "
            f"Tags evaluated: {len(tag_definitions)} | "
            f"Tags matched: {num_tags} | Total cost: ${total_cost:.4f}"
        )

        # Try to send to backend, but don't fail the task if this fails
        try:
            send_to_backend(BackendEventEnumType.TAG_EXTRACTION, final_payload)
        except Exception as backend_error:
            logger.error(
                f"Failed to send results to backend for task {task_id}: "
                f"{backend_error}"
            )
            # Continue and return the payload anyway - the extraction itself succeeded

        return final_payload

    except (OpenAIBadRequestError, AnthropicBadRequestError) as e:
        # API bad request errors - don't retry, fail immediately
        logger.error(
            f"Bad request for tag extraction task {task_id}, " f"flag_id={flag_id}: {e}"
        )
        error_payload = {
            "payload": {
                "flag_id": flag_id,
                "tags": [],
                "cost_metadata": None,
                "meta_data": meta_data,
            },
            "metadata": {
                "status": Status.FAILED.value,
                "message": f"Invalid API request: {str(e)}",
                "task_id": task_id,
                "error_type": "bad_request",
            },
        }
        send_to_backend(BackendEventEnumType.TAG_EXTRACTION, error_payload)
        return error_payload

    except ValidationError as e:
        # Pydantic validation errors - don't retry, fail immediately
        logger.error(
            f"Validation error for tag extraction task {task_id}, "
            f"flag_id={flag_id}: {e}"
        )
        error_payload = {
            "payload": {
                "flag_id": flag_id,
                "tags": [],
                "cost_metadata": None,
                "meta_data": meta_data,
            },
            "metadata": {
                "status": Status.FAILED.value,
                "message": f"Validation error: {str(e)}",
                "task_id": task_id,
                "error_type": "validation_error",
            },
        }
        send_to_backend(BackendEventEnumType.TAG_EXTRACTION, error_payload)
        return error_payload

    except ValueError as e:
        # Input validation errors - don't retry, fail immediately
        logger.error(
            f"Invalid input for tag extraction task {task_id}, "
            f"flag_id={flag_id}: {e}"
        )
        error_payload = {
            "payload": {
                "flag_id": flag_id,
                "tags": [],
                "cost_metadata": None,
                "meta_data": meta_data,
            },
            "metadata": {
                "status": Status.FAILED.value,
                "message": f"Invalid input: {str(e)}",
                "task_id": task_id,
                "error_type": "validation_error",
            },
        }
        send_to_backend(BackendEventEnumType.TAG_EXTRACTION, error_payload)
        return error_payload

    except OutputParserException as e:
        # LLM output parsing errors - might be worth retrying once
        retry_count = self.request.retries
        logger.warning(
            f"Output parsing failed for flag_id={flag_id} "
            f"(attempt {retry_count + 1}): {e}"
        )

        if retry_count >= 1:  # Only retry once for parsing errors
            logger.error(f"Output parsing failed permanently for task {task_id}")
            error_payload = {
                "payload": {
                    "flag_id": flag_id,
                    "tags": [],
                    "cost_metadata": None,
                    "meta_data": meta_data,
                },
                "metadata": {
                    "status": Status.FAILED.value,
                    "message": f"Failed to parse LLM output: {str(e)}",
                    "task_id": task_id,
                    "error_type": "parsing_error",
                },
            }
            send_to_backend(BackendEventEnumType.TAG_EXTRACTION, error_payload)
            return error_payload
        else:
            raise self.retry(exc=e, countdown=30)

    except SoftTimeLimitExceeded:
        # Soft timeout reached - task is taking too long
        processing_time = time.time() - start_time
        logger.error(
            f"Task {task_id} exceeded soft time limit after " f"{processing_time:.2f}s"
        )
        error_payload = {
            "payload": {
                "flag_id": flag_id,
                "tags": [],
                "cost_metadata": None,
                "meta_data": meta_data,
            },
            "metadata": {
                "status": Status.FAILED.value,
                "message": (
                    f"Task timeout: processing exceeded "
                    f"{processing_time:.0f}s limit"
                ),
                "task_id": task_id,
                "error_type": "timeout",
            },
        }
        send_to_backend(BackendEventEnumType.TAG_EXTRACTION, error_payload)
        raise  # Re-raise to mark task as failed

    except (
        OpenAIServerError,
        AnthropicInternalServerError,
        ConnectionError,
        TimeoutError,
    ) as e:
        # Network/transient errors - retry with backoff (handled by autoretry_for)
        retry_count = self.request.retries
        max_retries = self.max_retries
        logger.warning(
            f"Transient error for flag_id={flag_id} "
            f"(attempt {retry_count + 1}/{max_retries + 1}): {e}"
        )

        if retry_count >= max_retries:
            logger.error(f"Transient errors exhausted retries for task {task_id}")
            error_payload = {
                "payload": {
                    "flag_id": flag_id,
                    "tags": [],
                    "cost_metadata": None,
                    "meta_data": meta_data,
                },
                "metadata": {
                    "status": Status.FAILED.value,
                    "message": (
                        f"Service temporarily unavailable after "
                        f"{max_retries} retries: {str(e)}"
                    ),
                    "task_id": task_id,
                    "error_type": "transient_error",
                },
            }
            send_to_backend(BackendEventEnumType.TAG_EXTRACTION, error_payload)
            return error_payload
        else:
            raise  # Let autoretry_for handle it

    except Exception as e:
        # Unexpected system errors - needs immediate attention, don't retry
        logger.exception(
            f"Unexpected system error in tag extraction task {task_id}, "
            f"flag_id={flag_id}: {e}"
        )
        error_payload = {
            "payload": {
                "flag_id": flag_id,
                "tags": [],
                "cost_metadata": None,
                "meta_data": meta_data,
            },
            "metadata": {
                "status": Status.FAILED.value,
                "message": f"System error: {str(e)}",
                "task_id": task_id,
                "error_type": "system_error",
            },
        }
        send_to_backend(BackendEventEnumType.TAG_EXTRACTION, error_payload)
        # Re-raise for alerting/monitoring systems to catch
        raise
