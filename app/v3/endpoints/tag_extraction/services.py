from uuid import uuid4

from celery import group

from app.logging import logger
from app.v3.endpoints.tag_extraction.exceptions import TaskSubmissionException
from app.v3.endpoints.tag_extraction.schemas import TagExtractionRequest
from app.v3.endpoints.tag_extraction.tasks import tag_extraction_task


def start_tag_extraction(data: TagExtractionRequest) -> dict:
    """
    Kick off tag extraction tasks for each flag_id.
    Returns a metadata envelope with task information.
    """
    logger.info(
        f"Starting tag extraction for {len(data.flag_ids)} flag_ids "
        f"with {len(data.tags)} tags"
    )

    if not data.flag_ids:
        logger.error("Received empty flag_ids list")
        raise ValueError("flag_ids list cannot be empty")

    langfuse_session_id = uuid4().hex
    logger.debug(f"Generated langfuse session ID: {langfuse_session_id}")

    try:
        jobs = []
        for fid in data.flag_ids:
            jobs.append(
                tag_extraction_task.s(
                    flag_id=fid,
                    project_id=data.meta_data.get("project_id", ""),
                    tag_definitions=[t.model_dump() for t in data.tags],
                    meta_data=data.meta_data,
                    langfuse_session_id=langfuse_session_id,
                )
            )

        logger.info(f"Submitting {len(jobs)} tag extraction tasks to Celery")
        workflow = group(jobs).apply_async()
        task_id = workflow.id

        logger.info(
            f"Successfully started tag extraction workflow with task_id: " f"{task_id}"
        )

        metadata = data.meta_data.copy()
        metadata["message"] = "Tag extraction started in background"
        metadata["ai_metadata"] = {"task_id": task_id}
        return metadata

    except Exception as e:
        logger.exception(f"Failed to submit tag extraction tasks: {e}")
        raise TaskSubmissionException(f"Could not start tag extraction: {str(e)}")
