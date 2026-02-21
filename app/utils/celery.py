import time

import requests

from app.logging import logger


def get_celery_task_result_with_polling(
    task_id: str,
    poll_interval: int = 5,
    timeout: int = 1000,
    endpoint: str = "http://localhost:8000/v3/tasks/{task_id}/status",
):
    max_poll_attempts = max(timeout // poll_interval, 1)
    for _ in range(max_poll_attempts):
        task_status_endpoint = endpoint.format(task_id=task_id)
        status_response = requests.get(task_status_endpoint, timeout=timeout)
        celery_result = status_response.json()
        logger.info(f"Task {task_id} status: {celery_result}")
        if celery_result["status"] == "SUCCESS":
            logger.info(f"Task {task_id} completed successfully")
            return celery_result["result"]
        elif celery_result["status"] == "PENDING":
            logger.info(f"Task {task_id} is still pending")
            time.sleep(poll_interval)
            continue
        elif celery_result["status"] == "FAILED" or (
            celery_result["result"].get("metadata", {}).get("status", "") == "failed"
        ):
            error_message = f"Task {task_id} failed"
            logger.error(error_message)
            raise Exception(error_message)
        else:
            logger.error(f"Max wait time {timeout} seconds reached.")
            logger.error(f"Task {task_id} failed: {celery_result['error']}")
            return None
