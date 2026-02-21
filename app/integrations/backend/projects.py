import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from app.configs import settings
from app.logging import logger


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.75, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def get_project_details(project_id: str, user_token: str) -> dict:
    endpoint = settings.BACKEND_PROJECT_DETAILS_ENDPOINT.format(project_id=project_id)
    url = f"{settings.BACKEND_BASE_URL}/{endpoint}"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            url, headers={"Authorization": f"Bearer {user_token}"}
        )
        response.raise_for_status()

        if not response.content:
            logger.error(
                f"Empty response from project details API for project {project_id}"
            )
            raise ValueError("Empty response from project details API")

        return response.json()
