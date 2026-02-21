from datetime import datetime

import httpx
from sqlalchemy import select

from app.configs import settings
from app.core.database.crud import select_with_retry
from app.core.database.models import FileDetails
from app.logging import logger


def insert_cost_document(user_id: str, embedding_token: float, embedding_cost: float):
    """Insert a cost document into the cost usage collection.

    Args:
        user_id (str): The user ID.
        embedding_token (float): The embedding token count.
        embedding_cost (float): The embedding cost.

    Raises:
        Exception: For any exceptions that occur.
    """
    try:
        request_url = f"{settings.BACKEND_BASE_URL}/v1/usage"

        httpx.post(
            request_url,
            json={
                "userId": user_id,
                "input_token": 0,
                "total_generated_token": embedding_token,
                "input_cost": 0,
                "output_cost": embedding_cost,
                "provider": "openAI",
                "endPoint": "v3/get_title_summery/embedding",
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat(),
            },
        )
    except Exception as e:
        logger.exception(f"An error occurred while inserting cost document: {e}")


def flag_id_exists(flag_id: str) -> bool:
    """Check if a flag ID exists in the literature collection.

    Args:
        flag_id (str): The flag ID to check.

    Returns:
        bool: True if the flag ID exists, False otherwise.
    """
    query = select(FileDetails).where(FileDetails.flag_id == flag_id)

    response = select_with_retry(query)

    if not response:
        return False

    if isinstance(response, list):
        return len(response) > 0
