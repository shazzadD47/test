import base64

import requests
from bs4 import BeautifulSoup
from sqlalchemy import select

from app.core.database.crud import ResultType, select_with_retry
from app.core.database.models import (
    FileDetails,
)
from app.v3.endpoints.projects.exceptions import (
    DataFetchFailed,
)
from app.v3.endpoints.projects.logging import logger


def get_paper_summaries_by_flag_id(flag_ids: list[str]) -> list[dict] | None:
    """
    Retrieves all paper_summary values for a given project_id.

    Args:
        project_id: The project ID to filter the FileDetails rows

    Returns:
        A list of paper_summary strings, or None if an error occurs
    """
    query = select(
        FileDetails.flag_id,
        FileDetails.supplementary_id,
        FileDetails.paper_summary,
        FileDetails.summary,
    ).where(FileDetails.flag_id.in_(flag_ids))

    try:
        results = select_with_retry(query, result_type=ResultType.ROW_ALL)

        return [
            {
                "flag_id": result[0],
                "supplementary_id": result[1],
                "paper_summary": result[2],
                "abstract": result[3],
            }
            for result in results
        ]
    except Exception as e:
        logger.exception(f" failed  for project_id: {flag_ids}] : {str(e)}")
        raise DataFetchFailed()


def parse_html_content(html: str):
    soup = BeautifulSoup(html, "html.parser")
    result = []
    try:
        for tag in soup.find_all(True):  # True = all tags
            if tag.name == "img":
                src = tag.get("src")
                if src:
                    result.append({"type": "image", "value": src})
            else:  # any non-image tag
                text = tag.get_text(strip=True)
                if text:
                    result.append({"type": "text", "value": text})
    except Exception as e:
        logger.exception(f"Error parsing HTML content: {str(e)}")
        return []
    return result


def encode_image_from_url(url: str) -> str | None:
    try:
        response = requests.get(url, timeout=10)  # add timeout for safety
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except requests.exceptions.RequestException as e:
        logger.exception(f"Error fetching or encoding image from URL {url}: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error encoding image from URL {url}: {str(e)}")
    return None
