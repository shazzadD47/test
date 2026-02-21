import httpx

from app.configs import settings
from app.integrations.backend import generate_api_token
from app.logging import logger


def get_table_contents(paper_id: str, project_id: str = None) -> list[str]:
    request_url = f"{settings.BACKEND_BASE_URL}/knowledge-files/tables"

    token = generate_api_token(settings.BACKEND_SECRET)
    headers = {"x-api-key": f"{settings.BACKEND_KEY}###{token}"}

    params = {"flag_id": paper_id.strip()}
    if project_id:
        params["project_id"] = project_id.strip()

    response = httpx.get(request_url, headers=headers, params=params)

    table_csv_contents = []
    for table_csv_url in response.json():
        try:
            response = httpx.get(table_csv_url)
            table_csv_contents.append(response.text)
        except Exception as e:
            logger.warning(f"Error fetching table contents from {table_csv_url}: {e}")

    return table_csv_contents


async def async_get_table_contents(paper_id: str, project_id: str = None) -> list[str]:
    return get_table_contents(paper_id, project_id)
