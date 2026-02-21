import httpx

from app.v3.endpoints.get_title_summery.configs import settings as project_settings


def chart_dete_completion(
    image_path,
    endpoint: str = project_settings.CHART_DETEC_ENDPOINT,
) -> dict:
    """
    Send a request to the chart_dete completion endpoint.

    Args:
        image_path: Path to the image file
        base_url: Base URL of the API server

    Returns:
        dict: Response from the API containing completion
    """

    files = {"image": image_path}

    timeout_settings = httpx.Timeout(
        timeout=60.0,
        connect=30.0,
        read=30.0,
        write=30.0,
    )

    try:

        with httpx.Client(timeout=timeout_settings) as client:

            response = client.post(endpoint, files=files)

            response.raise_for_status()

            return response.json()

    except httpx.TimeoutException as e:
        raise Exception(f"Request timed out. Please try again later. Details: {str(e)}")
    except httpx.ReadTimeout as e:
        raise Exception(
            f"Read timeout. The server took too long to respond. Details: {str(e)}"
        )
    except httpx.HTTPError as e:
        raise Exception(f"HTTP error occurred: {str(e)}")
