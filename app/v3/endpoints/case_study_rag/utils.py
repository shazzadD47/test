import httpx
import pandas as pd
import requests

from app.configs import settings
from app.integrations.backend import generate_api_token
from app.v3.endpoints.case_study_rag.logging import logger


def process_items(items):
    context = ""
    for item in items:
        item_type = item.get("type")
        if item_type == "table":
            context += f"Type: {item_type}\n"
            csv_link = item.get(item_type)
            if csv_link:
                try:
                    response = requests.get(csv_link, timeout=30)
                    response.raise_for_status()
                    context += "CSV Content:\n"
                    context += response.text
                    context += "\n\n"
                except requests.exceptions.RequestException as e:
                    context += f"Failed to fetch CSV content: {e}\n"
        else:
            context += f"Type: {item_type}\n"
            content = item.get(item_type)
            context += f"Content: {content}\n\n"
    return context


def get_line_items(extraction_id: str):
    token = generate_api_token(settings.BACKEND_SECRET)
    headers = {"x-api-key": f"{settings.BACKEND_KEY}###{token}"}

    url = f"{settings.BACKEND_BASE_URL}/extractions/{extraction_id}"

    try:
        response = httpx.get(url, headers=headers)
        return response.json()
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return []


def get_table_structures(project_id: str):
    token = generate_api_token(settings.BACKEND_SECRET)
    headers = {"x-api-key": f"{settings.BACKEND_KEY}###{token}"}

    url = (
        f"{settings.BACKEND_BASE_URL}/extractions/project/{project_id}/table-structures"
    )

    try:
        response = httpx.get(url, headers=headers)
        return response.json()
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return []


def get_plot_context(items, project_id):
    dataframes = []
    plot_context = ""

    item: dict
    for item in items:
        item_type = item.get("type")
        if item_type == "plot":
            try:
                extraction_id = item.get("extractionId")
                line_items = get_line_items(extraction_id)

                for line in line_items:
                    x_axis_name = line.get("x_axis_name")
                    y_axis_name = line.get("y_axis_name")
                    normalized_points = line.get("storedNormalizedPoints")
                    labels = line.get("labels")
                    df = pd.DataFrame(normalized_points, columns=["x", "y"])
                    df = df.rename(columns={"x": x_axis_name, "y": y_axis_name})

                    table_labels_value_list = get_table_structures(project_id)
                    for table in table_labels_value_list:
                        table_structure = table.get("columns")
                        for values in table_structure:
                            if values["c_type"] == "general":
                                df[values["name"]] = labels.get(values["name"], "")
                                if values["hasUnit"]:
                                    df["unit"] = labels.get(
                                        f"{values['name']}_unit", ""
                                    )
                    dataframes.append(df)
                for df in dataframes:
                    plot_context += f"{df.to_dict()}"
            except Exception as e:
                logger.exception(f"Failed to extract plot data: {e}")
                return "plot data cannot be extracted"
    return plot_context
