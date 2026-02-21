import json
import logging
import os
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd
from ExtractTable import ExtractTable
from ExtractTable.exceptions import ServiceError
from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from PIL import Image, UnidentifiedImageError
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from app.configs import settings
from app.core.usage.calculate_cost import calculate_total_cost
from app.core.utils.decorators.helpers import (
    combine_cost_metadatas_of_models,
)
from app.core.vector_store import VectorStore
from app.utils.llms import (
    invoke_chain_with_retry,
    invoke_llm_with_retry,
)
from app.utils.texts import combine_langchain_contexts
from app.utils.tracing import setup_langfuse_handler
from app.utils.utils import check_if_null
from app.v3.endpoints.covariate_extraction.constants import (
    ATTRIBUTE_TYPES,
    COV_VALUE_MAPPING,
    GPT_MINI_MODEL,
    GROUP_COLUMN,
    INCORRECT_VALUE,
    NUMBER_COLUMNS,
    STRING_COLUMNS,
)
from app.v3.endpoints.covariate_extraction.logging import celery_logger as logger
from app.v3.endpoints.covariate_extraction.prompts import (
    IMAGE_TYPE_PROMPT,
    PAPER_CONTEXT_RAG_PROMPT,
)
from app.v3.endpoints.covariate_extraction.schemas import ImageType

os.makedirs(settings.SAVE_DIR, exist_ok=True)

llm_gpt_mini = ChatOpenAI(
    model=GPT_MINI_MODEL,
    temperature=0.2,
    max_tokens=16384,
)

tenacity_kwargs = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(
        multiplier=0.75,
        min=2,
        max=10,
    ),
    "reraise": True,
    "before_sleep": before_sleep_log(logger, logging.WARNING),
}


@retry(**tenacity_kwargs)
def get_image_from_url(url: str, return_media_type: bool = False):
    """
    Asynchronously fetch an image from a URL.

    Args:
        url (str): The URL of the image to fetch.
        return_media_type (bool): Whether to return the media type along with the image.

    Returns:
        bytes: The image content as bytes. If return_media_type is True,
          returns a tuple of (image, media_type).

    Raises:
        HTTPException: If the image cannot be fetched due to
        connection issues or other errors.
    """
    try:
        with httpx.Client() as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.exception(f"HTTP error occurred: {e}")
        raise HTTPException(
            status_code=response.status_code,
            detail="Failed to fetch image due to an HTTP error.",
        )
    except httpx.RequestError:
        logger.exception("Network error occurred.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to fetch image due to a network error.",
        )

    try:
        image = response.content
        image = Image.open(BytesIO(image))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The URL does not point to a valid image.",
        )
    except Exception:
        logger.exception("Failed to open downloaded image.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to open image.",
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    image_save_path = f"{uuid4().hex}.png"
    image.save(image_save_path, format="PNG")

    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")

    image_bytes = image_bytes.getvalue()
    media_type = "image/png"

    if return_media_type:
        return image_bytes, media_type, image_save_path

    return image_bytes, image_save_path


@observe()
def extract_image_type(
    image: str,
    media_type: str,
    langfuse_session_id: str = None,
):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)

    parser = PydanticOutputParser(pydantic_object=ImageType)
    format_instructions = parser.get_format_instructions()
    prompt = IMAGE_TYPE_PROMPT.format(format=format_instructions)
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    )

    structured_llm = llm_gpt_mini.with_structured_output(schema=ImageType)

    image_type = invoke_llm_with_retry(
        llm=structured_llm,
        messages=[message],
        config={
            "callbacks": [langfuse_handler],
        },
    )
    image_type = json.loads(image_type.json())["image_type"]

    return image_type


def client():
    """
    Initialize and return an instance of ExtractTable with the provided API key.

    Returns:
        ExtractTable: An instance of the ExtractTable class.
    """
    return ExtractTable(settings.CSV_API_KEY)


@retry(**tenacity_kwargs)
def extract_data_from_table_image(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Process an uploaded image file to extract tables as CSV format using
    the ExtractTable API.

    Args:
        file_path: str | Path: image file path to be processed

    Returns:
        dict: A dictionary containing the URL of the extracted CSV file
        and usage information.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        results = client().process_file(filepath=file_path, output_format="csv")
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service error: {e}",
        )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No table found in the uploaded image.",
        )

    csv_file_path = results[0]
    csv_data = pd.read_csv(csv_file_path)
    if os.path.exists(file_path):
        os.remove(file_path)
    return csv_data


@observe()
def extract_contexts_from_paper(
    paper_id: str,
    project_id: str,
    langfuse_session_id: str = None,
) -> RunnableSerializable:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_client = get_client()
    langfuse_client.update_current_trace(session_id=langfuse_session_id)
    langfuse_handler = CallbackHandler()
    retriever = VectorStore.get_retriever(
        search_kwargs={
            "k": 20,
            "filter": {
                "flag_id": paper_id.strip(),
                "project_id": project_id.strip(),
            },
        }
    )

    chain = retriever | combine_langchain_contexts

    contexts = invoke_chain_with_retry(
        chain=chain,
        input=PAPER_CONTEXT_RAG_PROMPT,
        config={
            "callbacks": [langfuse_handler],
        },
    )

    return contexts


def identify_trial_arm_column(df_columns: list[str]):
    columns = {}
    for column in df_columns:
        if isinstance(column, str):
            columns[column.lower().strip()] = column
        else:
            columns[column] = column

    for column in columns:
        if ("trial" in column or "trt" in column) and "arm" in column:
            return columns[column]
    return None


def extract_covariates_from_definition(covariates: list[dict], trial_arm: str) -> list:
    extracted_covariates = []
    is_attribute = False
    for covariate in covariates:
        if covariate["name"] == "COV":
            return ["COV"]

        if covariate["name"] == trial_arm:
            continue

        covariate_lower = covariate["name"].lower().strip()

        for attribute in ATTRIBUTE_TYPES:
            if covariate_lower.endswith(attribute):
                is_attribute = True

        if is_attribute:
            is_attribute = False
            continue

        extracted_covariates.append(covariate["name"])

    return extracted_covariates


def correct_value_type(value, type: str):
    if pd.isnull(value) or pd.isna(value):
        return None
    elif isinstance(value, (int, float)) and type == "str":
        return str(value)
    elif isinstance(value, str) and type == "num":
        try:
            return float(value)
        except Exception:
            return INCORRECT_VALUE
    return value


def remove_non_json_values(value, type: str):
    if type == "str":
        if check_if_null(value):
            return "N/A"
    elif type == "num":
        if check_if_null(value):
            return None
        try:
            if np.isinf(value) or np.isnan(value):
                return None
            return value
        except Exception:
            return value
    return value


def preprocess_covariate_value(
    covariate_value: str | float | int,
    covariate_type: str,
) -> str | float | int:
    covariate_value = remove_non_json_values(covariate_value, covariate_type)
    covariate_value = correct_value_type(covariate_value, covariate_type)
    if (
        covariate_value
        and isinstance(covariate_value, str)
        and covariate_type == "str"
        and covariate_value.strip().lower() in COV_VALUE_MAPPING
    ):
        return COV_VALUE_MAPPING[covariate_value.strip().lower()]
    return covariate_value


def clean_columns(
    table,
    columns_to_normalize,
):
    """
    Normalize the specified columns by stripping spaces
    and converting to lowercase, handling non-string values
    """
    for column in columns_to_normalize:
        table[column] = table[column].apply(
            lambda x: str(x).strip().lower() if isinstance(x, str) else x
        )

    return table


def clean_covariate_table(
    covariate_table: pd.DataFrame,
    normalize: bool = True,
    normalize_columns: list[str] = None,
) -> pd.DataFrame:
    """
    Clean the extracted covariates table by removing NaNs,
    stripping spaces, lowercasing, and normalizing selected columns.
    Handles missing fields gracefully and SKIPS unknown columns.
    """
    # If input is list of dicts, convert to DataFrame FIRST!
    if isinstance(covariate_table, list):
        keys = list(covariate_table[0].keys())
        oriented_table = {key: [] for key in keys}
        for row in covariate_table:
            for key in keys:
                oriented_table[key].append(row[key])
        covariate_table = pd.DataFrame(oriented_table)

    if normalize_columns is None:
        normalize_columns = ["COV", GROUP_COLUMN] + [
            col for col in covariate_table.columns if col.endswith("_UNIT")
        ]

    covariate_table = covariate_table.replace([np.inf, -np.inf], None)
    covariate_table = covariate_table.replace(np.nan, None)
    covariate_table = covariate_table.replace("null", None)

    trial_arm = identify_trial_arm_column(covariate_table.columns)
    if trial_arm is None:
        logger.warning(
            "Trial arm column could not be identified. Using default 'Trial_ARM'."
        )
        trial_arm = "Trial_ARM"

    string_columns = STRING_COLUMNS + [trial_arm, "COV", GROUP_COLUMN]
    processed_flag = False

    for column in covariate_table.columns:
        for num_attribute in NUMBER_COLUMNS:
            if column.lower().strip().endswith(num_attribute):
                covariate_table[column] = covariate_table[column].apply(
                    lambda x: preprocess_covariate_value(x, "num")
                )
                processed_flag = True
                break
        if processed_flag:
            processed_flag = False
            continue

        for str_attribute in string_columns:
            if (
                column.lower().strip().endswith(str_attribute)
                or column == str_attribute
            ):
                covariate_table[column] = covariate_table[column].apply(
                    lambda x: preprocess_covariate_value(x, "str")
                )
                processed_flag = True
                break
        if processed_flag:
            processed_flag = False
            continue

        covariate_table[column] = covariate_table[column].apply(
            lambda x: preprocess_covariate_value(x, "num")
        )

    if normalize:
        covariate_table = clean_columns(
            covariate_table,
            [trial_arm] + normalize_columns,
        )
    return covariate_table


def check_is_attribute(covariate: str) -> bool:
    is_number = any(
        covariate.strip().lower().endswith(number_attribute)
        for number_attribute in NUMBER_COLUMNS
    )
    is_string = any(
        covariate.strip().lower().endswith(string_attribute)
        for string_attribute in STRING_COLUMNS
    )
    return is_number or is_string


def assign_to_covariates_from_definition(
    prev_covariate_table,
    covariate_mapping,
    covariate_list,
):
    """
    Convert the extracted covariates table by assigning to
    the predefined covariates in the provided table definition
    via the covariate mapping.
    """
    trial_arm = identify_trial_arm_column(prev_covariate_table.columns)
    standardized_df = {covariate: [] for covariate in covariate_list}
    covariate_list_arm = identify_trial_arm_column(covariate_list)
    standardized_df_trial_arm = covariate_list_arm or trial_arm

    standardized_df[standardized_df_trial_arm] = []
    standardized_df[GROUP_COLUMN] = []

    arms_list = prev_covariate_table[trial_arm].unique()
    arm_groups_info = {}

    for arm in arms_list:
        arm_groups = list(
            prev_covariate_table.loc[
                prev_covariate_table[trial_arm] == arm, GROUP_COLUMN
            ].unique()
        )
        arm_groups = [group for group in arm_groups if not check_if_null(group)]

        if not arm_groups:
            prev_covariate_table.loc[
                prev_covariate_table[trial_arm] == arm, GROUP_COLUMN
            ] = "N/A"
            standardized_df[GROUP_COLUMN].append("N/A")
            standardized_df[standardized_df_trial_arm].append(arm)
            arm_groups_info[arm] = ["N/A"]
        else:
            arm_groups_info[arm] = arm_groups
            for group in arm_groups:
                group_name_str = str(group).lower().strip()
                arm_name_str = str(arm).lower().strip()
                combined_name = (
                    f"{arm_name_str}_{group_name_str}"
                    if group_name_str not in arm_name_str
                    else arm
                )
                standardized_df[standardized_df_trial_arm].append(combined_name)
            standardized_df[GROUP_COLUMN].extend(arm_groups)

    for covariate in covariate_list:
        if covariate.lower().strip() == "trial_arm" or check_is_attribute(covariate):
            continue

        for arm in arms_list:
            for group in arm_groups_info[arm]:
                if covariate not in covariate_mapping or check_if_null(
                    covariate_mapping[covariate]
                ):
                    standardized_df[covariate].append(None)
                    for attribute in NUMBER_COLUMNS:
                        attr = f"{covariate}_{attribute.upper()}"
                        if attr in covariate_list:
                            standardized_df[attr].append(None)
                    for attribute in STRING_COLUMNS:
                        attr = f"{covariate}_{attribute.upper()}"
                        if attr in covariate_list:
                            standardized_df[attr].append("N/A")
                    continue

                # Resolve mapped covariate name
                prev_covariate_name = covariate_mapping[covariate]
                for prev_cov in prev_covariate_table["COV"].unique():
                    if prev_covariate_name.lower().strip() == prev_cov.lower().strip():
                        prev_covariate_name = prev_cov
                        break

                mask = (
                    (prev_covariate_table[trial_arm] == arm)
                    & (prev_covariate_table[GROUP_COLUMN] == group)
                    & (prev_covariate_table["COV"] == prev_covariate_name)
                )

                cov_val = prev_covariate_table.loc[mask, "COV_VAL"].values
                standardized_df[covariate].append(
                    cov_val[0] if len(cov_val) > 0 else None
                )

                for attribute in NUMBER_COLUMNS + STRING_COLUMNS:
                    attr = f"{covariate}_{attribute.upper()}"
                    col_name = f"COV_{attribute.upper()}"
                    unit_col_name = f"COV_{attribute.upper()}_UNIT"
                    unit_attr = f"{attr}_UNIT"

                    if attr in covariate_list:
                        val = prev_covariate_table.loc[mask, col_name].values
                        fallback = (
                            None if attribute.upper() in NUMBER_COLUMNS else "N/A"
                        )
                        standardized_df[attr].append(
                            val[0] if len(val) > 0 else fallback
                        )

                        if unit_attr in covariate_list:
                            unit_val = prev_covariate_table.loc[
                                mask, unit_col_name
                            ].values
                            standardized_df[unit_attr].append(
                                unit_val[0] if len(unit_val) > 0 else "N/A"
                            )

    # Final safety: align all column lengths
    max_len = max(len(v) for v in standardized_df.values())
    for key in standardized_df:
        while len(standardized_df[key]) < max_len:
            fallback = None if key not in STRING_COLUMNS else "N/A"
            standardized_df[key].append(fallback)

    return pd.DataFrame(standardized_df)


def merge_covariate_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    combined_table = tables[0].copy()
    for table in tables[1:]:
        combined_table = combined_table.combine_first(table)
    return combined_table


def convert_optimized_to_original_format(optimized_json_data):
    """
    Converts the optimized JSON format with parallel arrays back to the original
    expanded format with individual covariate records.

    Args:
        optimized_json_data (dict): The optimized JSON data with parallel arrays

    Returns:
        dict: The original expanded format JSON
    """
    original_format = {"data": []}

    # Iterate through each trial arm in the optimized data
    for arm_data in optimized_json_data["data"]:
        trial_arm = arm_data["Trial_ARM"]
        group_name = arm_data["group_name"]

        # Get all the covariate arrays
        covs = arm_data["COV"]
        cov_vals = arm_data["COV_VAL"]
        cov_units = arm_data["COV_UNIT"]
        cov_stats = arm_data["COV_STAT"]
        cov_mins = arm_data["COV_MIN"]
        cov_min_units = arm_data["COV_MIN_UNIT"]
        cov_maxs = arm_data["COV_MAX"]
        cov_max_units = arm_data["COV_MAX_UNIT"]
        cov_vars = arm_data["COV_VAR"]
        cov_var_units = arm_data["COV_VAR_UNIT"]
        cov_var_stats = arm_data["COV_VAR_STAT"]

        # For each covariate in this arm, create an individual record
        for i, cov in enumerate(covs):
            cov_record = {
                "Trial_ARM": trial_arm,
                "group_name": group_name,
                "COV": cov,
                "COV_VAL": cov_vals[i] if i < len(cov_vals) else None,
                "COV_UNIT": cov_units[i] if i < len(cov_units) else None,
                "COV_STAT": cov_stats[i] if i < len(cov_stats) else None,
                "COV_MIN": cov_mins[i] if i < len(cov_mins) else None,
                "COV_MIN_UNIT": cov_min_units[i] if i < len(cov_min_units) else None,
                "COV_MAX": cov_maxs[i] if i < len(cov_maxs) else None,
                "COV_MAX_UNIT": cov_max_units[i] if i < len(cov_max_units) else None,
                "COV_VAR": cov_vars[i] if i < len(cov_vars) else None,
                "COV_VAR_UNIT": cov_var_units[i] if i < len(cov_var_units) else None,
                "COV_VAR_STAT": cov_var_stats[i] if i < len(cov_var_stats) else None,
            }

            # Add this record to the original format
            original_format["data"].append(cov_record)

    return original_format


def agrregate_cost_metadatas(covariate_tables: list[dict]) -> dict:
    all_cost_metadatas = []
    for table in covariate_tables:
        if (
            isinstance(table, dict)
            and "metadata" in table
            and isinstance(table["metadata"], dict)
            and "ai_metadata" in table["metadata"]
            and isinstance(table["metadata"]["ai_metadata"], dict)
            and "cost_metadata" in table["metadata"]["ai_metadata"]
            and isinstance(table["metadata"]["ai_metadata"]["cost_metadata"], dict)
            and len(table["metadata"]["ai_metadata"]["cost_metadata"]) > 0
            and "llm_cost_details" in table["metadata"]["ai_metadata"]["cost_metadata"]
            and (
                isinstance(
                    table["metadata"]["ai_metadata"]["cost_metadata"][
                        "llm_cost_details"
                    ],
                    dict,
                )
            )
            and (
                len(
                    table["metadata"]["ai_metadata"]["cost_metadata"][
                        "llm_cost_details"
                    ]
                )
                > 0
            )
        ):
            all_cost_metadatas.append(
                table.get("metadata", {})
                .get("ai_metadata", {})
                .get("cost_metadata", {})
                .get("llm_cost_details", {})
            )

    summarization_cost = {}
    for table in covariate_tables:
        if (
            isinstance(table, dict)
            and "metadata" in table
            and isinstance(table["metadata"], dict)
            and "ai_metadata" in table["metadata"]
            and isinstance(table["metadata"]["ai_metadata"], dict)
            and "summarization_cost_metadata" in table["metadata"]["ai_metadata"]
            and isinstance(
                table["metadata"]["ai_metadata"]["summarization_cost_metadata"], dict
            )
            and (
                "llm_cost_details"
                in table["metadata"]["ai_metadata"]["summarization_cost_metadata"]
            )
            and (
                isinstance(
                    table["metadata"]["ai_metadata"]["summarization_cost_metadata"][
                        "llm_cost_details"
                    ],
                    dict,
                )
            )
            and (
                len(
                    table["metadata"]["ai_metadata"]["summarization_cost_metadata"][
                        "llm_cost_details"
                    ]
                )
                > 0
            )
        ):
            summarization_cost = (
                table.get("metadata", {})
                .get("ai_metadata", {})
                .get("summarization_cost_metadata", {})
                .get("llm_cost_details", {})
            )
            break
    if len(summarization_cost) > 0:
        all_cost_metadatas.append(summarization_cost)

    if len(all_cost_metadatas) == 0:
        return {}
    else:
        cost_metadatas = combine_cost_metadatas_of_models(all_cost_metadatas)
        total_cost = calculate_total_cost(cost_metadatas)
        cost_metadatas = {k: v.model_dump() for k, v in cost_metadatas.items()}
        return {
            "total_cost": total_cost,
            "llm_cost_details": cost_metadatas,
        }


def find_suggested_loc_from_query(
    query: str,
):
    rephrased_question, suggested_location = query.split(
        "Suggestion regarding where to find:"
    )
    return rephrased_question, suggested_location
