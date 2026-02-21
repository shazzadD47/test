import logging
from uuid import uuid4

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sanitize_flag_id(flag_id: str, return_supplimentary_info: bool = False):
    parts = flag_id.split("-")
    is_supplimentary = parts[-2] == "supplementary"

    if not is_supplimentary:
        if return_supplimentary_info:
            return flag_id, None

        return flag_id
    else:
        supplimentary_id = parts[-1]
        flag_id = "-".join(parts[:-2])

        if return_supplimentary_info:
            return flag_id, supplimentary_id

        return flag_id


def check_if_null(value):
    try:
        # Handle pandas Series/DataFrame separately - they require element-wise checks
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return False

        # Handle numpy arrays - only check if empty (avoid pd.isna returning array)
        if isinstance(value, np.ndarray):
            return value.size == 0

        # Check for empty containers before scalar checks
        if isinstance(value, list):
            return len(value) == 0
        if isinstance(value, dict):
            return len(value) == 0

        # Use pd.isna() for scalar null/NaN detection
        # Handles: None, np.nan, pd.NA, pd.NaT, float('nan'), np.float32(nan), etc.
        try:
            if pd.isna(value):
                return True
        except (ValueError, TypeError):
            # pd.isna() can fail on some custom objects; fall through to other checks
            if value is None:
                return True

        # Check for null-like strings
        if isinstance(value, str) and value.strip().lower() in [
            "not applicable",
            "null",
            "na",
            "nan",
            "n/a",
            "none",
            "",
        ]:
            return True

        return False
    except Exception as e:
        logger.error(f"Error in check_if_null: {e} for value: {value}")
        return False


def fix_single_response(
    response: dict,
    schema: dict,
) -> dict:
    for field, value in response.items():
        if field in schema:
            if schema[field] == "string":
                if not isinstance(value, str):
                    response[field] = "N/A"
            elif schema[field] in ["number", "integer", "float"]:
                if not isinstance(value, (float, int)):
                    response[field] = None
            elif schema[field] == "boolean":
                if not isinstance(value, bool):
                    response[field] = False
            elif schema[field] == "list" and not isinstance(value, list):
                response[field] = []
            elif schema[field] == "dict" and not isinstance(value, dict):
                response[field] = {}
    for field in schema:
        if field not in response:
            if schema[field] == "string":
                response[field] = "N/A"
            elif schema[field] in ["number", "integer", "float"]:
                response[field] = None
            elif schema[field] == "boolean":
                response[field] = False
            elif schema[field] == "list":
                response[field] = []
            elif schema[field] == "dict":
                response[field] = {}

    return response


def fix_response(response: dict | list[dict], schema: dict) -> dict | list[dict]:
    """
    It creates a new response with the correct data type
    for each field. Fills the fields with None/N/A if the
    data type is incorrect. Fills all missing fields with
    the same.

    Args:
        response: dict | list[dict]
        schema: dict of field names and their data types
        the data types can be: string, number, integer, float,
        boolean, list, dict
    """
    if schema is None:
        return response
    if isinstance(response, list):
        fixed_response = []
        for item in response:
            fixed_response.append(fix_single_response(item, schema))
        return fixed_response

    elif isinstance(response, dict):
        return fix_single_response(response, schema)

    return response


def get_datatype_from_schema(schema: dict) -> dict:
    """
    It returns the data type of each field in the schema.
    The pyndatic schema of any pydantic model can be derived
    via PydanticsObject.schema().
    Pass this schema to this function to get the
    data type of each field in a dictionary.
    """
    if "properties" in schema:
        schema = schema["properties"]
    else:
        return {}
    datatype_schema = {}
    for field, value in schema.items():
        if "type" in value:
            datatype_schema[field] = value["type"]
        elif "anyOf" in value:
            datatype_schema[field] = value["anyOf"][0]["type"]
    return datatype_schema


def generate_unique_id(
    given_id: str = "",
):
    if given_id != "":
        return f"{given_id}_{uuid4().hex}"
    return f"{uuid4().hex}_{uuid4().hex}"
