import json
import re
import unicodedata

import httpx
import numpy as np
import pandas as pd

from app.utils.utils import check_if_null
from app.v3.endpoints.merging.constants import TableNames
from app.v3.endpoints.merging.logging import logger


def replace_with_v1_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if table_name.strip().lower() == "observation":
        column_map = {
            "PUBMEDID": " PUBMEDID",
            "doi": "DOI",
            "fileName": "FILE NAME",
            "fileId": "FILEID",
            "group": "GROUP NAME",
            "paperTitle": "PAPER TITLE",
            "uci": "UCI",
            "lci": "LCI",
            "varu": "VARU",
            "var": "VAR",
            "x_val": "X_VAL",
            "x_unit": "X_UNIT",
            "y_val": "Y_VAL",
            "y_unit": "Y_UNIT",
        }
        df["EXTRACTION COMMENTS"] = df["COMMENTS"].copy()
        df["FIGURE NO"] = df["SOURCE"].copy()
        df.rename(columns=column_map, inplace=True)
        missing_columns = [
            "IMAGES",
            "LINE NAME",
            "LINE STATUS",
            "PAPER STATUS",
            "PAPER COMMENTS",
        ]
        for column in missing_columns:
            df[column] = [None] * len(df)
    elif table_name.strip().lower() == TableNames.COVARIATE.value:
        df.drop(columns=["group"], inplace=True)
        column_map = {
            "doi": "DOI",
            "PUBMEDID": "PUBMEDID",
            "fileName": "FILE NAME",
            "fileId": "FILEID",
            "GROUP": "GROUP NAME",
            "paperTitle": "PAPER TITLE",
        }
        df.rename(columns=column_map, inplace=True)
        missing_columns = [
            "EXTRACTION COMMENTS",
            "FIGURE NO",
            "IMAGES",
            "LCI",
            "LINE NAME",
            "LINE STATUS",
            "PAPER COMMENTS",
            "PAPER STATUS",
            "UCI",
            "VAR",
            "VARU",
        ]
        for column in missing_columns:
            df[column] = [None] * len(df)
    elif table_name.strip().lower() == "dosing":
        df.drop(columns=["group"], inplace=True)
        column_map = {
            "doi": "DOI",
            "fileName": "FILE NAME",
            "fileId": "FILEID",
            "paperTitle": "PAPER TITLE",
        }
        df.rename(columns=column_map, inplace=True)
        missing_columns = [
            "PAPER COMMENTS",
            "PAPER STATUS",
            "IMAGES",
        ]
        for column in missing_columns:
            df[column] = [None] * len(df)
    return df


def replace_nans_with_na(
    value: str | None,
) -> str:
    """
    Replace NaN values with None for float and int types.
    Replace NaN values with "NA" for string types.
    Otherwise, return the value as is.
    """
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return value
    if isinstance(value, (float, int)) and check_if_null(value):
        return None
    elif isinstance(value, str) and check_if_null(value):
        return "NA"
    return value


def transform_filenames(filenames: list[str]):
    updated_filenames = []
    for _index, filename in enumerate(filenames):
        # Check if there's a year after the name
        if not re.match(r"^[a-zA-Z]+\d{4}_", filename):
            updated_filenames.append(filename)
        else:
            # Capitalize the first letter
            filename = filename.capitalize()
            # Capitalize all letters after the first underscore
            filename = re.sub(r"_(.+)", lambda m: "_" + m.group(1).upper(), filename)
            updated_filenames.append(filename)

    return updated_filenames


def standardize_word(word):
    # Use NFC for canonical composition
    normalized_word = unicodedata.normalize("NFC", word)
    return normalized_word


def standardize_percentage(value: str) -> str:
    value = re.sub(r"%|\bpercent\b|\bpercents\b", "percentage", value)
    return value


def check_if_range(value: str) -> bool:
    """
    Check if a value represents a range (e.g., "10-20", "5 to 10", "from 1 to 5").

    Args:
        value: The value to check

    Returns:
        bool: True if the value is a range, False otherwise
    """
    range_patterns = [
        r"^\d*\.?\d+\s*-\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*:\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s+to\s+\d*\.?\d+$",
        r"^from\s+\d*\.?\d+\s+to\s+\d*\.?\d+$",
        r"^\d*\.?\d+\s*till\s+\d*\.?\d+$",
        r"^\d*\.?\d+\s*until\s+\d*\.?\d+$",
        r"^\d*\.?\d+\s*–\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*—\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2010\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2011\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2012\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2013\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2014\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2015\s*\d*\.?\d+$",
    ]

    for pattern in range_patterns:
        try:
            if re.match(pattern, str(value)):
                return True
        except Exception as e:
            logger.error(f"Error while checking range. value: {value}, error: {e}")
            continue
    return False


def return_range_indices(values: list[str]) -> list[int]:
    range_indices = []
    for index, value in enumerate(values):
        if check_if_range(value):
            range_indices.append(index)
    return range_indices


def check_if_comma_or_space_separated_numbers(value: str) -> bool:
    """
    Check if a value contains MULTIPLE comma or space separated numbers.

    Returns True only if there are 2+ numbers separated by comma or space.
    Single numbers return False.

    Args:
        value: The value to check

    Returns:
        bool: True if the value has multiple numbers separated by comma/space
    """
    try:
        value = str(value).strip()
        # Pattern requires at least one separator followed by another number
        # This ensures we have 2+ numbers (uses + not * for the group)
        pattern = r"^\s*-?\d+\.?\d*(\s*[,\s]\s*-?\d+\.?\d*\s*)+\s*,?\s*$"
        if re.match(pattern, value):
            return True
        return False
    except Exception as e:
        logger.error(
            f"Error while checking comma or space separated numbers. "
            f"value: {value}, error: {e}"
        )
        return False


def check_if_number_in_string(value: str) -> str:
    """
    check if a value is a whole number when
    its datatype should be string.
    return True if it is a whole number.
    return False if it is not a whole number.
    """
    value = replace_nans_with_na(value)
    try:
        float(value)
        return True
    except ValueError:
        return False


def check_if_string_in_number(value: str) -> bool:
    """
    check if a value is a string when its datatype should be number.
    return True if it is a string.
    return False if it is not a string.
    """
    if (
        check_if_null(value)
        or check_if_range(value)
        or check_if_comma_or_space_separated_numbers(value)
    ):
        return False
    try:
        float(value)
        return False
    except ValueError:
        return True


def preprocess_string(value: str) -> str:
    value = str(value)
    value = standardize_word(value)
    value = value.strip()
    value = value.lower()
    return value


def check_if_unit_starts_with_number(value: str) -> bool:
    if re.match(r"^\d", value):
        return True
    return False


def clean_numerical_columns(
    df: pd.DataFrame, numerical_columns: list[str]
) -> pd.DataFrame:
    def _clean_function(value: str | float | int | None) -> int | float | None:
        if check_if_null(value):
            return None
        if isinstance(value, str):
            value = value.strip()
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Value {value} is not a number")
        return value

    for column in numerical_columns:
        if column in df.columns:
            try:
                df[column] = df[column].apply(_clean_function)
            except Exception as e:
                logger.error(f"Error while cleaning {column}: {e}")
                continue
    return df


def break_comma_separated_numbers(value: str) -> list[float]:
    return [float(number.strip()) for number in value.split(",")]


def validate_var_column_values(value: str) -> tuple[bool, str]:
    """
    Validate individual items in _VAR columns that may contain
    comma-separated or range values.

    Note: check_if_range and check_if_comma_or_space_separated_numbers
    already validate that parts are valid numbers through their regex patterns,
    so no additional validation is needed for those cases.

    Args:
        value: The value to validate (may be comma/hyphen separated)

    Returns:
        tuple: (is_valid, cleaned_value)
            - is_valid: True if the value is valid (null, range, comma/space
              separated numbers, or single number)
            - cleaned_value: The original value as string (unchanged), or None if null
    """
    if check_if_null(value):
        return True, None

    value_str = str(value).strip()

    # check_if_range regex patterns (e.g., \d*\.?\d+) already validate
    # that both parts are valid numbers
    if check_if_range(value_str):
        return True, value_str

    # check_if_comma_or_space_separated_numbers regex (-?\d+\.?\d*)
    # already validates that all parts are valid numbers
    if check_if_comma_or_space_separated_numbers(value_str):
        return True, value_str

    # Default handling for single values: try to convert to float
    try:
        float(value_str)
        return True, value_str
    except ValueError:
        return False, value_str


def is_var_column(column_name: str) -> bool:
    """
    Check if a column name ends with _VAR (case insensitive).

    Args:
        column_name: The column name to check

    Returns:
        bool: True if the column name ends with _VAR
    """
    return column_name.upper().endswith("_VAR")


def download_file_from_url(url: str) -> tuple[str, bytes]:
    """Download a file from a URL.

    Args:
        url (str): The URL of the file to download.

    Returns:
        tuple[str, bytes]: The file type and the file content in bytes
    """
    try:
        with httpx.Client(timeout=30) as client:
            downloaded_file = client.get(url)
            file_type = downloaded_file.headers["Content-Type"]
            file_content = downloaded_file.content
            return file_type, file_content
    except Exception as e:
        print(f"Error downloading file from URL: {url}")
        print(e)
        return None, None


def expand_plot_points(df: pd.DataFrame) -> pd.DataFrame:
    def expand_points(row):
        """Expand each point in the points column into a separate row"""
        points_str = row["points"]

        # Parse the JSON string
        try:
            points = json.loads(points_str)
        except Exception:
            return None

        expanded_rows = []

        for point in points:
            # Create a new row based on the original row
            new_row = row.copy()

            # Extract UCI, LCI, VAR from point
            new_row["UCI"] = point.get("uci")
            new_row["LCI"] = point.get("lci")
            new_row["VAR"] = point.get("var")

            # Get x_cat and y_cat from extraColumns
            extra_cols = point.get("extraColumns", {})
            x_cat = extra_cols.get("x_cat", "")
            y_cat = extra_cols.get("y_cat", "")

            # Modify ARM_TIME: use x_cat if not empty, otherwise use x
            if x_cat and x_cat.strip():
                new_row["ARM_TIME"] = x_cat
            else:
                new_row["ARM_TIME"] = point.get("x")

            # Modify DV: use y_cat if not empty, otherwise use y
            if y_cat and y_cat.strip():
                new_row["DV"] = y_cat
            else:
                new_row["DV"] = point.get("y")

            expanded_rows.append(new_row)

        return expanded_rows

    # Process each row and expand points
    if "ARM_TIME" in df.columns:
        df.drop(columns=["ARM_TIME"], inplace=True)
    if "DV" in df.columns:
        df.drop(columns=["DV"], inplace=True)
    all_expanded_rows = []
    for _, row in df.iterrows():
        expanded = expand_points(row)
        if expanded:
            all_expanded_rows.extend(expanded)

    # Create new dataframe from expanded rows
    df_expanded = pd.DataFrame(all_expanded_rows)

    # Reset index
    df_expanded.reset_index(drop=True, inplace=True)

    # remove points column
    df_expanded.drop(columns=["points"], inplace=True)

    return df_expanded
